"""Real-time VOGeo-Gaze: one frame at a time, with a running eyeball centre.

The offline engine runs 16-frame clips and takes the median of the clip centres as
the eyeball centre of the whole recording.  Live, each new frame goes through the
per-frame part of the network (segmentation, ellipse fits, refraction correction,
closed-form reconstruction) right away.  Its features are also pushed into a ring
buffer at ~12 Hz (the rate F_tem was trained at); every few pushes F_tem runs on the
last 16 and gives a clip centre.  The median of the recent clip centres is the
running eyeball centre, and each frame's gaze is its corrected pupil ray lifted
onto the eyeball sphere about it, as in runtime/center.py.

For a trial the centre is frozen at the start of calibration, so the calibration
mapping and the experiment use the same eye model.
"""
import collections
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

from vogeo_gaze import load_model
from vogeo_gaze.geometry.anatomy import R_EYE, R_IRIS
from vogeo_gaze.geometry.camera import pixel_to_ray
from vogeo_gaze.geometry.ellipse import el2_theta, el2_to_el, minor_axis
from vogeo_gaze.models.reconstruction import reconstruct_eye
from vogeo_gaze.runtime.center import relift
from vogeo_gaze.runtime.overlay import blend_segmentation
from vogeo_gaze.runtime.records import BLINK_OPEN_EYE_SCORE
from vogeo_gaze.runtime.videoio import letterbox, to_canvas

L_P = float(np.sqrt(R_EYE ** 2 - R_IRIS ** 2))
REFINER_HZ = 12.0     # rate F_tem was trained at
CLIP_FRAMES = 16      # frames per F_tem clip
CLIP_EVERY = 4        # run F_tem every 4 pushes (~3 Hz)
MAX_CLIPS = 24        # clip centres kept for the running median (~8 s)


def _frame_stage(model, x):
    """Per-frame part of VOGeoGaze.forward on (N,3,H,W): everything before F_tem."""
    bottleneck, skips = model.resunet.encode(x)
    skips.reverse()
    logits = model.resunet.decode(bottleneck, skips)
    fit = model.ellipse_fit(torch.sigmoid(logits))
    pooled = torch.cat([bottleneck.mean(dim=(-2, -1)), bottleneck.amax(dim=(-2, -1))], dim=-1)
    z_spt, z_tmp = model.z_spt(pooled), model.z_tmp(pooled)
    direction = F.normalize(model.f_dir(z_spt), dim=-1, eps=1e-6)
    ent_el2, iris_el2 = fit["entpup_el2"], fit["iris_el2"]
    pupil_conf, iris_conf = fit["pupil_conf"].reshape(-1, 1), fit["iris_conf"].reshape(-1, 1)
    axis = minor_axis(el2_theta(ent_el2))
    g = axis * torch.where((axis * direction).sum(dim=-1, keepdim=True) >= 0.0, 1.0, -1.0).to(axis.dtype)
    pup_el2 = model.f_ref(ent_el2, -g, iris_el2[:, 2:3])
    c_eye, gaze, c_pupil, r_pupil = reconstruct_eye(pup_el2, iris_el2, g, model.fpx, model.img_size)
    geom = {"c_eye": c_eye, "gaze": gaze, "c_pupil": c_pupil, "r_pupil": r_pupil,
            "pupil_el2": pup_el2, "entpup_el2": ent_el2, "iris_el2": iris_el2,
            "pupil_conf": pupil_conf, "iris_conf": iris_conf}
    return logits, fit, z_tmp, geom


def cuda_devices():
    """[(torch device string, name)] of the NVIDIA GPUs usable by this torch build."""
    if not torch.cuda.is_available():
        return []
    return [(f"cuda:{i}", torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]


def resolve_device(device):
    if device in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU NVIDIA non disponibile: installare PyTorch con CUDA (vedi README) "
                           "o scegliere CPU")
    return device


def torch_build_info():
    """Short description of the installed torch build, for the GUI."""
    cuda = torch.version.cuda
    return f"PyTorch {torch.__version__}" + (f" · CUDA {cuda}" if cuda else " · solo CPU")


def angles(gaze):
    """(N,3) unit gaze -> (N,2) theta_H, theta_V in degrees (same convention as gaze.csv)."""
    return np.stack([np.degrees(np.arctan2(gaze[:, 0], -gaze[:, 2])),
                     np.degrees(np.arcsin(np.clip(gaze[:, 1], -1, 1)))], axis=1)


class RealtimeGaze:
    """Stateful single-frame VOGeo-Gaze."""

    def __init__(self, model=None, device=None, conf_threshold=0.5, fp16=False):
        """``device``: None/'auto' (CUDA when available), 'cuda', 'cuda:N' or 'cpu'.
        ``fp16``: autocast to half precision on NVIDIA GPUs (ignored on CPU)."""
        device = resolve_device(device)
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True   # fixed 240x320 input: pick the fastest kernels once
        self.model = model or load_model(device=device)
        self.device = next(self.model.parameters()).device
        self.fp16 = bool(fp16) and self.device.type == "cuda"
        self.img_size = self.model.img_size
        self.conf_threshold = conf_threshold
        self.reset_center()

    @property
    def device_label(self):
        if self.device.type == "cuda":
            name = torch.cuda.get_device_name(self.device)
            return f"{name}{' · fp16' if self.fp16 else ''}"
        return f"CPU · {torch.get_num_threads()} thread"

    # ---- eyeball centre ----
    def reset_center(self):
        self._buffer = collections.deque(maxlen=CLIP_FRAMES)
        self._pushes, self._last_push = 0, -1e9
        self._clip_centers = collections.deque(maxlen=MAX_CLIPS)
        self._frame_centers = collections.deque(maxlen=90)
        self.frozen_center = None

    def running_center(self):
        if self._clip_centers:
            return np.median(np.stack(self._clip_centers), axis=0)
        if len(self._frame_centers) >= 8:
            return np.median(np.stack(self._frame_centers), axis=0)
        return None

    def center(self):
        return self.frozen_center if self.frozen_center is not None else self.running_center()

    def center_state(self):
        """('bloccato' | 'stimato' | 'provvisorio' | 'assente', number of clip centres)."""
        if self.frozen_center is not None:
            return "bloccato", len(self._clip_centers)
        if self._clip_centers:
            return "stimato", len(self._clip_centers)
        return ("provvisorio" if len(self._frame_centers) >= 8 else "assente"), 0

    def freeze(self):
        c = self.running_center()
        if c is not None:
            self.frozen_center = c.copy()
        return self.frozen_center

    def unfreeze(self):
        self.frozen_center = None

    def _update_center(self, z_tmp, geom, t, trusted):
        if trusted:
            c = geom["c_eye"][0].float().cpu().numpy()
            if np.isfinite(c).all():
                self._frame_centers.append(c)
        # Blink / low-confidence frames would only dilute the 16-frame F_tem window.
        if not trusted or t - self._last_push < 1.0 / REFINER_HZ:
            return
        self._last_push = t
        self._buffer.append((z_tmp[0], {k: v[0] for k, v in geom.items()}))
        self._pushes += 1
        if len(self._buffer) == CLIP_FRAMES and self._pushes % CLIP_EVERY == 0:
            z = torch.stack([b[0] for b in self._buffer])[None]
            g = {k: torch.stack([b[1][k] for b in self._buffer])[None] for k in geom}
            c = self.model.f_tem(z, g)["c_eye_clip"][0].float().cpu().numpy()
            if np.isfinite(c).all():
                self._clip_centers.append(c)

    # ---- one frame ----
    @torch.no_grad()
    def process(self, gray, t=None, want_seg=True):
        """(240,320) uint8 canvas frame -> dict of per-frame results (numpy / floats)."""
        t = time.perf_counter() if t is None else t
        x = torch.from_numpy(gray).to(self.device, non_blocking=True).float().mul_(1.0 / 255.0)
        x = x[None, None].repeat(1, 3, 1, 1)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
            logits, fit, z_tmp, geom = _frame_stage(self.model, x)
        geom = {k: v.float() for k, v in geom.items()}
        z_tmp = z_tmp.float()

        pupil_conf = float(geom["pupil_conf"][0, 0])
        iris_conf = float(geom["iris_conf"][0, 0])
        open_eye = float(fit["open_eye_score"].reshape(-1)[0])
        trusted = (pupil_conf >= self.conf_threshold and iris_conf >= self.conf_threshold
                   and open_eye >= BLINK_OPEN_EYE_SCORE)
        self._update_center(z_tmp, geom, t, trusted)

        gaze0 = F.normalize(geom["gaze"], dim=-1).float().cpu().numpy()
        center = self.center()
        if center is not None:
            ray = pixel_to_ray(geom["pupil_el2"][:, :2], self.model.fpx, self.img_size).float().cpu().numpy()
            gaze, _, hit = relift(ray, gaze0, center, L_P)
            gaze_ok = bool(hit[0])
        else:
            gaze, gaze_ok = gaze0, False
        th = angles(gaze)[0]
        finite = bool(np.isfinite(th).all())
        valid = (finite and gaze_ok and trusted and open_eye >= BLINK_OPEN_EYE_SCORE)
        out = {
            "t": t, "theta_h": float(th[0]), "theta_v": float(th[1]), "valid": valid, "gaze_ok": gaze_ok,
            "pupil_conf": pupil_conf, "iris_conf": iris_conf, "open_eye": open_eye,
            "r_pupil": float(geom["r_pupil"][0, 0]),
            "entpup_el": el2_to_el(geom["entpup_el2"])[0].float().cpu().numpy(),
            "pupil_el": el2_to_el(geom["pupil_el2"])[0].float().cpu().numpy(),
            "iris_el": el2_to_el(geom["iris_el2"])[0].float().cpu().numpy(),
        }
        if want_seg:
            out["seg"] = blend_segmentation(logits, torch.from_numpy(gray)[None].to(self.device))[0].cpu().numpy()
        return out


class TrackerThread:
    """Pulls frames from a source, runs RealtimeGaze, keeps the latest result and
    hands every result to the registered listeners (called from this thread)."""

    def __init__(self, gaze: RealtimeGaze, source, flip_h=False, flip_v=False):
        self.gaze, self.source = gaze, source
        self.flip_h, self.flip_v = flip_h, flip_v
        self.latest, self.error = None, None
        self.listeners = []
        self.cam_fps = self.model_fps = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="tracker")

    def start(self):
        self.source.start()
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)
        self.source.stop()

    @property
    def running(self):
        return self._thread.is_alive()

    def _run(self):
        h, w = self.gaze.img_size
        lb, lb_shape, last_idx = None, None, -1
        times = collections.deque(maxlen=30)
        try:
            while not self._stop.is_set():
                item = self.source.read(after=last_idx, timeout=0.5)
                if item is None:
                    if self.source.finished:
                        break
                    continue
                frame, t, last_idx = item
                if lb_shape != frame.shape[:2]:
                    lb_shape = frame.shape[:2]
                    lb = letterbox(frame.shape[1], frame.shape[0], w, h)
                gray = to_canvas(frame, lb)
                if self.flip_h or self.flip_v:
                    gray = np.ascontiguousarray(gray[::-1 if self.flip_v else 1, ::-1 if self.flip_h else 1])
                res = self.gaze.process(gray, t)
                res["gray"] = gray
                times.append(time.perf_counter())
                if len(times) > 1:
                    self.model_fps = (len(times) - 1) / max(times[-1] - times[0], 1e-6)
                self.cam_fps = self.source.fps_measured
                self.latest = res
                for cb in list(self.listeners):
                    cb(res)
        except BaseException as exc:  # shown in the GUI
            self.error = exc
