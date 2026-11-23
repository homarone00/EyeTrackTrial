"""Video inputs: webcams / eye cameras, video files and a synthetic eye for testing.

Every source runs its own grab thread and keeps only the newest frame, so a slow
network never builds up latency: ``read(after=idx)`` waits for a frame newer than
``idx`` and returns ``(frame, t_capture, idx)``.
"""
import collections
import sys
import threading
import time

import cv2
import numpy as np

SIM_NAME = "Simulatore (test)"
FILE_NAME = "File video..."


def list_cameras(max_probe=6):
    """[(index, name)] of the video inputs.  DirectShow names through pygrabber when
    available (same index order as cv2.CAP_DSHOW), otherwise probed indices."""
    if sys.platform == "win32":
        try:
            from pygrabber.dshow_graph import FilterGraph
            return list(enumerate(FilterGraph().get_input_devices()))
        except Exception:
            pass
    found = []
    for i in range(max_probe):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        if cap.isOpened():
            found.append((i, f"Camera {i}"))
        cap.release()
    return found


class _Latest:
    """Newest frame and a measured frame rate, shared by the grab thread and readers."""

    def __init__(self):
        self._cond = threading.Condition()
        self._item, self._idx = None, -1
        self._times = collections.deque(maxlen=30)
        self.finished = False

    def put(self, frame, t):
        with self._cond:
            self._idx += 1
            self._item = (frame, t, self._idx)
            self._times.append(t)
            self._cond.notify_all()

    def finish(self):
        with self._cond:
            self.finished = True
            self._cond.notify_all()

    def read(self, after=-1, timeout=0.5):
        with self._cond:
            self._cond.wait_for(lambda: self._idx > after or self.finished, timeout=timeout)
            return self._item if self._idx > after else None

    @property
    def fps(self):
        t = self._times
        return (len(t) - 1) / max(t[-1] - t[0], 1e-6) if len(t) > 1 else 0.0


class _ThreadedSource:
    label = "sorgente"

    def __init__(self):
        self._latest = _Latest()
        self._stop = threading.Event()
        self._thread = None
        self.error = None

    def start(self):
        self._open()
        self._thread = threading.Thread(target=self._loop, daemon=True, name=self.label)
        self._thread.start()

    def _loop(self):
        try:
            while not self._stop.is_set():
                frame = self._grab()
                if frame is None:
                    break
                self._latest.put(frame, time.perf_counter())
        except BaseException as exc:
            self.error = exc
        finally:
            self._latest.finish()
            self._close()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)

    def read(self, after=-1, timeout=0.5):
        return self._latest.read(after, timeout)

    @property
    def finished(self):
        return self._latest.finished

    @property
    def fps_measured(self):
        return self._latest.fps

    def _open(self):
        pass

    def _close(self):
        pass


class CameraSource(_ThreadedSource):
    def __init__(self, index, name="", resolution=None, fps=None):
        super().__init__()
        self.index, self.label, self.resolution, self.fps = index, name or f"Camera {index}", resolution, fps
        self.cap = None
        self.size = None

    def _open(self):
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.index, backend)
        if not self.cap.isOpened():
            raise IOError(f"impossibile aprire {self.label}")
        if self.resolution:
            # MJPG lets most USB cameras deliver higher resolutions at full rate.
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        if self.fps:
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        ok, frame = self.cap.read()
        if not ok:
            self.cap.release()
            raise IOError(f"{self.label}: nessun fotogramma")
        self.size = (frame.shape[1], frame.shape[0])

    def _grab(self):
        ok, frame = self.cap.read()
        if not ok:
            time.sleep(0.01)
            return None if self._stop.is_set() else self._grab()
        return frame

    def _close(self):
        if self.cap is not None:
            self.cap.release()


class VideoFileSource(_ThreadedSource):
    """Plays a recorded eye video in real time, looping."""

    def __init__(self, path, loop=True):
        super().__init__()
        self.path, self.loop, self.label = str(path), loop, str(path).replace("\\", "/").rsplit("/", 1)[-1]
        self.cap, self.size, self._next = None, None, None

    def _open(self):
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise IOError(f"impossibile aprire {self.path}")
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.period = 1.0 / (fps if fps > 0 else 30.0)
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def _grab(self):
        now = time.perf_counter()
        self._next = now if self._next is None else self._next
        if self._next > now:
            time.sleep(self._next - now)
        self._next = max(self._next + self.period, time.perf_counter() - self.period)
        ok, frame = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return frame if ok else None

    def _close(self):
        if self.cap is not None:
            self.cap.release()


class TargetBus:
    """Where the (simulated) participant should look: normalised screen position in
    [-1, 1] (x right, y down) or None, with its history for a reaction delay."""

    def __init__(self):
        self._hist = collections.deque(maxlen=400)
        self.set(None)

    def set(self, xy):
        self._hist.append((time.perf_counter(), None if xy is None else (float(xy[0]), float(xy[1]))))

    def get(self, delay=0.0):
        t = time.perf_counter() - delay
        value = self._hist[0][1]
        for ts, xy in self._hist:
            if ts > t:
                break
            value = xy
        return value


class SimulatedEyeSource(_ThreadedSource):
    """Synthetic close-up eye (320x240, 60 Hz) whose gaze follows ``TargetBus``.

    The iris and pupil are 3-D circles on a 12 mm eyeball 130 mm from a pinhole
    camera with the model's focal length, projected exactly; there is no corneal
    refraction, so the model's absolute angles are biased, which calibration absorbs.
    Without a target the eye makes small random fixations around the centre.
    """
    label = "simulatore"
    FPX, W, H = 1066.6666666666667, 320, 240
    MAX_YAW, MAX_PITCH = 20.0, 13.0   # degrees at the screen edges

    def __init__(self, bus: TargetBus, fps=60.0, noise_deg=0.15, seed=0, blink_every=(2.5, 6.0),
                 blink_dur=0.22):
        super().__init__()
        self.bus, self.period, self.noise = bus, 1.0 / fps, noise_deg
        self.blink_every, self.blink_dur = blink_every, blink_dur   # blink_every=None: no blinks
        self._blink_at = None
        self.rng = np.random.default_rng(seed)
        self.size = (self.W, self.H)
        self._yaw = self._pitch = 0.0
        self._idle_target, self._idle_until = (0.0, 0.0), 0.0
        self._next = None
        self._texture = self._make_texture()

    def _make_texture(self):
        rng = np.random.default_rng(1)
        ang = np.linspace(0, 2 * np.pi, 720, endpoint=False)
        return 1.0 + 0.18 * np.convolve(np.r_[rng.standard_normal(720), rng.standard_normal(20)],
                                        np.ones(9) / 9, mode="same")[:720], ang

    def _circle(self, center, normal, radius, n=96):
        normal = normal / np.linalg.norm(normal)
        a = np.cross(normal, [0.0, 1.0, 0.0])
        a /= np.linalg.norm(a)
        b = np.cross(normal, a)
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        p = center + radius * (np.cos(ang)[:, None] * a + np.sin(ang)[:, None] * b)
        uv = p[:, :2] / p[:, 2:3] * self.FPX + [(self.W - 1) / 2, (self.H - 1) / 2]
        return uv

    def _grab(self):
        now = time.perf_counter()
        self._next = now if self._next is None else self._next
        if self._next > now:
            time.sleep(self._next - now)
        self._next = max(self._next + self.period, time.perf_counter() - self.period)

        target = self.bus.get(delay=0.18)          # saccade latency
        if target is None:
            if now > self._idle_until:
                self._idle_target = tuple(self.rng.uniform(-0.5, 0.5, 2))
                self._idle_until = now + self.rng.uniform(0.6, 1.5)
            target = self._idle_target
        yaw_t, pitch_t = target[0] * self.MAX_YAW, target[1] * self.MAX_PITCH
        k = 0.45                                    # fast first-order approach
        self._yaw += k * (yaw_t - self._yaw)
        self._pitch += k * (pitch_t - self._pitch)
        yaw = np.radians(self._yaw + self.rng.normal(0, self.noise))
        pitch = np.radians(self._pitch + self.rng.normal(0, self.noise))
        # Participant faces the camera: looking at the screen centre = gaze (0, 0, -1).
        gaze = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), -np.cos(yaw) * np.cos(pitch)])
        return self._render(np.array([1.0, 2.0, 130.0]), gaze, self._lid(now))

    def _lid(self, now):
        """Eyelid closure in [0, 1]: random blinks, closing fast and opening slower."""
        if not self.blink_every:
            return 0.0
        if self._blink_at is None:
            self._blink_at = now + self.rng.uniform(*self.blink_every)
        p = (now - self._blink_at) / self.blink_dur
        if p >= 1.0:
            self._blink_at = now + self.rng.uniform(*self.blink_every)
            return 0.0
        if p < 0:
            return 0.0
        return min(1.0, p / 0.35) if p < 0.35 else max(0.0, 1.0 - (p - 0.55) / 0.45) if p > 0.55 else 1.0

    def _render(self, c_eye, gaze, lid=0.0):
        W, H = self.W, self.H
        img = np.full((H, W), 118, np.uint8)
        # Eye opening (almond), sclera inside.
        xs = np.linspace(-1, 1, 64)
        top = np.stack([W / 2 + xs * 150, H / 2 - 4 - 92 * (1 - xs ** 2) ** 0.8], 1)
        bot = np.stack([W / 2 - xs * 150, H / 2 + 6 + 78 * (1 - xs ** 2) ** 0.9], 1)
        opening = np.concatenate([top, bot]).astype(np.int32)
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, [opening], 255)
        eye = np.full_like(img, 205)
        limbus_c = c_eye + L_P_SIM * gaze
        iris = self._circle(limbus_c, gaze, 6.0)
        cv2.fillPoly(eye, [np.round(iris).astype(np.int32)], 95)
        # Radial iris texture.
        tex, ang = self._texture
        cx, cy = iris.mean(0)
        for r_frac in (0.45, 0.7, 0.9):
            pts = (np.stack([cx + (iris[:, 0] - cx) * r_frac, cy + (iris[:, 1] - cy) * r_frac], 1))
            cv2.polylines(eye, [np.round(pts).astype(np.int32)], True, int(85 + 25 * tex[0]), 1, cv2.LINE_AA)
        pupil = self._circle(c_eye + (L_P_SIM + 0.5) * gaze, gaze, 2.0)
        cv2.fillPoly(eye, [np.round(pupil).astype(np.int32)], 18, cv2.LINE_AA)
        # Corneal reflex.
        cv2.circle(eye, (int(cx + 9), int(cy - 10)), 3, 250, -1, cv2.LINE_AA)
        if lid > 0:
            # Upper lid comes down over the eye: skin above a curve that moves to the lower lid.
            lid_y = top[:, 1] + lid * (bot[::-1, 1] - top[:, 1])
            lid_poly = np.concatenate([np.stack([top[:, 0], lid_y], 1), top[::-1]]).astype(np.int32)
            cv2.fillPoly(mask, [lid_poly], 0)
        img = np.where(mask > 0, eye, img)
        cv2.polylines(img, [opening], True, 60, 2, cv2.LINE_AA)
        if lid > 0:
            cv2.polylines(img, [np.stack([top[:, 0], lid_y], 1).astype(np.int32)], False, 60, 2, cv2.LINE_AA)
        img = cv2.GaussianBlur(img, (5, 5), 1.1)
        noise = self.rng.normal(0, 3.0, img.shape)
        return np.clip(img + noise, 0, 255).astype(np.uint8)


L_P_SIM = float(np.sqrt(12.0 ** 2 - 6.0 ** 2))
