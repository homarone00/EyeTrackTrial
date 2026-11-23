"""Participant screen: calibration and experiment sequences in a fullscreen Tk window.

A sequence is a list of steps.  Each step lasts ``dur`` seconds and draws one
stimulus; steps with an analysis window (``a0`` seconds after onset to the end)
become events that are later matched with the gaze samples by timestamp
(time.perf_counter, the same clock the camera threads stamp frames with).

Target types of the experiment (``TYPE_LABELS``):
  punto_piccolo / punto_grande   static dot, two sizes
  saccade                        central cross, then a jump to a peripheral dot
  inseguimento                   dot moving on an ellipse (smooth pursuit)
  lettera                        a letter to fixate
  immagine                       a small picture (EyeTrackTrial/stimuli/*.png|jpg or built-in icons)
"""
import math
import random
import string
import time
from pathlib import Path

import tkinter as tk

HERE = Path(__file__).resolve().parent
STIM_DIR = HERE / "stimuli"

BG = "#808080"
FG = "#000000"
TYPE_LABELS = {
    "punto_piccolo": "Punto piccolo",
    "punto_grande": "Punto grande",
    "saccade": "Saccade",
    "inseguimento": "Inseguimento",
    "lettera": "Lettera",
    "immagine": "Immagine",
}
ICON_COLORS = ["#ef4444", "#2563eb", "#16a34a", "#f59e0b", "#a855f7", "#0ea5e9"]
ICON_SHAPES = ["stella", "cuore", "casa", "rombo", "fiore", "triangolo"]


# ---- sequences --------------------------------------------------------------------

def region_cells(width, height, margin):
    """3x3 grid cells (x0, y0, x1, y1) inside the margins, row-major."""
    xs = [margin + i * (width - 2 * margin) / 3 for i in range(4)]
    ys = [margin + i * (height - 2 * margin) / 3 for i in range(4)]
    return [(xs[c], ys[r], xs[c + 1], ys[r + 1]) for r in range(3) for c in range(3)]


def _balanced_positions(n, width, height, rng, margin, exclude_center=False):
    """n random positions spread evenly over the 3x3 regions."""
    cells = region_cells(width, height, margin)
    if exclude_center:
        cells = cells[:4] + cells[5:]
    order = []
    while len(order) < n:
        block = list(range(len(cells)))
        rng.shuffle(block)
        order += block
    pos = []
    for k in order[:n]:
        x0, y0, x1, y1 = cells[k]
        pad_x, pad_y = (x1 - x0) * 0.12, (y1 - y0) * 0.12
        pos.append((rng.uniform(x0 + pad_x, x1 - pad_x), rng.uniform(y0 + pad_y, y1 - pad_y)))
    return pos


def calibration_sequence(points, dwell=1.5, shrink=0.6):
    """One step per calibration point (already ordered); samples from the end of the
    shrinking animation + 0.15 s to the end of the step."""
    steps = [{"kind": "blank", "dur": 0.5}]
    for i, (x, y) in enumerate(points):
        steps.append({"kind": "cal", "dur": shrink + dwell, "x": x, "y": y, "shrink": shrink, "group": i,
                      "event": {"fase": "calibrazione", "punto": i, "tipo": "calibrazione"},
                      "a0": shrink + 0.15})
    return steps


def experiment_sequence(counts, width, height, fix_dur=1.5, seed=None, margin_frac=0.06):
    """Shuffled trials; ``counts``: {type: n trials}."""
    rng = random.Random(seed)
    margin = margin_frac * min(width, height)
    trials = []
    for kind in ("punto_piccolo", "punto_grande", "lettera", "immagine"):
        for x, y in _balanced_positions(counts.get(kind, 0), width, height, rng, margin):
            trials.append({"tipo": kind, "x": x, "y": y})
    for x, y in _balanced_positions(counts.get("saccade", 0), width, height, rng, margin, exclude_center=True):
        trials.append({"tipo": "saccade", "x": x, "y": y})
    for _ in range(counts.get("inseguimento", 0)):
        trials.append({"tipo": "inseguimento"})
    rng.shuffle(trials)

    letters = [c for c in string.ascii_uppercase if c not in "IJOQ"]
    images = list_images()
    steps = [{"kind": "blank", "dur": 0.5}]
    for n, tr in enumerate(trials):
        ev = {"fase": "esperimento", "prova": n, "tipo": tr["tipo"]}
        kind = tr["tipo"]
        first = len(steps)
        if kind in ("punto_piccolo", "punto_grande"):
            r = 5 if kind == "punto_piccolo" else 18
            steps.append({"kind": "dot", "dur": fix_dur, "x": tr["x"], "y": tr["y"], "r": r,
                          "event": ev, "a0": min(0.5, fix_dur / 2)})
        elif kind == "lettera":
            steps.append({"kind": "letter", "dur": fix_dur, "x": tr["x"], "y": tr["y"],
                          "text": rng.choice(letters), "event": ev, "a0": min(0.5, fix_dur / 2)})
        elif kind == "immagine":
            img = rng.choice(images) if images else None
            steps.append({"kind": "image", "dur": fix_dur, "x": tr["x"], "y": tr["y"], "file": img,
                          "shape": rng.choice(ICON_SHAPES), "color": rng.choice(ICON_COLORS),
                          "event": ev, "a0": min(0.6, fix_dur / 2)})
        elif kind == "saccade":
            steps.append({"kind": "cross", "dur": rng.uniform(0.8, 1.2), "x": width / 2, "y": height / 2})
            steps.append({"kind": "dot", "dur": 1.2, "x": tr["x"], "y": tr["y"], "r": 10,
                          "event": ev, "a0": 0.45})
        elif kind == "inseguimento":
            rx, ry = 0.36 * width, 0.36 * height
            period = rng.uniform(5.0, 7.0)
            direction = rng.choice([1, -1])
            phase0 = rng.uniform(0, 2 * math.pi)
            path = {"cx": width / 2, "cy": height / 2, "rx": rx, "ry": ry,
                    "omega": direction * 2 * math.pi / period, "phase0": phase0}
            x0, y0 = pursuit_position(path, 0.0)
            steps.append({"kind": "dot", "dur": 0.8, "x": x0, "y": y0, "r": 10})
            steps.append({"kind": "pursuit", "dur": period, "path": path, "r": 10,
                          "event": dict(ev, path=path), "a0": 0.6})
        steps.append({"kind": "blank", "dur": 0.15})
        for st in steps[first:]:
            st["group"] = n
    return steps


def repeat_sequence(steps, groups, attempt):
    """Copies of the steps of the given trial / calibration-point ``groups`` (with
    their fixation and blank steps), events tagged with ``tentativo``."""
    out = [{"kind": "blank", "dur": 0.6}]
    for st in steps:
        if st.get("group") in groups:
            new = {k: v for k, v in st.items() if not k.startswith("_")}
            if "event" in new:
                new["event"] = dict(new["event"], tentativo=attempt)
            out.append(new)
            if st["kind"] == "cal":
                out.append({"kind": "blank", "dur": 0.2})
    return out


def pursuit_position(path, t):
    a = path["phase0"] + path["omega"] * t
    return path["cx"] + path["rx"] * math.cos(a), path["cy"] + path["ry"] * math.sin(a)


def list_images():
    if not STIM_DIR.is_dir():
        return []
    return sorted(str(p) for p in STIM_DIR.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif"))


# ---- drawing ------------------------------------------------------------------------

def _icon(canvas, shape, x, y, s, color):
    """Simple built-in pictures, about 2s wide, centred on (x, y)."""
    if shape == "stella":
        pts = []
        for k in range(10):
            r = s if k % 2 == 0 else s * 0.45
            a = -math.pi / 2 + k * math.pi / 5
            pts += [x + r * math.cos(a), y + r * math.sin(a)]
        canvas.create_polygon(pts, fill=color, outline="")
    elif shape == "cuore":
        pts = []
        for k in range(60):
            t = 2 * math.pi * k / 60
            px = 16 * math.sin(t) ** 3
            py = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
            pts += [x + px * s / 17, y - py * s / 17]
        canvas.create_polygon(pts, fill=color, outline="", smooth=True)
    elif shape == "casa":
        canvas.create_rectangle(x - s * 0.7, y - s * 0.1, x + s * 0.7, y + s, fill=color, outline="")
        canvas.create_polygon(x - s, y - s * 0.05, x, y - s, x + s, y - s * 0.05, fill=color, outline="")
        canvas.create_rectangle(x - s * 0.18, y + s * 0.4, x + s * 0.18, y + s, fill="white", outline="")
    elif shape == "rombo":
        canvas.create_polygon(x, y - s, x + s * 0.75, y, x, y + s, x - s * 0.75, y, fill=color, outline="")
    elif shape == "fiore":
        for k in range(6):
            a = k * math.pi / 3
            px, py = x + s * 0.55 * math.cos(a), y + s * 0.55 * math.sin(a)
            canvas.create_oval(px - s * 0.42, py - s * 0.42, px + s * 0.42, py + s * 0.42, fill=color, outline="")
        canvas.create_oval(x - s * 0.35, y - s * 0.35, x + s * 0.35, y + s * 0.35, fill="#fde047", outline="")
    else:  # triangolo
        canvas.create_polygon(x, y - s, x + s * 0.95, y + s * 0.75, x - s * 0.95, y + s * 0.75,
                              fill=color, outline="")


class StimulusWindow:
    """Fullscreen window on one monitor that plays sequences.

    ``on_target(xy_norm | None)`` is called with the current target position in
    [-1, 1] (used by the eye simulator).  ``Esc`` aborts; ``Space`` starts after
    the instruction screen.
    """

    TICK_MS = 8

    def __init__(self, root, monitor, on_target=None):
        self.root, self.on_target = root, on_target
        x, y, self.width, self.height = monitor
        self.win = tk.Toplevel(root)
        self.win.configure(background=BG, cursor="none")
        self.win.overrideredirect(True)
        self.win.geometry(f"{self.width}x{self.height}{x:+d}{y:+d}")
        self.win.attributes("-topmost", True)
        self.canvas = tk.Canvas(self.win, width=self.width, height=self.height, background=BG,
                                highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.win.bind("<Escape>", lambda e: self.abort())
        self.win.bind("<space>", lambda e: self._space())
        self.win.focus_force()
        self._images = {}
        self._after = None
        self._gen = 0
        self._wait_space = None
        self.closed = False

    # ---- control ----
    def play(self, steps, intro, on_done, on_abort):
        """Show ``intro`` until Space, then run ``steps``; ``on_done(events)``."""
        self.steps, self.on_done, self.on_abort = steps, on_done, on_abort
        self.events, self._i, self._t_step = [], -1, None
        self._message(intro + "\n\nPremere SPAZIO per iniziare  ·  ESC per interrompere")
        self._target(None)
        self._wait_space = self._start
        self.win.after(50, self.win.focus_force)

    def play_more(self, steps, text, on_done, pause=1.8):
        """Continue in the same window (e.g. repeats after blinks): ``text`` for
        ``pause`` seconds, then ``steps``; events keep accumulating."""
        self.steps, self.on_done, self._i = steps, on_done, -1
        self._message(text)
        self._target(None)
        self._after = self.win.after(int(pause * 1000), self._start)

    def message(self, text, on_space):
        self._message(text)
        self._target(None)
        self._wait_space = on_space

    def _space(self):
        if self._wait_space is not None:
            cb, self._wait_space = self._wait_space, None
            cb()

    def _start(self):
        self._next_step()
        self._tick()

    def abort(self):
        if self._after is not None:
            self.win.after_cancel(self._after)
            self._after = None
        cb = getattr(self, "on_abort", None)
        self.close()
        if cb:
            cb()

    def close(self):
        if not self.closed:
            self.closed = True
            self._target(None)
            try:
                self.win.destroy()
            except tk.TclError:
                pass

    # ---- playback ----
    def _next_step(self):
        self._i += 1
        if self._i >= len(self.steps):
            self._finish()
            return
        self._t_step = None
        self._draw(self.steps[self._i], 0.0, first=True)

    def _finish(self):
        self._after = None
        self._gen += 1
        self._target(None)
        self.canvas.delete("all")
        self.on_done(self.events)

    def _tick(self, gen=None):
        # One tick loop per sequence: a loop left over from a finished sequence stops
        # here when play_more() started a new one from inside on_done.
        if gen is None:
            gen = self._gen
        if self.closed or gen != self._gen or not 0 <= self._i < len(self.steps):
            return
        step = self.steps[self._i]
        now = time.perf_counter()
        if self._t_step is None:
            # Onset = first tick after the stimulus was drawn and flushed.
            self._t_step = now
            if "event" in step:
                ev = dict(step["event"])
                ev.update({k: step[k] for k in ("x", "y", "r") if k in step})
                ev.update({"t_on": now, "t_a0": now + step.get("a0", 0.0), "t_off": None})
                step["_ev"] = ev
                self.events.append(ev)
        elapsed = now - self._t_step
        if elapsed >= step["dur"]:
            if "_ev" in step:
                step["_ev"]["t_off"] = now
            self._next_step()
            if gen != self._gen or self.closed or self._i >= len(self.steps):
                return
        else:
            self._draw(step, elapsed, first=False)
        self._after = self.win.after(self.TICK_MS, lambda: self._tick(gen))

    def _target(self, xy):
        if self.on_target is not None:
            if xy is None:
                self.on_target(None)
            else:
                self.on_target((2 * xy[0] / self.width - 1, 2 * xy[1] / self.height - 1))

    def _message(self, text):
        c = self.canvas
        c.delete("all")
        c.create_text(self.width / 2, self.height / 2, text=text, fill="white", justify="center",
                      font=("Segoe UI", max(16, self.height // 40)))

    def _draw(self, step, elapsed, first):
        c, kind = self.canvas, step["kind"]
        if kind == "cal":
            frac = min(1.0, elapsed / step["shrink"])
            r = 28 - 18 * frac
            c.delete("all")
            x, y = step["x"], step["y"]
            c.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="")
            c.create_oval(x - 3, y - 3, x + 3, y + 3, fill=FG, outline="")
            if first:
                self._target((x, y))
            return
        if kind == "pursuit":
            x, y = pursuit_position(step["path"], elapsed)
            r = step["r"]
            if first:
                c.delete("all")
                step["_item"] = c.create_oval(x - r, y - r, x + r, y + r, fill=FG, outline="white", width=2)
            else:
                c.coords(step["_item"], x - r, y - r, x + r, y + r)
            self._target((x, y))
            return
        if not first:
            return
        c.delete("all")
        if kind == "blank":
            self._target(None)
            return
        x, y = step["x"], step["y"]
        if kind == "dot":
            r = step["r"]
            c.create_oval(x - r, y - r, x + r, y + r, fill=FG, outline="white", width=2 if r > 6 else 1)
            if r > 8:
                c.create_oval(x - 2, y - 2, x + 2, y + 2, fill="white", outline="")
        elif kind == "cross":
            s = 14
            c.create_line(x - s, y, x + s, y, fill="white", width=3)
            c.create_line(x, y - s, x, y + s, fill="white", width=3)
        elif kind == "letter":
            c.create_text(x, y, text=step["text"], fill=FG, font=("Segoe UI", 34, "bold"))
        elif kind == "image":
            if step.get("file"):
                img = self._load(step["file"])
                c.create_image(x, y, image=img)
            else:
                _icon(c, step["shape"], x, y, 30, step["color"])
        self._target((x, y))

    def _load(self, path, size=80):
        if path not in self._images:
            from PIL import Image, ImageTk
            im = Image.open(path).convert("RGBA")
            im.thumbnail((size, size))
            self._images[path] = ImageTk.PhotoImage(im)
        return self._images[path]
