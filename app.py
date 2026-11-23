"""EyeTrackTrial: real-time VOGeo-Gaze with a calibration + accuracy trial.

    python EyeTrackTrial/app.py

Tabs:
  Live   video input selector (eye cameras, video file, simulator), compute device
         (NVIDIA GPU / CPU), live segmentation and gaze.
  Prova  calibration on the participant screen, then the experiment; gaze error
         per target type, screen region and eccentricity, saved to risultati/.
"""
import collections
import datetime
import json
import os
import random
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent))   # vogeo_gaze from the repository, if not pip-installed

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

from theme import COLORS, FONT_FAMILY, FONT_SIZE, apply_theme
from sources import (FILE_NAME, SIM_NAME, CameraSource, SimulatedEyeSource, TargetBus, VideoFileSource,
                     list_cameras)
from blinks import LiveCounter
from stimuli import TYPE_LABELS, StimulusWindow, calibration_sequence, experiment_sequence, repeat_sequence

SETTINGS_FILE = HERE / "settings.json"
RESULTS_DIR = HERE / "risultati"
RESOLUTIONS = {"Predefinita": None, "640×480": (640, 480), "1280×720": (1280, 720), "1920×1080": (1920, 1080)}
CAL_POINTS = ["5", "9", "13", "17", "25"]
DEFAULT_COUNTS = {"punto_piccolo": 10, "punto_grande": 10, "saccade": 10, "inseguimento": 3,
                  "lettera": 8, "immagine": 8}
VIEW_SCALE = 1.2          # largest display scale of the 320x240 canvas (384x288); smaller if the window is
TRACE_SECONDS = 10.0
TYPE_COLORS = {"punto_piccolo": "#2563eb", "punto_grande": "#0ea5e9", "saccade": "#a855f7",
               "inseguimento": "#16a34a", "lettera": "#f59e0b", "immagine": "#ef4444"}


# ---- helpers ------------------------------------------------------------------------

def load_settings():
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(settings):
    try:
        SETTINGS_FILE.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def list_monitors(root):
    """[(label, (x, y, w, h))], primary first."""
    try:
        from screeninfo import get_monitors
        mons = sorted(get_monitors(), key=lambda m: not m.is_primary)
        return [(f"{i + 1}: {m.width}×{m.height}{'  (principale)' if m.is_primary else ''}",
                 (m.x, m.y, m.width, m.height)) for i, m in enumerate(mons)]
    except Exception:
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        return [(f"1: {w}×{h}  (principale)", (0, 0, w, h))]


def draw_el(img, el, color, scale, thickness=2):
    theta, cx, cy, a, b = [float(v) for v in el]
    if not np.isfinite([theta, cx, cy, a, b]).all() or a < 1 or b < 1:
        return
    cv2.ellipse(img, (int(cx * scale + 0.5), int(cy * scale + 0.5)), (int(a * scale + 0.5), int(b * scale + 0.5)),
                float(np.degrees(theta)), 0, 360, color, thickness, lineType=cv2.LINE_AA)


def fmt(v, digits=1, unit=""):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{digits}f}{unit}"


class ScrollColumn(ttk.Frame):
    """Fixed-width column that gets a vertical scrollbar only when its content is
    taller than the window (small screens, display scaling)."""

    BAR_SPACE = 20   # px always reserved for the scrollbar, so the column never changes width

    def __init__(self, parent, width):
        super().__init__(parent)
        self.width = int(width)
        self.canvas = tk.Canvas(self, width=self.width, highlightthickness=0, background=COLORS["bg"])
        self.canvas.pack(side="left", fill="y", expand=True)
        slot = ttk.Frame(self, width=self.BAR_SPACE)
        slot.pack(side="left", fill="y")
        slot.pack_propagate(False)
        self.bar = ttk.Scrollbar(slot, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.bar.set)
        self.inner = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window(0, 0, window=self.inner, anchor="nw", width=self.width)
        self.inner.bind("<Configure>", self._update)
        self.canvas.bind("<Configure>", self._update)
        self.canvas.bind_all("<MouseWheel>", self._wheel, add="+")
        self._shown = False

    def set_width(self, width):
        """Fix the column width (content is laid out to it; opening a section never widens it)."""
        self.width = int(width)
        self.canvas.configure(width=self.width)
        self.canvas.itemconfigure(self._win, width=self.width)
        self._update()

    def _update(self, _=None):
        self.canvas.configure(scrollregion=(0, 0, self.width, self.inner.winfo_reqheight()))
        overflow = self.inner.winfo_reqheight() > self.canvas.winfo_height() + 1
        if overflow and not self._shown:
            self.bar.pack(side="right", fill="y")
        elif not overflow and self._shown:
            self.bar.pack_forget()
            self.canvas.yview_moveto(0)
        self._shown = overflow

    def _wheel(self, event):
        widget = self.winfo_containing(event.x_root, event.y_root)
        if self._shown and widget is not None and str(widget).startswith(str(self)):
            self.canvas.yview_scroll(int(-event.delta / 120), "units")


class Collapsible(ttk.Frame):
    """Card with a clickable header (chevron, title, one-line summary) and a body that
    opens and closes, like the advanced-parameters card of SmartAnkle.  Closed at start."""

    def __init__(self, parent, title, on_toggle=None):
        super().__init__(parent, style="Card.TFrame")
        self.on_toggle = on_toggle
        head = ttk.Frame(self, style="Inner.TFrame", padding=(12, 8))
        head.pack(fill="x")
        self.chevron = ttk.Label(head, text="▸", style="Chevron.TLabel", width=2)
        self.chevron.pack(side="left")
        title_label = ttk.Label(head, text=title, style="CardTitle.TLabel")
        title_label.pack(side="left")
        self.summary = ttk.Label(head, text="", style="Summary.TLabel")
        self.summary.pack(side="right")
        for w in (head, self.chevron, title_label, self.summary):
            w.bind("<Button-1>", lambda e: self.toggle())
            w.configure(cursor="hand2")
        self.body = ttk.Frame(self, style="Inner.TFrame", padding=(14, 0, 14, 10))
        self.body.columnconfigure(1, weight=1)
        self.is_open = False

    def toggle(self):
        self.is_open = not self.is_open
        if self.is_open:
            self.body.pack(fill="x")
        else:
            self.body.pack_forget()
        self.chevron.configure(text="▾" if self.is_open else "▸")
        if self.on_toggle:
            self.on_toggle()

    def set_summary(self, text, limit=34):
        self.summary.configure(text=text if len(text) <= limit else text[:limit - 1] + "…")


# ---- application --------------------------------------------------------------------

class App:
    def __init__(self, root):
        self.root = root
        self.ui = max(1.0, root.winfo_fpixels("1i") / 96.0)   # Windows display scaling (1.25 at 125 %)
        root.title("EyeTrackTrial - VOGeo-Gaze")
        apply_theme(root)
        self.settings = load_settings()
        self._restore_window()

        self.gaze = None              # RealtimeGaze, cached per (device, fp16)
        self.gaze_key = None
        self.tracker = None
        self.bus = TargetBus()
        self.trace = collections.deque(maxlen=900)
        self.blink_counter = LiveCounter()
        self._closed = False
        self.recording = None         # list while calibration / experiment record samples
        self.mapper, self.cal_report = None, None
        self.session_dir = None
        self.stim = None
        self.cal_samples = None
        self._last_frame_t = None
        self._photo = {}
        self._starting = False
        self.cameras = []
        self.monitors = []
        self.file_path = self.settings.get("file_path")

        self._vars()
        self._build_header()
        ttk.Label(root, text="Developed by Omar Carpentiero", style="Footer.TLabel",
                  padding=(24, 0, 24, 4)).pack(side="bottom", anchor="e")
        self._build_body()
        self.refresh_sources()
        self.refresh_monitors()
        self._update_device_chips()
        self._update_trial_buttons()
        self._update_summaries()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(40, self._poll)
        root.after(1000, self._slow_poll)

    def _restore_window(self):
        """Maximized by default and when it was closed maximized; otherwise the saved
        size, clamped to the screen and centred, so no part opens off-screen."""
        root = self.root
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        size = self.settings.get("_window_size")
        try:
            w, h = (int(v) for v in size.split("x"))
        except (AttributeError, ValueError):
            w, h = 1440, 900
        w, h = min(w, sw), min(h, sh - 80)          # leave room for the taskbar
        root.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - 80 - h) // 2)}")
        root.minsize(min(1100, sw), min(650, sh - 80))
        if self.settings.get("_window_zoomed", True):
            try:
                root.state("zoomed")
            except tk.TclError:
                pass

    # ---- variables / persistence ----
    def _vars(self):
        s = self.settings
        self.v = {
            "source": tk.StringVar(value=s.get("source", "")),
            "resolution": tk.StringVar(value=s.get("resolution", "Predefinita")),
            "flip_h": tk.BooleanVar(value=s.get("flip_h", False)),
            "flip_v": tk.BooleanVar(value=s.get("flip_v", False)),
            "device": tk.StringVar(value=s.get("device", "Automatico")),
            "fp16": tk.BooleanVar(value=s.get("fp16", False)),
            "conf": tk.StringVar(value=s.get("conf", "0.5")),
            "subject": tk.StringVar(value=s.get("subject", "")),
            "monitor": tk.StringVar(value=s.get("monitor", "")),
            "cal_points": tk.StringVar(value=s.get("cal_points", "13")),
            "cal_dwell": tk.StringVar(value=s.get("cal_dwell", "1.5")),
            "fix_dur": tk.StringVar(value=s.get("fix_dur", "1.5")),
            "repeat_blinks": tk.BooleanVar(value=s.get("repeat_blinks", True)),
            "max_repeats": tk.StringVar(value=s.get("max_repeats", "2")),
        }
        counts, enabled = s.get("counts", {}), s.get("enabled", {})
        for k in TYPE_LABELS:
            self.v[f"n_{k}"] = tk.StringVar(value=str(counts.get(k, DEFAULT_COUNTS[k])))
            self.v[f"on_{k}"] = tk.BooleanVar(value=enabled.get(k, True))
        self._save_id = None
        for var in self.v.values():
            var.trace_add("write", lambda *_: self._schedule_save())

    def _schedule_save(self):
        self._update_summaries()
        if self._save_id is not None:
            self.root.after_cancel(self._save_id)
        self._save_id = self.root.after(500, self._save)

    def _save(self):
        self._save_id = None
        s = self.settings
        for k in ("source", "resolution", "flip_h", "flip_v", "device", "fp16", "conf", "subject", "monitor",
                  "cal_points", "cal_dwell", "fix_dur", "repeat_blinks", "max_repeats"):
            s[k] = self.v[k].get()
        s["counts"] = {k: self.v[f"n_{k}"].get() for k in TYPE_LABELS}
        s["enabled"] = {k: self.v[f"on_{k}"].get() for k in TYPE_LABELS}
        s["file_path"] = self.file_path
        zoomed = self.root.state() == "zoomed"
        s["_window_zoomed"] = zoomed
        if not zoomed and self.root.state() == "normal":
            s["_window_size"] = self.root.geometry().split("+")[0]
        save_settings(s)

    # ---- header ----
    def _build_header(self):
        header = ttk.Frame(self.root, style="Header.TFrame", padding=(24, 10))
        header.pack(fill="x")
        titles = ttk.Frame(header, style="Header.TFrame")
        titles.pack(side="left")
        ttk.Label(titles, text="EyeTrackTrial", style="HeaderTitle.TLabel").pack(anchor="w")
        ttk.Label(titles, text="Tracciamento oculare in tempo reale  ·  VOGeo-Gaze  ·  Mind The Brain Lab",
                  style="HeaderSub.TLabel").pack(anchor="w")

        status = ttk.Frame(header, style="Header.TFrame")
        status.pack(side="right")
        self.status_dot = tk.Canvas(status, width=14, height=14, highlightthickness=0, background=COLORS["header"])
        self.status_item = self.status_dot.create_oval(2, 2, 12, 12, fill=COLORS["ready"], outline="")
        self.status_dot.pack(side="left", padx=(0, 8))
        self.status_label = ttk.Label(status, text="Pronto", style="Status.TLabel")
        self.status_label.pack(side="left")

        chips = tk.Frame(header, background=COLORS["header"])
        chips.pack(side="right", padx=(0, 36))
        self.chips = {}
        for key, title in (("camera", "Video"), ("model", "Modello"), ("gpu", "GPU NVIDIA"),
                           ("screen", "Schermi"), ("center", "Centro occhio")):
            chip = tk.Frame(chips, background=COLORS["chip"], padx=12, pady=6)
            chip.pack(side="left", padx=5)
            top = tk.Frame(chip, background=COLORS["chip"])
            top.pack(anchor="w")
            dot = tk.Canvas(top, width=12, height=12, highlightthickness=0, background=COLORS["chip"])
            item = dot.create_oval(1, 1, 11, 11, fill=COLORS["unknown"], outline="")
            dot.pack(side="left", padx=(0, 6))
            tk.Label(top, text=title, background=COLORS["chip"], foreground="white",
                     font=(FONT_FAMILY, FONT_SIZE - 3, "bold")).pack(side="left")
            detail = tk.Label(chip, text="—", background=COLORS["chip"], foreground=COLORS["header_muted"],
                              font=(FONT_FAMILY, FONT_SIZE - 5), anchor="w", width=17)
            detail.pack(anchor="w")
            self.chips[key] = (dot, item, detail)

    def set_chip(self, key, color, text):
        dot, item, detail = self.chips[key]
        dot.itemconfigure(item, fill=COLORS.get(color, color))
        detail.configure(text=text)

    def set_status(self, text, color="ready"):
        self.status_label.configure(text=text)
        self.status_dot.itemconfigure(self.status_item, fill=COLORS.get(color, color))

    # ---- small builders ----
    def _section(self, parent, key, title):
        """A closed collapsible card; returns its body to fill."""
        sec = Collapsible(parent, title)
        sec.pack(fill="x", pady=(0, 8))
        self.sections[key] = sec
        return sec.body

    def _update_summaries(self):
        """One-line summary of every section, shown in its header while it is closed."""
        if not getattr(self, "sections", None):
            return
        v = self.v
        src = v["source"].get()
        if src == FILE_NAME and self.file_path:
            src = os.path.basename(self.file_path)
        res = v["resolution"].get()
        self.sections["source"].set_summary(src + ("" if res == "Predefinita" else f" · {res}"))
        dev = v["device"].get().split(" · ")[0]
        self.sections["model"].set_summary(f"{dev} · soglia {v['conf'].get()}" + (" · fp16" if v["fp16"].get() else ""))
        mon = v["monitor"].get().split("  (")[0].split(": ")[-1]
        self.sections["participant"].set_summary(f"{v['subject'].get().strip() or 'anonimo'} · {mon}")
        self.sections["calibration"].set_summary(f"{v['cal_points'].get()} punti · {v['cal_dwell'].get()} s")
        self.sections["experiment"].set_summary(self.duration_label.cget("text"))

    def _row(self, card, r, label, widget):
        ttk.Label(card, text=label, style="Card.TLabel").grid(row=r, column=0, sticky="w", padx=(0, 10), pady=3)
        widget.grid(row=r, column=1, sticky="ew", pady=3)

    def _value(self, card, r, label):
        ttk.Label(card, text=label, style="Card.TLabel").grid(row=r, column=0, sticky="w", padx=(0, 10), pady=1)
        val = ttk.Label(card, text="—", style="Value.TLabel")
        val.grid(row=r, column=1, sticky="e", pady=1)
        return val

    # ---- layout: one screen ----
    def _build_body(self):
        body = ttk.Frame(self.root, padding=(24, 14, 24, 4))
        body.pack(fill="both", expand=True)

        # Left: collapsible settings, all closed at start (scroll if the window is short);
        # the action buttons stay pinned below them.
        column = ttk.Frame(body)
        column.pack(side="left", fill="y")
        buttons = ttk.Frame(column)
        buttons.pack(side="bottom", fill="x", pady=(8, 0))
        scroll = ScrollColumn(column, width=380)
        scroll.pack(side="top", fill="both", expand=True)
        left = scroll.inner
        self.sections = {}
        ttk.Label(left, text="IMPOSTAZIONI", style="Section.TLabel").pack(anchor="w", pady=(0, 6))

        card = self._section(left, "source", "Sorgente video")
        row = ttk.Frame(card, style="Inner.TFrame")
        self.source_box = ttk.Combobox(row, textvariable=self.v["source"], state="readonly", width=18)
        self.source_box.pack(side="left", fill="x", expand=True)
        self.source_box.bind("<<ComboboxSelected>>", self._on_source_selected)
        ttk.Button(row, text="↻", width=3, style="Ghost.TButton", command=self.refresh_sources).pack(
            side="left", padx=(6, 0))
        self._row(card, 0, "Ingresso", row)
        self._row(card, 1, "Risoluzione", ttk.Combobox(card, textvariable=self.v["resolution"], state="readonly",
                                                       values=list(RESOLUTIONS), width=14))
        ttk.Checkbutton(card, text="Specchia orizzontalmente", variable=self.v["flip_h"],
                        style="Card.TCheckbutton", command=self._apply_flip).grid(row=2, column=0, columnspan=2,
                                                                                  sticky="w")
        ttk.Checkbutton(card, text="Capovolgi verticalmente", variable=self.v["flip_v"],
                        style="Card.TCheckbutton", command=self._apply_flip).grid(row=3, column=0, columnspan=2,
                                                                                  sticky="w")
        self.file_label = ttk.Label(card, text="", style="Summary.TLabel", wraplength=320)
        self.file_label.grid(row=4, column=0, columnspan=2, sticky="w")

        card = self._section(left, "model", "Modello")
        self.device_box = ttk.Combobox(card, textvariable=self.v["device"], state="readonly", width=18)
        self._row(card, 0, "Dispositivo", self.device_box)
        ttk.Checkbutton(card, text="Mezza precisione fp16 (solo GPU)", variable=self.v["fp16"],
                        style="Card.TCheckbutton").grid(row=1, column=0, columnspan=2, sticky="w")
        self._row(card, 2, "Soglia confidenza", ttk.Spinbox(card, from_=0.1, to=0.95, increment=0.05,
                                                            textvariable=self.v["conf"], width=6))
        self.device_hint = ttk.Label(card, text="", style="Summary.TLabel", wraplength=320)
        self.device_hint.grid(row=3, column=0, columnspan=2, sticky="w")

        card = self._section(left, "participant", "Partecipante e schermo")
        self._row(card, 0, "Codice soggetto", ttk.Entry(card, textvariable=self.v["subject"], width=16))
        row = ttk.Frame(card, style="Inner.TFrame")
        self.monitor_box = ttk.Combobox(row, textvariable=self.v["monitor"], state="readonly", width=14)
        self.monitor_box.pack(side="left", fill="x", expand=True)
        self.monitor_box.bind("<<ComboboxSelected>>", lambda e: self._draw_map())
        ttk.Button(row, text="↻", width=3, style="Ghost.TButton", command=self.refresh_monitors).pack(
            side="left", padx=(6, 0))
        self._row(card, 1, "Schermo", row)

        card = self._section(left, "calibration", "Calibrazione")
        self._row(card, 0, "Punti", ttk.Combobox(card, textvariable=self.v["cal_points"], values=CAL_POINTS,
                                                 state="readonly", width=6))
        self._row(card, 1, "Durata per punto (s)", ttk.Spinbox(card, from_=0.8, to=4.0, increment=0.1,
                                                              textvariable=self.v["cal_dwell"], width=6))

        card = self._section(left, "experiment", "Esperimento")
        types = ttk.Frame(card, style="Inner.TFrame")
        types.grid(row=0, column=0, columnspan=2, sticky="ew")
        for i, (k, label) in enumerate(TYPE_LABELS.items()):   # two columns of three
            r, c = i % 3, 2 * (i // 3)
            ttk.Checkbutton(types, text=label, variable=self.v[f"on_{k}"], style="Card.TCheckbutton").grid(
                row=r, column=c, sticky="w", padx=(0 if c == 0 else 8, 4))
            ttk.Spinbox(types, from_=1, to=60, textvariable=self.v[f"n_{k}"], width=3).grid(
                row=r, column=c + 1, sticky="e", pady=2)
        types.columnconfigure(1, weight=1)
        self._row(card, 1, "Durata fissazione (s)",
                  ttk.Spinbox(card, from_=0.8, to=4.0, increment=0.1, textvariable=self.v["fix_dur"], width=5))
        blink_row = ttk.Frame(card, style="Inner.TFrame")
        ttk.Checkbutton(blink_row, text="Ripeti persi per ammiccamento, max", variable=self.v["repeat_blinks"],
                        style="Card.TCheckbutton").pack(side="left")
        ttk.Spinbox(blink_row, from_=1, to=3, textvariable=self.v["max_repeats"], width=3).pack(side="right")
        blink_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        self.duration_label = ttk.Label(card, text="", style="Summary.TLabel")
        self.duration_label.grid(row=3, column=0, columnspan=2, sticky="w")
        for k in TYPE_LABELS:
            for p in ("n_", "on_"):
                self.v[p + k].trace_add("write", lambda *_: self._update_duration())
        self.v["fix_dur"].trace_add("write", lambda *_: self._update_duration())
        self._update_duration()

        card = self._section(left, "status", "Stato")
        self.stat = {}
        for r, (key, label) in enumerate((("cam_fps", "Video (fps)"), ("model_fps", "Modello (fps)"),
                                          ("theta", "θ orizzontale / verticale"),
                                          ("conf", "Confidenza pupilla / iride"),
                                          ("open_eye", "Occhio aperto"), ("blinks", "Ammiccamenti"),
                                          ("center", "Centro occhio"))):
            self.stat[key] = self._value(card, r, label)
        ttk.Button(card, text="Reimposta centro occhio", style="Ghost.TButton",
                   command=self.reset_center).grid(row=10, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.sections["status"].set_summary("fermo")

        # Column width fixed once, to the widest section when open, so opening a section never
        # widens the column.
        self.root.update_idletasks()
        widest = max(max(sec.body.winfo_reqwidth(), sec.winfo_reqwidth()) for sec in self.sections.values())
        scroll.set_width(max(380, widest + 2))

        # Pinned actions, in the order they are used.
        row = ttk.Frame(buttons)
        row.pack(fill="x")
        self.start_btn = ttk.Button(row, text="▶  Avvia", style="Accent.TButton", command=self.start_tracking)
        self.start_btn.pack(side="left", fill="x", expand=True)
        self.stop_btn = ttk.Button(row, text="■  Ferma", style="Danger.TButton", command=self.stop_tracking,
                                   state="disabled")
        self.stop_btn.pack(side="left", fill="x", expand=True, padx=(8, 0))
        self.cal_btn = ttk.Button(buttons, text="1 · Calibrazione", style="Primary.TButton", command=self.run_calibration)
        self.cal_btn.pack(fill="x", pady=(8, 0))
        self.exp_btn = ttk.Button(buttons, text="2 · Esperimento", style="Accent.TButton", command=self.run_experiment)
        self.exp_btn.pack(fill="x", pady=(8, 0))
        ttk.Button(buttons, text="Apri cartella risultati", style="Ghost.TButton",
                   command=self.open_results).pack(fill="x", pady=(8, 0))

        # Right: everything visible at once.
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(24, 0))
        self.view_scale = VIEW_SCALE
        self.view_holders = []
        self.map_box = (int(240 * VIEW_SCALE * 16 / 9), int(240 * VIEW_SCALE))
        right.bind("<Configure>", self._fit_right)
        self.trial_info = ttk.Label(right, text="", style="Description.TLabel", wraplength=1300, justify="left")
        self.trial_info.pack(anchor="w")

        # Row 1: camera, segmentation, participant screen.
        top = ttk.Frame(right)
        top.pack(fill="x", pady=(4, 0))
        self.view_labels = {}
        for key, title in (("camera", "CAMERA  ·  ellissi rilevate"), ("seg", "SEGMENTAZIONE")):
            col = ttk.Frame(top)
            col.pack(side="left", padx=(0, 16), anchor="n")
            ttk.Label(col, text=title, style="Section.TLabel").pack(anchor="w", pady=(0, 4))
            holder = tk.Frame(col, background=COLORS["log_bg"], width=int(320 * VIEW_SCALE),
                              height=int(240 * VIEW_SCALE))
            holder.pack()
            holder.pack_propagate(False)
            self.view_holders.append(holder)
            lab = tk.Label(holder, background=COLORS["log_bg"], foreground=COLORS["log_fg"],
                           text="Nessun video\nscegliere un ingresso e premere Avvia", font=(FONT_FAMILY, FONT_SIZE - 1))
            lab.pack(fill="both", expand=True)
            self.view_labels[key] = lab
        map_col = ttk.Frame(top)
        map_col.pack(side="left", anchor="n")
        ttk.Label(map_col, text="SCHERMO  ·  bersagli (●), sguardo (×), sguardo live (rosso)",
                  style="Section.TLabel").pack(anchor="w", pady=(0, 4))
        self.map_canvas = tk.Canvas(map_col, width=self.map_box[0], height=self.map_box[1], background="#9ca3af",
                                    highlightthickness=1, highlightbackground=COLORS["border"])
        self.map_canvas.pack(anchor="w")
        self.map_legend = tk.Frame(map_col, background=COLORS["bg"])
        self.map_legend.pack(anchor="w", pady=(2, 0))
        for i, (k, label) in enumerate(TYPE_LABELS.items()):
            cell = tk.Frame(self.map_legend, background=COLORS["bg"])
            cell.grid(row=i // 3, column=i % 3, sticky="w", padx=(0, 10))
            tk.Label(cell, text="●", foreground=TYPE_COLORS[k], background=COLORS["bg"],
                     font=(FONT_FAMILY, FONT_SIZE - 3)).pack(side="left")
            tk.Label(cell, text=label, foreground=COLORS["muted"], background=COLORS["bg"],
                     font=(FONT_FAMILY, FONT_SIZE - 4)).pack(side="left")
        for col in self.view_holders:
            ttk.Label(col.master, style="Description.TLabel", wraplength=int(320 * VIEW_SCALE),
                      text=("bianco pupilla · giallo corretta · ciano iride" if col is self.view_holders[0]
                            else "blu pupilla · rosso iride · verde aperto")).pack(anchor="w", pady=(2, 0))

        # Row 2: gaze trace and calibration quality.
        mid = ttk.Frame(right)
        mid.pack(fill="x", pady=(10, 0))
        cal_col = ttk.Frame(mid)
        cal_col.pack(side="right", fill="y", padx=(16, 0))
        trace_col = ttk.Frame(mid)
        trace_col.pack(side="left", fill="both", expand=True)
        ttk.Label(trace_col, text=f"SGUARDO  ·  {TRACE_SECONDS:.0f} s  (blu: θ orizz., arancio: θ vert., "
                                  "grigio: occhio chiuso)", style="Section.TLabel").pack(anchor="w", pady=(0, 4))
        self.trace_canvas = tk.Canvas(trace_col, width=200, height=130, background=COLORS["card"],
                                      highlightthickness=1, highlightbackground=COLORS["border"])
        self.trace_canvas.pack(fill="both", expand=True)
        ttk.Label(cal_col, text="CALIBRAZIONE", style="Section.TLabel").pack(anchor="w", pady=(0, 4))
        card = ttk.Frame(cal_col, style="Card.TFrame", padding=(14, 8))
        card.pack(fill="both", expand=True)
        self.cal_stat = {}
        items = (("stato", "Stato"), ("punti", "Punti usati"), ("ordine", "Modello"), ("blinks", "Ammicc. / ripetuti"),
                 ("res", "Residuo medio"), ("loo", "Leave-one-out"), ("max", "Residuo massimo"))
        for i, (key, label) in enumerate(items):
            r, c = i % 4, 2 * (i // 4)
            ttk.Label(card, text=label, style="Card.TLabel").grid(row=r, column=c, sticky="w",
                                                                  padx=(0 if c == 0 else 24, 10), pady=1)
            val = ttk.Label(card, text="—", style="Value.TLabel")
            val.grid(row=r, column=c + 1, sticky="e", pady=1)
            self.cal_stat[key] = val

        # Row 3: experiment errors.
        ttk.Label(right, text="ERRORE DELL'ESPERIMENTO  (mediana per prova, px)", style="Section.TLabel").pack(
            anchor="w", pady=(10, 4))
        self.result_nb = ttk.Notebook(right)
        self.result_nb.pack(fill="both", expand=True)
        cols = [("categoria", "Categoria", 150), ("n_prove", "Prove", 60), ("n_prove_valide", "Valide", 60),
                ("errore_mediano_px", "Mediana", 80), ("errore_medio_px", "Media", 70), ("ds_px", "DS", 60),
                ("iqr", "Q1–Q3", 100), ("precisione_rms_px", "Precisione RMS", 110),
                ("campioni_validi_pct", "Validi", 70), ("persi_blink_pct", "Persi blink", 85),
                ("ammiccamenti", "Ammicc.", 70), ("prove_ripetute", "Ripetute", 70)]
        self.tables = {}
        # Column widths from the heading text, so they follow the font size and Windows display scaling.
        import tkinter.font as tkfont
        head_font = tkfont.Font(family=FONT_FAMILY, size=FONT_SIZE - 2, weight="bold")
        body_font = tkfont.Font(family=FONT_FAMILY, size=FONT_SIZE, weight="bold")
        for key, title in (("tipo", "Per tipo di bersaglio"), ("regione", "Per regione dello schermo"),
                           ("eccentricita", "Per eccentricità")):
            frame = ttk.Frame(self.result_nb)
            tree = ttk.Treeview(frame, columns=[c[0] for c in cols], show="headings", height=6)
            for cid, head, _ in cols:
                tree.heading(cid, text=head)
                sample = "Centro-sinistra" if cid == "categoria" else "100 %" if cid.endswith("pct") else "999–999"
                width = max(head_font.measure(head), body_font.measure(sample)) + 18
                tree.column(cid, width=width, minwidth=width, anchor="w" if cid == "categoria" else "e",
                            stretch=cid == "categoria")
            tree.tag_configure("total", font=(FONT_FAMILY, FONT_SIZE, "bold"))
            bar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=bar.set)
            bar.pack(side="right", fill="y")
            tree.pack(side="left", fill="both", expand=True)
            self.result_nb.add(frame, text=title)
            self.tables[key] = tree
        self.trials_df = None

    def refresh_sources(self):
        self.cameras = list_cameras()
        names = [name for _, name in self.cameras] + [FILE_NAME, SIM_NAME]
        self.source_box.configure(values=names)
        if self.v["source"].get() not in names:
            self.v["source"].set(names[0])
        self._update_file_label()
        self._update_device_chips()

    def _on_source_selected(self, _=None):
        if self.v["source"].get() == FILE_NAME:
            self._choose_file()
        self._update_file_label()

    def _choose_file(self):
        path = filedialog.askopenfilename(
            title="Video dell'occhio", filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("Tutti i file", "*.*")],
            initialdir=os.path.dirname(self.file_path) if self.file_path else None)
        if path:
            self.file_path = path
            self._schedule_save()
        return bool(path)

    def _update_file_label(self):
        src = self.v["source"].get()
        text = ""
        if src == FILE_NAME:
            text = f"File: {os.path.basename(self.file_path)}" if self.file_path else "Nessun file scelto"
        elif src == SIM_NAME:
            text = "Occhio sintetico che guarda i bersagli della prova: per provare il programma senza camera."
        self.file_label.configure(text=text)

    def _apply_flip(self):
        if self.tracker is not None:
            self.tracker.flip_h, self.tracker.flip_v = self.v["flip_h"].get(), self.v["flip_v"].get()

    def _update_device_chips(self):
        import torch
        from tracker import cuda_devices, torch_build_info
        gpus = cuda_devices()
        values = ["Automatico"] + [f"GPU {d.split(':')[1]} · {name}" for d, name in gpus] + ["CPU"]
        self.device_box.configure(values=values)
        if self.v["device"].get() not in values:
            self.v["device"].set("Automatico")
        if gpus:
            self.set_chip("gpu", "ready", gpus[0][1][:26])
            self.device_hint.configure(text=torch_build_info())
        else:
            build = torch_build_info()
            self.set_chip("gpu", "missing" if torch.version.cuda is None else "unknown",
                          "PyTorch solo CPU" if torch.version.cuda is None else "nessuna GPU rilevata")
            self.device_hint.configure(
                text=f"{build}: per la GPU installare PyTorch con CUDA (README).")

    def _device_arg(self):
        d = self.v["device"].get()
        if d.startswith("GPU "):
            return f"cuda:{d.split()[1]}"
        return "cpu" if d == "CPU" else "auto"

    # ---- tracking ----
    def start_tracking(self):
        if self._starting or (self.tracker is not None and self.tracker.running):
            return
        src_name = self.v["source"].get()
        if src_name == FILE_NAME and not (self.file_path and os.path.exists(self.file_path)):
            if not self._choose_file():
                return
            self._update_file_label()
        try:
            conf = float(self.v["conf"].get())
        except ValueError:
            conf = 0.5
        device, fp16 = self._device_arg(), self.v["fp16"].get()
        self._starting = True
        self.blink_counter.reset()
        self.start_btn.configure(state="disabled")
        self.set_status("Avvio...", "running")
        self.set_chip("model", "running", "caricamento...")

        def work():
            try:
                from tracker import RealtimeGaze, TrackerThread
                key = (device, fp16)
                if self.gaze is None or self.gaze_key != key:
                    self.gaze, self.gaze_key = RealtimeGaze(device=None if device == "auto" else device,
                                                            fp16=fp16, conf_threshold=conf), key
                    self.mapper = None    # different model numerics: recalibrate
                self.gaze.conf_threshold = conf
                if src_name == SIM_NAME:
                    source = SimulatedEyeSource(self.bus)
                elif src_name == FILE_NAME:
                    source = VideoFileSource(self.file_path)
                else:
                    index = next(i for i, n in self.cameras if n == src_name)
                    source = CameraSource(index, src_name, RESOLUTIONS.get(self.v["resolution"].get()))
                tracker = TrackerThread(self.gaze, source, self.v["flip_h"].get(), self.v["flip_v"].get())
                tracker.listeners.append(self._on_sample)
                tracker.start()
                self.root.after(0, lambda: self._started(tracker, source))
            except Exception as exc:
                msg = str(exc) or exc.__class__.__name__
                self.root.after(0, lambda: self._start_failed(msg))

        threading.Thread(target=work, daemon=True).start()

    def _started(self, tracker, source):
        self._starting = False
        self.tracker = tracker
        self.stop_btn.configure(state="normal")
        size = getattr(source, "size", None)
        self.set_chip("camera", "ready", f"{source.label[:18]}" + (f" · {size[0]}×{size[1]}" if size else ""))
        self.set_chip("model", "ready", self.gaze.device_label[:30])
        self.set_status("In acquisizione", "ready")
        self._update_trial_buttons()

    def _start_failed(self, msg):
        self._starting = False
        self.start_btn.configure(state="normal")
        self.set_chip("camera", "missing", "errore")
        self.set_chip("model", "ready" if self.gaze else "missing", self.gaze.device_label[:30] if self.gaze else "errore")
        self.set_status("Errore", "missing")
        messagebox.showerror("Avvio non riuscito", msg)

    def stop_tracking(self):
        if self.stim is not None:
            self.stim.abort()
        if self.tracker is not None:
            self.tracker.stop()
            self.tracker = None
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.set_chip("camera", "unknown", "fermo")
        self.sections["status"].set_summary("fermo")
        self.set_status("Pronto", "ready")
        self._update_trial_buttons()

    def reset_center(self):
        if self.gaze is None:
            return
        if self.mapper is not None and not messagebox.askyesno(
                "Reimposta centro", "La calibrazione attuale non sarà più valida. Continuare?"):
            return
        self.gaze.reset_center()
        self.mapper = None
        self._update_trial_buttons()
        self._draw_map()

    def _on_sample(self, res):
        """Tracker thread: keep what the GUI and the trial need."""
        self._closed = self.blink_counter.update(res["t"], res["open_eye"], res["pupil_conf"])
        self.trace.append((res["t"], res["theta_h"], res["theta_v"], res["valid"], self._closed))
        rec = self.recording
        if rec is not None:
            rec.append({k: res[k] for k in ("t", "theta_h", "theta_v", "valid", "gaze_ok", "pupil_conf",
                                            "iris_conf", "open_eye", "r_pupil")})

    # ---- periodic GUI updates ----
    def _poll(self):
        try:
            tr = self.tracker
            if tr is not None:
                if tr.error is not None or tr.source.error is not None:
                    err = tr.error or tr.source.error
                    self.stop_tracking()
                    messagebox.showerror("Acquisizione interrotta", str(err))
                elif not tr.running:
                    self.stop_tracking()
                else:
                    res = tr.latest
                    if res is not None and res["t"] != self._last_frame_t:
                        self._last_frame_t = res["t"]
                        self._show_frame(res)
                        self._live_gaze(res)
        finally:
            self.root.after(33, self._poll)

    def _slow_poll(self):
        try:
            tr, g = self.tracker, self.gaze
            res = tr.latest if tr is not None else None
            if res is not None:
                self.stat["cam_fps"].configure(text=fmt(tr.cam_fps, 0))
                self.stat["model_fps"].configure(text=fmt(tr.model_fps, 0))
                self.stat["theta"].configure(text=f'{fmt(res["theta_h"], 1, "°")}  /  {fmt(res["theta_v"], 1, "°")}')
                self.stat["conf"].configure(text=f'{fmt(res["pupil_conf"], 2)}  /  {fmt(res["iris_conf"], 2)}')
                self.stat["open_eye"].configure(text=fmt(res["open_eye"], 2))
                bc = self.blink_counter
                rate = bc.rate_per_min(res["t"])
                self.stat["blinks"].configure(text=f"{bc.count}" + ("" if not np.isfinite(rate) else f"  ·  {rate:.0f}/min"))
            if g is not None:
                state, n = g.center_state()
                self.stat["center"].configure(text=f"{state}" + (f" ({n})" if n else ""))
                if tr is not None:
                    self.sections["status"].set_summary(f"{tr.model_fps:.0f} fps · centro {state}")
                color = {"bloccato": "accent", "stimato": "ready", "provvisorio": "warning"}.get(state, "unknown")
                self.set_chip("center", color, {"bloccato": "bloccato per la prova", "stimato": f"stimato · {n} clip",
                                                "provvisorio": "provvisorio", "assente": "guardare in giro"}[state])
            self._draw_trace()
        finally:
            self.root.after(250, self._slow_poll)

    def _show_frame(self, res):
        s = self.view_scale
        w, h = int(320 * s), int(240 * s)
        cam = cv2.cvtColor(cv2.resize(res["gray"], (w, h), interpolation=cv2.INTER_LINEAR), cv2.COLOR_GRAY2RGB)
        draw_el(cam, res["iris_el"], (50, 255, 255), s)
        draw_el(cam, res["entpup_el"], (255, 255, 255), s)
        draw_el(cam, res["pupil_el"], (255, 255, 0), s, 1)
        if self._closed:
            cv2.rectangle(cam, (0, 0), (w - 1, h - 1), (255, 170, 0), 6)
            cv2.putText(cam, "OCCHIO CHIUSO", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 170, 0), 2, cv2.LINE_AA)
        elif not res["valid"]:
            cv2.putText(cam, "non valido", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 80, 80), 2, cv2.LINE_AA)
        seg = cv2.cvtColor(cv2.resize(res["seg"], (w, h), interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2RGB)
        for key, img in (("camera", cam), ("seg", seg)):
            photo = ImageTk.PhotoImage(Image.fromarray(img))
            self.view_labels[key].configure(image=photo, text="")
            self._photo[key] = photo

    def _draw_trace(self):
        c = self.trace_canvas
        c.delete("all")
        W, H = max(c.winfo_width(), 100), max(c.winfo_height(), 50)
        span = 30.0
        y = lambda v: H / 2 - v / span * (H / 2 - 8)
        for v in (-20, -10, 0, 10, 20):
            c.create_line(36, y(v), W, y(v), fill=COLORS["disabled"] if v else COLORS["border"])
            c.create_text(30, y(v), text=f"{v}°", anchor="e", fill=COLORS["muted"], font=(FONT_FAMILY, FONT_SIZE - 4))
        data = list(self.trace)
        if len(data) < 2:
            return
        t_end = data[-1][0]
        x = lambda t: 36 + (W - 40) * (1 - (t_end - t) / TRACE_SECONDS)
        start = None
        for d in data + [(t_end, 0, 0, True, False)]:
            if d[0] < t_end - TRACE_SECONDS:
                continue
            if d[4] and start is None:
                start = d[0]
            elif not d[4] and start is not None:
                c.create_rectangle(x(start), 4, x(d[0]), H - 4, fill="#e5e7eb", outline="")
                start = None
        for idx, color in ((1, COLORS["accent"]), (2, COLORS["warning"])):
            pts = []
            for d in data:
                if d[0] < t_end - TRACE_SECONDS:
                    continue
                if not d[3] or not np.isfinite(d[idx]):
                    if len(pts) >= 4:
                        c.create_line(*pts, fill=color, width=2)
                    pts = []
                    continue
                pts += [x(d[0]), y(max(-span, min(span, d[idx])))]
            if len(pts) >= 4:
                c.create_line(*pts, fill=color, width=2)

    def refresh_monitors(self):
        self.monitors = list_monitors(self.root)
        labels = [m[0] for m in self.monitors]
        self.monitor_box.configure(values=labels)
        if self.v["monitor"].get() not in labels:
            # Default: a secondary screen if there is one.
            self.v["monitor"].set(labels[1] if len(labels) > 1 else labels[0])
        n = len(self.monitors)
        self.set_chip("screen", "ready" if n > 1 else "warning",
                      "1 schermo" if n == 1 else f"{n} schermi")
        self._draw_map()

    def monitor(self):
        label = self.v["monitor"].get()
        return next((m for l, m in self.monitors if l == label), self.monitors[0][1])

    def _counts(self):
        counts = {}
        for k in TYPE_LABELS:
            try:
                n = int(self.v[f"n_{k}"].get())
            except ValueError:
                n = 0
            counts[k] = max(0, n) if self.v[f"on_{k}"].get() else 0
        return counts

    def _update_duration(self):
        try:
            fix = float(self.v["fix_dur"].get())
        except ValueError:
            fix = 1.5
        c = self._counts()
        seconds = ((c["punto_piccolo"] + c["punto_grande"] + c["lettera"] + c["immagine"]) * (fix + 0.15)
                   + c["saccade"] * 2.35 + c["inseguimento"] * 6.95)
        self.duration_label.configure(text=f"{sum(c.values())} prove  ·  circa {seconds / 60:.1f} min")
        self._update_summaries()

    def _update_trial_buttons(self):
        running = self.tracker is not None and self.tracker.running
        busy = self.stim is not None
        self.cal_btn.configure(state="normal" if running and not busy else "disabled")
        self.exp_btn.configure(state="normal" if running and not busy and self.mapper is not None else "disabled")
        if not running:
            self.trial_info.configure(text="Scegliere la sorgente video e premere ▶ Avvia.")
        elif self.mapper is None:
            self.trial_info.configure(text="Pronto per la calibrazione. Il partecipante deve tenere la testa ferma; "
                                           "il centro dell'occhio viene bloccato all'inizio della calibrazione.")
        elif not busy:
            self.trial_info.configure(text="Calibrazione valida: il punto rosso sulla mappa è lo sguardo stimato "
                                           "in tempo reale. Avviare l'esperimento quando pronti.")
        if self.mapper is None:
            self.cal_stat["stato"].configure(text="non calibrato")

    # ---- calibration ----
    def _new_session(self):
        subject = "".join(ch for ch in self.v["subject"].get().strip() if ch.isalnum() or ch in "-_") or "anonimo"
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = RESULTS_DIR / f"{subject}_{stamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def _sim_target(self):
        return self.bus.set if self.v["source"].get() == SIM_NAME else None

    def run_calibration(self):
        g = self.gaze
        if self.tracker is None or g is None:
            return
        if g.running_center() is None:
            messagebox.showwarning("Centro dell'occhio",
                                   "Il centro dell'occhio non è ancora stimato: far guardare al partecipante "
                                   "intorno per qualche secondo e riprovare.")
            return
        g.freeze()
        self.mapper = None
        try:
            n, dwell = int(self.v["cal_points"].get()), float(self.v["cal_dwell"].get())
        except ValueError:
            n, dwell = 13, 1.5
        from calibration import calibration_grid
        mon = self.monitor()
        points = calibration_grid(n, mon[2], mon[3])
        # Centre point first, the others in random order.
        first = min(points, key=lambda p: (p[0] - mon[2] / 2) ** 2 + (p[1] - mon[3] / 2) ** 2)
        rest = [p for p in points if p != first]
        random.shuffle(rest)
        steps = calibration_sequence([first] + rest, dwell=dwell)
        self._new_session()
        self._phase = {"steps": steps, "attempt": 0, "repeated": set()}
        self.recording = []
        self.stim = StimulusWindow(self.root, mon, on_target=self._sim_target())
        self.stim.play(steps, "Calibrazione\n\nFissare il centro di ogni punto finché non scompare.",
                       self._calibration_done, self._trial_aborted)
        self.set_status("Calibrazione in corso", "running")
        self._update_trial_buttons()

    def _max_repeats(self):
        if not self.v["repeat_blinks"].get():
            return 0
        try:
            return max(0, int(self.v["max_repeats"].get()))
        except ValueError:
            return 2

    def _repeat_lost(self, events, done_cb, what):
        """Replay the trials / points lost to blinks in the same window; True if it did."""
        from analysis import lost_to_blinks
        ph = self._phase
        if ph["attempt"] >= self._max_repeats():
            return False
        lost = lost_to_blinks(list(self.recording or []), events)
        if not lost:
            return False
        ph["attempt"] += 1
        ph["repeated"] |= lost
        steps = repeat_sequence(ph["steps"], lost, ph["attempt"])
        n = len(lost)
        self.set_status(f"Ripetizione di {n} {what}", "running")
        self.stim.play_more(steps, f"Ripetiamo {n} {what} persi per un ammiccamento.\nTenere gli occhi aperti.",
                            done_cb)
        return True

    def _calibration_done(self, events):
        from analysis import calibration_points
        from calibration import fit_calibration
        if self._repeat_lost(events, self._calibration_done, "punti"):
            return
        samples, self.recording = self.recording or [], None
        self.stim.close()
        self.stim = None
        points, df, blinks = calibration_points(samples, events)
        try:
            self.mapper, self.cal_report = fit_calibration(points)
        except ValueError as exc:
            self.mapper, self.cal_report = None, None
            self.set_status("Calibrazione fallita", "missing")
            self._update_trial_buttons()
            messagebox.showerror("Calibrazione", str(exc))
            return
        mon = self.monitor()
        rep = self.cal_report
        n_blinks = int((blinks["tipo"] == "ammiccamento").sum()) if len(blinks) else 0
        rep["ammiccamenti"] = n_blinks
        rep["punti_ripetuti"] = sorted(int(k) + 1 for k in self._phase["repeated"])
        with open(self.session_dir / "calibrazione.json", "w", encoding="utf-8") as f:
            json.dump({"mapper": self.mapper.to_dict(), "report": rep, "schermo": mon,
                       "centro_occhio_mm": self.gaze.frozen_center.tolist(),
                       "dispositivo": self.gaze.device_label, "eventi": events,
                       "impostazioni": self.settings}, f, indent=2)
        df.to_csv(self.session_dir / "calibrazione_campioni.csv", index=False)
        blinks.to_csv(self.session_dir / "calibrazione_ammiccamenti.csv", index=False)
        self.cal_stat["stato"].configure(text="calibrato")
        self.cal_stat["punti"].configure(text=f"{rep['punti_usati']} / {len(points)}")
        self.cal_stat["ordine"].configure(text="polinomio 2° grado" if rep["ordine"] == 2 else "affine")
        self.cal_stat["res"].configure(text=fmt(rep["residuo_medio_px"], 0, " px"))
        self.cal_stat["loo"].configure(text=fmt(rep["residuo_loo_medio_px"] or np.nan, 0, " px"))
        self.cal_stat["max"].configure(text=fmt(rep["residuo_max_px"], 0, " px"))
        self.cal_stat["blinks"].configure(text=f"{n_blinks} / {len(self._phase['repeated'])}")
        self.trials_df = None
        self._draw_map()
        self.set_status("Calibrato", "ready")
        self._update_trial_buttons()
        if rep["punti_usati"] < len(points):
            messagebox.showwarning("Calibrazione", f"{len(points) - rep['punti_usati']} punti senza abbastanza campioni "
                                                   "validi (ammiccamenti?) sono stati esclusi.")

    def _trial_aborted(self):
        self.recording = None
        self.stim = None
        self.set_status("Interrotto", "warning")
        self._update_trial_buttons()

    # ---- experiment ----
    def run_experiment(self):
        if self.mapper is None or self.tracker is None:
            return
        counts = self._counts()
        if not sum(counts.values()):
            messagebox.showwarning("Esperimento", "Nessun tipo di bersaglio selezionato.")
            return
        try:
            fix = float(self.v["fix_dur"].get())
        except ValueError:
            fix = 1.5
        mon = self.monitor()
        steps = experiment_sequence(counts, mon[2], mon[3], fix_dur=fix)
        self._phase = {"steps": steps, "attempt": 0, "repeated": set()}
        self.recording = []
        self.stim = StimulusWindow(self.root, mon, on_target=self._sim_target())
        self.stim.play(steps, "Esperimento\n\nGuardare ogni stimolo che compare (punti, lettere, immagini)\n"
                              "e seguire con lo sguardo il punto quando si muove.",
                       self._experiment_done, self._trial_aborted)
        self.set_status("Esperimento in corso", "running")
        self._update_trial_buttons()

    def _experiment_done(self, events):
        from analysis import analyze, blink_summary
        if self._repeat_lost(events, self._experiment_done, "prove"):
            return
        samples, self.recording = self.recording or [], None
        self.stim.close()
        self.stim = None
        mon = self.monitor()
        df, trials, summaries, blinks = analyze(samples, events, self.mapper, mon[2], mon[3])
        self.trials_df = trials
        out = self.session_dir
        df.to_csv(out / "esperimento_campioni.csv", index=False)
        trials.to_csv(out / "prove.csv", index=False)
        blinks.to_csv(out / "ammiccamenti.csv", index=False)
        for key, table in summaries.items():
            table.to_csv(out / f"riepilogo_{key}.csv", index=False)
        duration = float(df["t"].iloc[-1] - df["t"].iloc[0]) if len(df) > 1 else 0.0
        bsum = blink_summary(blinks, duration)
        bsum["prove_ripetute"] = len(self._phase["repeated"])
        with open(out / "riepilogo_ammiccamenti.json", "w", encoding="utf-8") as f:
            json.dump(bsum, f, indent=2)
        with open(out / "eventi.json", "w", encoding="utf-8") as f:
            json.dump(events, f, indent=1)
        for key, tree in self.tables.items():
            tree.delete(*tree.get_children())
            table = summaries.get(key)
            if table is None:
                continue
            # Totale first, so it stays visible however short the window is.
            for _, r in pd.concat([table.tail(1), table.iloc[:-1]]).iterrows():
                tree.insert("", "end", tags=("total",) if r["categoria"] == "Totale" else (), values=(
                    r["categoria"], r["n_prove"], r["n_prove_valide"], fmt(r["errore_mediano_px"], 0),
                    fmt(r["errore_medio_px"], 0), fmt(r["ds_px"], 0),
                    f"{fmt(r['q1_px'], 0)}–{fmt(r['q3_px'], 0)}", fmt(r["precisione_rms_px"], 1),
                    fmt(r["campioni_validi_pct"], 0, " %"), fmt(r["persi_blink_pct"], 0, " %"),
                    r["ammiccamenti"], r["prove_ripetute"]))
        self._draw_map()
        total = summaries.get("tipo")
        med = float(total.iloc[-1]["errore_mediano_px"]) if total is not None and len(total) else float("nan")
        self.set_status("Esperimento completato", "ready")
        self._update_trial_buttons()
        self.trial_info.configure(
            text=f"Esperimento completato: errore mediano {fmt(med, 0, ' px')}.  Ammiccamenti: {bsum['ammiccamenti']} "
                 f"({fmt(bsum['frequenza_per_min'], 0)}/min, durata mediana {fmt(bsum['durata_mediana_ms'], 0)} ms), "
                 f"prove ripetute: {bsum['prove_ripetute']}.  Risultati salvati in {out}")

    def open_results(self):
        path = self.session_dir if self.session_dir and self.session_dir.exists() else RESULTS_DIR
        path.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(path)
        else:
            import subprocess
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])

    # ---- screen map ----
    def _fit_right(self, event):
        """Largest scale (up to VIEW_SCALE) at which camera, segmentation and screen map
        fit side by side and the trace, calibration and tables still fit below."""
        mon = self.monitor() if self.monitors else (0, 0, 16, 9)
        aspect = mon[2] / mon[3]
        scale = min(VIEW_SCALE, (event.width - 40) / (640 + 240 * aspect), (event.height - 540 * self.ui) / 240.0)
        scale = max(0.55, round(scale, 2))
        if abs(scale - self.view_scale) > 0.01:
            self.view_scale = scale
            for holder in self.view_holders:
                holder.configure(width=int(320 * scale), height=int(240 * scale))
                for child in holder.master.winfo_children():
                    if isinstance(child, ttk.Label) and str(child.cget("style")) == "Description.TLabel":
                        child.configure(wraplength=int(320 * scale))
        box = (int(240 * scale * aspect), int(240 * scale))
        if box != self.map_box:
            self.map_box = box
            self._draw_map()

    def _map_transform(self):
        mon = self.monitor()
        c = self.map_canvas
        W = int(min(self.map_box[0], self.map_box[1] * mon[2] / mon[3]))
        H = int(W * mon[3] / mon[2])
        c.configure(width=W, height=H)
        k = W / mon[2]
        return (lambda x, y: (x * k, y * k)), mon

    def _draw_map(self):
        if not self.monitors:
            return
        c = self.map_canvas
        c.delete("all")
        tf, mon = self._map_transform()
        W, H = tf(mon[2], mon[3])
        for i in (1, 2):
            c.create_line(W * i / 3, 0, W * i / 3, H, fill="#d1d5db", dash=(3, 4))
            c.create_line(0, H * i / 3, W, H * i / 3, fill="#d1d5db", dash=(3, 4))
        rep = self.cal_report if self.mapper is not None else None
        if rep:
            for p in rep["punti"]:
                x, y = tf(p["x"], p["y"])
                c.create_line(x - 6, y, x + 6, y, fill="white", width=2)
                c.create_line(x, y - 6, x, y + 6, fill="white", width=2)
                if "gaze_x" in p:
                    gx, gy = tf(p["gaze_x"], p["gaze_y"])
                    c.create_line(x, y, gx, gy, fill="#1f2937")
                    c.create_text(gx, gy, text="×", fill="#1f2937", font=(FONT_FAMILY, 10, "bold"))
        if self.trials_df is not None and len(self.trials_df):
            for _, r in self.trials_df.iterrows():
                x, y = tf(r["target_x"], r["target_y"])
                color = TYPE_COLORS.get(r["tipo"], "black")
                c.create_oval(x - 4, y - 4, x + 4, y + 4, fill=color, outline="white")
                if np.isfinite(r["gaze_x"]):
                    gx, gy = tf(r["gaze_x"], r["gaze_y"])
                    c.create_line(x, y, gx, gy, fill=color)
                    c.create_text(gx, gy, text="×", fill=color, font=(FONT_FAMILY, 10, "bold"))
        self._gaze_item = c.create_oval(-20, -20, -10, -10, fill="#dc2626", outline="white", width=2)

    def _live_gaze(self, res):
        item = getattr(self, "_gaze_item", None)
        if item is None:
            return
        c = self.map_canvas
        if self.mapper is None or not res["valid"]:
            c.coords(item, -20, -20, -10, -10)
            return
        tf, _ = self._map_transform()
        gx, gy = self.mapper.map([[res["theta_h"], res["theta_v"]]])[0]
        x, y = tf(gx, gy)
        c.coords(item, x - 7, y - 7, x + 7, y + 7)

    # ---- shutdown ----
    def on_close(self):
        self._save()
        if self.stim is not None:
            self.stim.close()
        if self.tracker is not None:
            self.tracker.stop()
        self.root.destroy()


def main():
    try:   # real pixels, so screen coordinates match the monitors
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
