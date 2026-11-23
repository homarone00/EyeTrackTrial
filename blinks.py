"""Blinks (sbattimento delle palpebre).

A frame is *closed* when the model's open-eye score (share of the iris inside the
open eye) is below 0.7, the threshold of the offline tool, or when the pupil is
not found at all.  A blink is a run of closed frames.  The eyelid already covers
part of the pupil just before the score drops and just after it recovers, and the
eye often makes a small movement on reopening, so samples from ``PRE_S`` before a
blink to ``POST_S`` after it are excluded as well.

Runs shorter than ``MIN_S`` are treated as single-frame dropouts, runs longer than
``MAX_S`` as an eye closure rather than a blink (both are still excluded, only the
count differs).
"""
import numpy as np
import pandas as pd

from vogeo_gaze.runtime.records import BLINK_OPEN_EYE_SCORE

PRE_S = 0.10
POST_S = 0.15
MIN_S = 0.04
MAX_S = 0.60
PUPIL_LOST_CONF = 0.15


def is_closed(open_eye, pupil_conf):
    return (open_eye < BLINK_OPEN_EYE_SCORE) | (pupil_conf < PUPIL_LOST_CONF)


def detect(t, closed):
    """(blinks DataFrame, padded mask) for timestamps ``t`` and closed flags.

    blinks: inizio_s, fine_s, durata_s, tipo ('ammiccamento' | 'chiusura' | 'perdita breve').
    padded: True for samples inside a blink or its PRE_S / POST_S margin.
    """
    t = np.asarray(t, np.float64)
    closed = np.asarray(closed, bool)
    padded = np.zeros(len(t), bool)
    rows = []
    if len(t) == 0:
        return pd.DataFrame(columns=["inizio_s", "fine_s", "durata_s", "tipo"]), padded
    edges = np.flatnonzero(np.diff(np.r_[0, closed.astype(np.int8), 0]))
    for start, stop in zip(edges[::2], edges[1::2]):      # closed[start:stop]
        t0 = t[start - 1] if start > 0 else t[start]      # last open frame before
        t1 = t[stop] if stop < len(t) else t[stop - 1]    # first open frame after
        dur = float(t[stop - 1] - t[start]) + float(np.median(np.diff(t))) if len(t) > 1 else 0.0
        kind = "perdita breve" if dur < MIN_S else "chiusura" if dur > MAX_S else "ammiccamento"
        rows.append({"inizio_s": float(t[start]), "fine_s": float(t[stop - 1]), "durata_s": dur, "tipo": kind})
        padded |= (t >= t0 - PRE_S) & (t <= t1 + POST_S)
    padded |= closed
    return pd.DataFrame(rows, columns=["inizio_s", "fine_s", "durata_s", "tipo"]), padded


def apply(samples):
    """List of tracker samples -> DataFrame with 'chiuso', 'blink' (padded) and
    'valid' updated to exclude the blink margins; plus the blinks table."""
    df = pd.DataFrame(samples)
    if df.empty:
        return df, pd.DataFrame(columns=["inizio_s", "fine_s", "durata_s", "tipo"])
    df = df.sort_values("t").reset_index(drop=True)
    df["chiuso"] = is_closed(df["open_eye"].to_numpy(), df["pupil_conf"].to_numpy())
    blinks, padded = detect(df["t"].to_numpy(), df["chiuso"].to_numpy())
    df["blink"] = padded
    df["valid_modello"] = df["valid"].astype(bool)
    df["valid"] = df["valid_modello"] & ~df["blink"]
    return df, blinks


class LiveCounter:
    """Blink count for the Live tab, updated one sample at a time (no look-ahead)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.count, self.closed, self.t_close, self.t_first = 0, False, None, None
        self.recent = []          # blink end times, for the rate

    def update(self, t, open_eye, pupil_conf):
        if self.t_first is None:
            self.t_first = t
        closed = bool(is_closed(np.float64(open_eye), np.float64(pupil_conf)))
        if closed and not self.closed:
            self.t_close = t
        elif not closed and self.closed and self.t_close is not None:
            if MIN_S <= t - self.t_close <= MAX_S:
                self.count += 1
                self.recent.append(t)
        self.closed = closed
        self.recent = [x for x in self.recent if x > t - 60.0]
        return closed

    def rate_per_min(self, t):
        span = min(60.0, t - (self.t_first or t))
        return len(self.recent) / span * 60.0 if span >= 10.0 else float("nan")
