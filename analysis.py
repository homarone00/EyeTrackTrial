"""Experiment error, per trial and per category (target type, screen region, eccentricity).

Samples are matched with events by timestamp.  Blinks are detected first
(blinks.py): samples inside a blink or its margins are invalid.  The error of a
sample is the Euclidean distance (px) between the mapped gaze and the target
position at that time; a trial's error is the median over its valid samples in the
analysis window.  Smooth pursuit is split into 0.5 s segments, each counted as one
trial at its mean target position, so it can also be split by region and
eccentricity.  A trial repeated after blinks is analysed on its last attempt.
"""
import math

import numpy as np
import pandas as pd

import blinks as blk
from stimuli import TYPE_LABELS, pursuit_position

REGION_NAMES = ["Alto-sinistra", "Alto-centro", "Alto-destra",
                "Centro-sinistra", "Centro", "Centro-destra",
                "Basso-sinistra", "Basso-centro", "Basso-destra"]
PURSUIT_SEGMENT_S = 0.5
MIN_VALID_SAMPLES = 3
N_ECC_BINS = 4
REPEAT_MIN_VALID_FRACTION = 0.5   # a trial with less valid data than this is repeated
REPEAT_MIN_VALID_SAMPLES = 5


def region_of(x, y, width, height):
    col = min(2, max(0, int(3 * x / width)))
    row = min(2, max(0, int(3 * y / height)))
    return REGION_NAMES[3 * row + col]


def ecc_edges(width, height, n=N_ECC_BINS):
    return np.linspace(0.0, math.hypot(width / 2, height / 2), n + 1)


def ecc_label(e, edges):
    k = int(np.clip(np.searchsorted(edges, e, side="right") - 1, 0, len(edges) - 2))
    return f"{edges[k]:.0f}–{edges[k + 1]:.0f} px"


def _group(ev):
    return ev.get("prova", ev.get("punto"))


def last_attempts(events):
    """Only the last attempt of every trial / calibration point."""
    last = {}
    for ev in events:
        if ev.get("t_off") is not None:
            last[_group(ev)] = ev
    return [ev for ev in events if last.get(_group(ev)) is ev]


def lost_to_blinks(samples, events):
    """Groups (trial or calibration-point ids) of the latest attempts whose analysis
    window has too little valid data, mostly because of blinks."""
    df, _ = blk.apply(samples)
    if df.empty:
        return {_group(ev) for ev in last_attempts(events)}
    t, valid = df["t"].to_numpy(), df["valid"].to_numpy()
    bad = set()
    for ev in last_attempts(events):
        sel = (t >= ev["t_a0"]) & (t <= ev["t_off"])
        n, n_valid = int(sel.sum()), int(valid[sel].sum())
        if n_valid < REPEAT_MIN_VALID_SAMPLES or n_valid < REPEAT_MIN_VALID_FRACTION * max(n, 1):
            bad.add(_group(ev))
    return bad


def calibration_points(samples, events):
    """Valid (blink-free) theta samples per calibration point, pooled over attempts."""
    df, blinks = blk.apply(samples)
    points = {}
    for ev in events:
        if ev.get("t_off") is None:
            continue
        p = points.setdefault(ev["punto"], {"x": ev["x"], "y": ev["y"], "theta": []})
        if not df.empty:
            sel = (df["t"] >= ev["t_a0"]) & (df["t"] <= ev["t_off"]) & df["valid"]
            p["theta"].append(df.loc[sel, ["theta_h", "theta_v"]].to_numpy())
    out = []
    for k in sorted(points):
        p = points[k]
        p["theta"] = np.concatenate(p["theta"]) if p["theta"] else np.zeros((0, 2))
        out.append(p)
    return out, df, blinks


def analyze(samples, events, mapper, width, height):
    """-> (samples_df with per-sample error, trials_df, {category: summary_df}, blinks_df)."""
    df, blinks = blk.apply(samples)
    edges = ecc_edges(width, height)
    cx, cy = width / 2, height / 2
    if df.empty:
        return df, pd.DataFrame(), {}, blinks
    xy = mapper.map(df[["theta_h", "theta_v"]].to_numpy())
    df["gaze_x"], df["gaze_y"] = xy[:, 0], xy[:, 1]
    t = df["t"].to_numpy()
    df["prova"] = -1
    df["tentativo"] = 0
    df["target_x"] = np.nan
    df["target_y"] = np.nan

    # Which trial each blink happened in (any attempt).
    blinks["prova"] = -1
    blinks["tentativo"] = 0
    for ev in events:
        if ev.get("t_off") is None:
            continue
        inside = (blinks["inizio_s"] >= ev["t_on"]) & (blinks["inizio_s"] <= ev["t_off"])
        blinks.loc[inside, "prova"] = ev["prova"]
        blinks.loc[inside, "tentativo"] = ev.get("tentativo", 0)
    n_attempts = {}
    for ev in events:
        if ev.get("t_off") is not None:
            n_attempts[ev["prova"]] = n_attempts.get(ev["prova"], 0) + 1

    trials = []
    for ev in last_attempts(events):
        idx = np.flatnonzero((t >= ev["t_a0"]) & (t <= ev["t_off"]))
        df.loc[idx, "prova"] = ev["prova"]
        df.loc[idx, "tentativo"] = ev.get("tentativo", 0)
        extra = {"tentativi": n_attempts[ev["prova"]],
                 "ammiccamenti": int(((blinks["prova"] == ev["prova"])
                                      & (blinks["tentativo"] == ev.get("tentativo", 0))
                                      & (blinks["tipo"] == "ammiccamento")).sum())}
        if ev["tipo"] == "inseguimento":
            if len(idx):
                pos = np.array([pursuit_position(ev["path"], tt - ev["t_on"]) for tt in t[idx]])
                df.loc[idx, "target_x"], df.loc[idx, "target_y"] = pos[:, 0], pos[:, 1]
            seg_id = np.floor((t[idx] - ev["t_a0"]) / PURSUIT_SEGMENT_S).astype(int)
            for s in np.unique(seg_id):
                trials.append(_trial_row(df, idx[seg_id == s], ev, None, None, s, extra))
        else:
            df.loc[idx, "target_x"], df.loc[idx, "target_y"] = ev["x"], ev["y"]
            trials.append(_trial_row(df, idx, ev, ev["x"], ev["y"], None, extra))
    df["errore_px"] = np.hypot(df["gaze_x"] - df["target_x"], df["gaze_y"] - df["target_y"])

    tr = pd.DataFrame(trials)
    if tr.empty:
        return df, tr, {}, blinks
    tr["regione"] = [region_of(x, y, width, height) for x, y in zip(tr["target_x"], tr["target_y"])]
    tr["eccentricita_px"] = np.hypot(tr["target_x"] - cx, tr["target_y"] - cy)
    tr["fascia_eccentricita"] = [ecc_label(e, edges) for e in tr["eccentricita_px"]]
    summaries = {
        "tipo": summarize(tr, "tipo_label", [TYPE_LABELS[k] for k in TYPE_LABELS]),
        "regione": summarize(tr, "regione", REGION_NAMES),
        "eccentricita": summarize(tr, "fascia_eccentricita",
                                  [ecc_label((a + b) / 2, edges) for a, b in zip(edges[:-1], edges[1:])]),
    }
    return df, tr, summaries, blinks


def blink_summary(blinks, duration_s):
    """Counts, rate and durations of the blinks of a recording."""
    real = blinks[blinks["tipo"] == "ammiccamento"] if len(blinks) else blinks
    return {"ammiccamenti": int(len(real)),
            "chiusure_lunghe": int((blinks["tipo"] == "chiusura").sum()) if len(blinks) else 0,
            "perdite_brevi": int((blinks["tipo"] == "perdita breve").sum()) if len(blinks) else 0,
            "frequenza_per_min": float(len(real) / duration_s * 60.0) if duration_s > 0 else float("nan"),
            "durata_mediana_ms": float(real["durata_s"].median() * 1000) if len(real) else float("nan")}


def _trial_row(df, idx, ev, x, y, segment, extra):
    part = df.iloc[idx]
    valid = part[part["valid"].astype(bool)]
    if x is None:  # pursuit segment: mean target position
        x, y = float(part["target_x"].mean()), float(part["target_y"].mean())
    row = {"prova": ev["prova"], "segmento": segment, "tipo": ev["tipo"], "tipo_label": TYPE_LABELS[ev["tipo"]],
           "target_x": x, "target_y": y, "n_campioni": len(part), "n_validi": len(valid),
           "n_persi_blink": int((part["blink"] & part["valid_modello"]).sum() + part["chiuso"].sum()),
           **extra}
    if len(valid) >= MIN_VALID_SAMPLES:
        ex = valid["gaze_x"].to_numpy() - valid["target_x"].to_numpy()
        ey = valid["gaze_y"].to_numpy() - valid["target_y"].to_numpy()
        err = np.hypot(ex, ey)
        row.update({"errore_px": float(np.median(err)), "errore_medio_px": float(err.mean()),
                    "offset_x_px": float(np.median(ex)), "offset_y_px": float(np.median(ey)),
                    "precisione_rms_px": float(np.sqrt(np.mean(np.diff(valid["gaze_x"]) ** 2
                                                               + np.diff(valid["gaze_y"]) ** 2)))
                    if len(valid) > 1 else np.nan,
                    "gaze_x": float(np.median(valid["gaze_x"])), "gaze_y": float(np.median(valid["gaze_y"]))})
    else:
        row.update({"errore_px": np.nan, "errore_medio_px": np.nan, "offset_x_px": np.nan, "offset_y_px": np.nan,
                    "precisione_rms_px": np.nan, "gaze_x": np.nan, "gaze_y": np.nan})
    return row


def summarize(tr, column, order):
    rows = []
    groups = [(name, tr[tr[column] == name]) for name in order if (tr[column] == name).any()]
    groups.append(("Totale", tr))
    for name, g in groups:
        e = g["errore_px"].dropna()
        # Blinks and repeats are per trial: count each trial once (pursuit has several segments).
        per_trial = g.drop_duplicates("prova")
        rows.append({
            "categoria": name,
            "n_prove": int(len(g)),
            "n_prove_valide": int(len(e)),
            "errore_mediano_px": float(e.median()) if len(e) else np.nan,
            "errore_medio_px": float(e.mean()) if len(e) else np.nan,
            "ds_px": float(e.std()) if len(e) > 1 else np.nan,
            "q1_px": float(e.quantile(0.25)) if len(e) else np.nan,
            "q3_px": float(e.quantile(0.75)) if len(e) else np.nan,
            "precisione_rms_px": float(g["precisione_rms_px"].median()) if g["precisione_rms_px"].notna().any() else np.nan,
            "campioni_validi_pct": float(100 * g["n_validi"].sum() / max(g["n_campioni"].sum(), 1)),
            "persi_blink_pct": float(100 * g["n_persi_blink"].sum() / max(g["n_campioni"].sum(), 1)),
            "ammiccamenti": int(per_trial["ammiccamenti"].sum()),
            "prove_ripetute": int((per_trial["tentativi"] > 1).sum()),
        })
    return pd.DataFrame(rows)
