"""Calibration: model gaze angles (theta_H, theta_V) -> screen pixels.

Each calibration point gives the robust centre (median after a MAD cut) of its
valid samples.  A second-order polynomial per screen axis is fitted by least
squares (terms 1, h, v, hv, h^2, v^2); with fewer than 8 usable points it falls
back to an affine map.  Inputs are standardised for conditioning.
"""
import numpy as np


def calibration_grid(n_points, width, height, margin=0.08):
    """Screen positions (px) of a calibration grid: 5, 9, 13, 17 or 25 points.

    9 = 3x3, 25 = 5x5; 13 = 3x3 plus the 4 centres of the 2x2 quadrants;
    17 = 13 plus the 4 edge midpoints between the outer points of a 5x5 grid.
    """
    def px(fx, fy):
        return (margin + fx * (1 - 2 * margin)) * width, (margin + fy * (1 - 2 * margin)) * height

    g3 = [(x, y) for y in (0, 0.5, 1) for x in (0, 0.5, 1)]
    if n_points == 5:
        pts = [(0.5, 0.5), (0, 0), (1, 0), (0, 1), (1, 1)]
    elif n_points == 9:
        pts = g3
    elif n_points == 13:
        pts = g3 + [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
    elif n_points == 17:
        pts = g3 + [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75),
                    (0.25, 0), (0.75, 0), (0.25, 1), (0.75, 1)]
    elif n_points == 25:
        pts = [(x, y) for y in np.linspace(0, 1, 5) for x in np.linspace(0, 1, 5)]
    else:
        raise ValueError(f"numero di punti non supportato: {n_points}")
    return [px(fx, fy) for fx, fy in pts]


def robust_center(theta, min_samples=5):
    """(N,2) samples -> median after dropping points > 3 MAD away, or None."""
    theta = np.asarray(theta, np.float64)
    theta = theta[np.isfinite(theta).all(axis=1)]
    if len(theta) < min_samples:
        return None
    med = np.median(theta, axis=0)
    d = np.linalg.norm(theta - med, axis=1)
    mad = np.median(d) * 1.4826 + 1e-6
    keep = d <= 3 * mad + 0.05
    return np.median(theta[keep], axis=0) if keep.sum() >= min_samples else med


def _terms(hv, order):
    h, v = hv[:, 0], hv[:, 1]
    cols = [np.ones_like(h), h, v]
    if order == 2:
        cols += [h * v, h * h, v * v]
    return np.stack(cols, axis=1)


class GazeMapper:
    def __init__(self, coef=None, mu=None, sd=None, order=2):
        self.coef, self.mu, self.sd, self.order = coef, mu, sd, order

    @classmethod
    def fit(cls, theta, screen, ridge=1e-3):
        """theta (P,2) robust point centres, screen (P,2) px targets."""
        theta, screen = np.asarray(theta, np.float64), np.asarray(screen, np.float64)
        order = 2 if len(theta) >= 8 else 1
        if len(theta) < 3:
            raise ValueError("servono almeno 3 punti di calibrazione validi")
        mu, sd = theta.mean(0), theta.std(0) + 1e-6
        A = _terms((theta - mu) / sd, order)
        reg = ridge * np.eye(A.shape[1])
        reg[0, 0] = 0.0
        coef = np.linalg.solve(A.T @ A + reg, A.T @ screen)
        return cls(coef, mu, sd, order)

    def map(self, theta):
        theta = np.atleast_2d(np.asarray(theta, np.float64))
        return _terms((theta - self.mu) / self.sd, self.order) @ self.coef

    def to_dict(self):
        return {"order": self.order, "coef": self.coef.tolist(), "mu": self.mu.tolist(), "sd": self.sd.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(np.array(d["coef"]), np.array(d["mu"]), np.array(d["sd"]), d["order"])


def fit_calibration(points):
    """points: [{"x", "y", "theta": (N,2) valid samples}] -> (mapper, report).

    report: per point centre, residual (px), number of samples; mean/max residual,
    and leave-one-out residuals as an honest estimate of the calibration error.
    """
    rows, used = [], []
    for i, p in enumerate(points):
        c = robust_center(p["theta"])
        rows.append({"punto": i + 1, "x": p["x"], "y": p["y"], "n_campioni": int(len(p["theta"])),
                     "theta_h": None if c is None else float(c[0]),
                     "theta_v": None if c is None else float(c[1])})
        if c is not None:
            used.append((i, c, (p["x"], p["y"])))
    if len(used) < 3:
        raise ValueError(f"solo {len(used)} punti con abbastanza campioni validi: ripetere la calibrazione")
    theta = np.array([u[1] for u in used])
    screen = np.array([u[2] for u in used])
    mapper = GazeMapper.fit(theta, screen)
    pred = mapper.map(theta)
    loo = []
    for k in range(len(used)):
        others = [j for j in range(len(used)) if j != k]
        try:
            m = GazeMapper.fit(theta[others], screen[others])
            loo.append(float(np.linalg.norm(m.map(theta[k:k + 1])[0] - screen[k])))
        except (ValueError, np.linalg.LinAlgError):
            loo.append(float("nan"))
    for (i, _, s), p, e_loo in zip(used, pred, loo):
        rows[i].update({"gaze_x": float(p[0]), "gaze_y": float(p[1]),
                        "residuo_px": float(np.linalg.norm(p - s)), "residuo_loo_px": e_loo})
    res = np.array([r["residuo_px"] for r in rows if "residuo_px" in r])
    report = {"punti": rows, "punti_usati": len(used), "ordine": mapper.order,
              "residuo_medio_px": float(res.mean()), "residuo_max_px": float(res.max()),
              "residuo_loo_medio_px": float(np.nanmean(loo)) if np.isfinite(loo).any() else None}
    return mapper, report
