"""Jump guards: winding / spacing / smoothness. Numpy + yaml."""

from pathlib import Path

import numpy as np
import yaml

DEFAULTS = {"winding_reversal_deg": 5.0, "spacing_frac": 0.30,
            "spacing_window": 10, "kink_deg": 60.0}


def load_cfg(path=None):
    cfg = dict(DEFAULTS)
    if path and Path(path).exists():
        cfg.update(yaml.safe_load(Path(path).read_text()))
    return cfg


def check_winding(cum_angle_deg, tol_deg=5.0):
    """Fail idx where cumulative winding reverses by more than tol."""
    a = np.asarray(cum_angle_deg, dtype=float)
    d = np.diff(a)
    bad = np.where(d < -tol_deg)[0] + 1
    return {"violations": [int(i) for i in bad],
            "pass": bool(len(bad) == 0)}


def check_spacing(gaps, window=10, frac=0.30):
    """Fail idx where gap deviates >frac from trailing-window median."""
    g = np.asarray(gaps, dtype=float)
    bad = []
    for i in range(len(g)):
        med = np.median(g[max(0, i - window):i] if i > 0 else g[i:i + 1])
        if med > 0 and abs(g[i] - med) / med > frac:
            bad.append(int(i))
    return {"violations": bad, "pass": bool(not bad)}


def check_curvature(points_xy, kink_deg=60.0):
    """Fail idx where adjacent-segment turn angle exceeds kink."""
    p = np.asarray(points_xy, dtype=float)
    v = np.diff(p, axis=0)
    n = np.linalg.norm(v, axis=1)
    n[n == 0] = 1.0
    u = v / n[:, None]
    dots = np.clip((u[:-1] * u[1:]).sum(1), -1.0, 1.0)
    turns = np.degrees(np.arccos(dots))
    bad = [int(i + 1) for i, t in enumerate(turns) if t > kink_deg]
    return {"violations": bad, "pass": bool(not bad)}
