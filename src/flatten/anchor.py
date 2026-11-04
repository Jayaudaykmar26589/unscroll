"""Flatten anchoring: 512-pt resample, global origin alignment, shear.

Reads per-slice curve JSONLs {z, points, origin, shift}, writes anchored
flat-sheet coordinates + per-slice shift log. Shear metric = |median
glyph aspect - 1| on Tier A val (needs letter boxes; computed in eval).
"""
import json
from pathlib import Path

import numpy as np


def resample(points, n=512):
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return np.zeros((0, 2))
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    total = s[-1] or 1.0
    out = np.empty((n, 2))
    tgt = np.linspace(0, total, n)
    out[:, 0] = np.interp(tgt, s, pts[:, 0])
    out[:, 1] = np.interp(tgt, s, pts[:, 1])
    return out


def _shift_open(c, k):
    """Translate open curve along arc length; clamp ends (no wrap)."""
    idx = np.clip(np.arange(len(c)) - k, 0, len(c) - 1)
    return c[idx]


def align_origins(curves, max_shift=64):
    """Global alignment: minimize total inter-slice shift.

    curves: list of (N,2) OPEN segments. Returns (aligned, shifts) where
    shifts[z] is the linear arc-length shift applied (clamped ends).
    Greedy chain from slice 0 (global objective approximated).
    """
    if not curves:
        return [], []
    aligned, shifts = [curves[0]], [0]
    for c in curves[1:]:
        best_k, best_d = 0, float("inf")
        prev = aligned[-1]
        # Skeleton traversal direction can reverse between slices. Compare
        # corresponding samples before searching arc-length shifts.
        if np.linalg.norm(c[::-1] - prev) < np.linalg.norm(c - prev):
            c = c[::-1]
        for k in range(-max_shift, max_shift + 1, 4):  # coarse
            d = np.linalg.norm(_shift_open(c, k) - prev)
            if d < best_d:
                best_k, best_d = k, d
        for k in range(best_k - 4, best_k + 5):  # refine
            if abs(k) > max_shift:
                continue
            d = np.linalg.norm(_shift_open(c, k) - prev)
            if d < best_d:
                best_k, best_d = k, d
        aligned.append(_shift_open(c, best_k))
        shifts.append(best_k)
    return aligned, shifts


def anchor_curves(curve_dir, out_path):
    curve_dir = Path(curve_dir)
    files = sorted(curve_dir.glob("*.jsonl"))
    records = [json.loads(f.read_text()) for f in files]
    records.sort(key=lambda r: r["z"])
    if len({r["z"] for r in records}) != len(records):
        raise ValueError("duplicate curve slice indices")
    curves, kept = [], []
    for rec in records:
        if len(rec["points"]) >= 2:
            curves.append(resample(rec["points"]))
            kept.append(rec)
    aligned, shifts = align_origins(curves)
    sheet = np.stack(aligned) if aligned else np.zeros((0, 512, 2))
    log = [{**{k: rec[k] for k in ("z", "source_z", "source_file") if k in rec},
            "shift": int(s)} for rec, s in zip(kept, shifts)]
    out = {"sheet_shape": list(sheet.shape),
           "shifts": log,
           "missing_z": [r["z"] for r in records if len(r["points"]) < 2],
           "note": "sheet[row, i] = (x, y) full-res coords; "
                   "shifts[row] identifies the source slice. Missing slices "
                   "are omitted, not interpolated; rows need not be adjacent z."}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(str(Path(out_path).with_suffix(".npy")), sheet)
    Path(out_path).write_text(json.dumps(out, indent=1))
    return out_path, sheet.shape
