"""How good is the reference? The 2023 human-traced layer vs the community's 2026 re-trace.

  python scripts/reference_check.py     -> reports/seeded_eval/reference_check.json

The Vesuvius Challenge open data bucket holds a 2026 version of segment
20231210121321 (20260604223808-20231210121321_v8) mapped onto the same 7.91 um
volume Unscroll uses (tifxyz, 20-voxel grid). This compares, slice by slice:
  1. the 2023 reference used by every Unscroll score vs the 2026 re-trace;
  2. Unscroll's traces on that sheet (all clicks on it, fixed tracer, and the
     100-slice trace) scored against both.
Data: CC BY-NC 4.0, Vesuvius Challenge (s3://vesuvius-challenge-open-data/PHercParis4/).
"""
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.seeded_eval import OUT, _suffix, _window_mesh  # noqa: E402
from src.eval.meshref import SurfaceIndex, curve_xyz, sample_curve, slice_segments  # noqa: E402

SEG = "20231210121321"
V8 = ROOT / "data" / "cache" / "v8_7.91um.tifxyz"
Z_LO, Z_HI = 6090, 6170


def v8_mesh():
    x, y, z = (tifffile.imread(V8 / f"{a}.tif").astype(float) for a in "xyz")
    P = np.stack([x, y, z], -1)
    ok = (z > 0) & (x != -1)
    H, W = z.shape
    idx = np.arange(H * W).reshape(H, W)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    q = ok[:-1, :-1] & ok[:-1, 1:] & ok[1:, :-1] & ok[1:, 1:]
    f = np.concatenate([np.stack([a[q], b[q], c[q]], 1), np.stack([b[q], d[q], c[q]], 1)])
    v = P.reshape(-1, 3)
    zt = v[f][:, :, 2]
    f = f[(zt.max(1) >= Z_LO) & (zt.min(1) <= Z_HI)]
    used = np.unique(f)
    remap = np.full(len(v), -1)
    remap[used] = np.arange(len(used))
    return v[used], remap[f]


def trace_points(folder):
    pts, wts = [], []
    for c in sorted(Path(folder).glob("[0-9]*.jsonl")):
        rec = json.loads(c.read_text().splitlines()[0])
        if len(rec["points"]) >= 2:
            q, w = sample_curve(curve_xyz(rec), 1.0)
            pts.append(q)
            wts.append(w)
    return (np.concatenate(pts), np.concatenate(wts)) if pts else (None, None)


def stats(d, w=None):
    w = np.ones(len(d)) if w is None else w / w.sum() * len(d)
    wn = w / w.sum()
    return {"within2": round(float(wn[d <= 2].sum()), 4), "within5": round(float(wn[d <= 5].sum()), 4),
            "median_vx": round(float(np.median(d)), 3), "points": int(len(d))}


def main():
    v8 = SurfaceIndex(*v8_mesh())
    old_v, old_f = _window_mesh(SEG)
    old = SurfaceIndex(old_v, old_f)
    # 1. the 2023 reference, sampled on each slice, measured against the 2026 re-trace
    pts = []
    for z in range(6100, 6150):
        seg = slice_segments(old_v, old_f, z)
        if len(seg):
            p = np.concatenate([seg[:, 0], seg[:, 1]])[::3, :2]       # slice_segments gives x, y
            pts.append(np.column_stack([p, np.full(len(p), float(z))]))
    pts = np.concatenate(pts)
    d_ref = v8.distances(pts)
    report = {"segment": SEG, "reference_2026": "20260604223808-20231210121321_v8, on volume 20230205180739 (7.91 um)",
              "old_reference_vs_2026": stats(d_ref), "traces": []}
    # 2. Unscroll's traces on this sheet
    folders = [("dev", OUT / "curves_walk" / SEG), ("heldout", OUT / "curves_walk_heldout" / SEG)]
    folders += [(f"large_{k}", OUT / "curves_walk_large" / f"{SEG}_{k}") for k in range(4)]
    folders += [("w100_verified", ROOT / "reports/flatten/curves_20231210121321_kaggle")]
    for name, folder in folders:
        p, w = trace_points(folder)
        if p is None:
            continue
        report["traces"].append({"trace": name, "vs_2023_reference": stats(old.distances(p), w),
                                 "vs_2026_reference": stats(v8.distances(p), w)})
    (OUT / "reference_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["old_reference_vs_2026"]))
    for t in report["traces"]:
        print(t["trace"], "2023:", t["vs_2023_reference"], "2026:", t["vs_2026_reference"])


if __name__ == "__main__":
    main()
