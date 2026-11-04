"""Score traced curves against synthetic ground truth.

  python scripts/synth_score.py --synth data/synth/crush2 --curves curves_synth/crush2

The target sheet is identified by its spiral angle theta, which is continuous
along the physical sheet (a winding index is not: it changes at angle 0). The
click's nearest centre-line point on the first traced slice fixes theta_c.
For each traced point, find the nearest ground-truth centre-line point:
  on_target  - within --tol px and |theta - theta_c| < pi (same sheet)
  jumped     - within --tol px and |theta - theta_c| >= pi (another winding)
  off_sheet  - farther than --tol px from any sheet centre line
Writes score.json next to the curves and prints a summary. NumPy only.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

CHUNK = 256


def nearest(points, gx, gy):
    """Index and distance of the nearest ground-truth point for each point."""
    idx = np.empty(len(points), dtype=np.int64)
    dist = np.empty(len(points), dtype=np.float64)
    for s in range(0, len(points), CHUNK):
        p = points[s:s + CHUNK]
        d2 = (p[:, 0:1] - gx[None]) ** 2 + (p[:, 1:2] - gy[None]) ** 2
        j = np.argmin(d2, axis=1)
        idx[s:s + CHUNK] = j
        dist[s:s + CHUNK] = np.sqrt(d2[np.arange(len(p)), j])
    return idx, dist


def score(synth_dir, curves_dir, tol=2.0, click=None):
    synth_dir, curves_dir = Path(synth_dir), Path(curves_dir)
    meta = json.loads((synth_dir / "meta.json").read_text())
    truth = np.load(synth_dir / "truth.npz")
    files = sorted(curves_dir.glob("[0-9]*.jsonl"))
    if not files:
        raise SystemExit(f"no curve files in {curves_dir}")
    records = [json.loads(f.read_text().splitlines()[0]) for f in files]
    records.sort(key=lambda r: r["source_z"])
    click = click or meta["click"]
    theta_c = None
    rows = []
    for rec in records:
        z = rec["source_z"]
        gx, gy, gt = truth[f"x_{z:05d}"], truth[f"y_{z:05d}"], truth[f"t_{z:05d}"]
        if theta_c is None:
            j, _ = nearest(np.array([click[:2]], dtype=float), gx, gy)
            theta_c = float(gt[j[0]])
        pts = np.asarray(rec["points"], dtype=float).reshape(-1, 2)
        if len(pts) == 0:
            rows.append({"z": z, "points": 0, "on_target": 0.0, "jumped": 0.0,
                         "off_sheet": 0.0, "mean_dist_target": None, "length_px": 0.0})
            continue
        j_any, d_any = nearest(pts, gx, gy)
        same = np.abs(gt - theta_c) < np.pi
        _, d_tgt = nearest(pts, gx[same], gy[same])
        near = d_any <= tol
        on = near & same[j_any]
        jump = near & ~same[j_any]
        length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        rows.append({"z": z, "points": int(len(pts)), "on_target": float(on.mean()),
                     "jumped": float(jump.mean()), "off_sheet": float((~near).mean()),
                     "mean_dist_target": float(d_tgt.mean()), "length_px": round(length, 1)})
    traced = [r for r in rows if r["points"]]
    summary = {
        "synth": str(synth_dir), "curves": str(curves_dir), "crush": meta["config"]["crush"],
        "view": meta.get("view"), "click": click, "theta_c": theta_c, "tol_px": tol,
        "slices": len(rows), "slices_with_trace": len(traced),
        "on_target_mean": round(float(np.mean([r["on_target"] for r in traced])), 4) if traced else 0.0,
        "on_target_min": round(float(min(r["on_target"] for r in traced)), 4) if traced else 0.0,
        "jumped_mean": round(float(np.mean([r["jumped"] for r in traced])), 4) if traced else 0.0,
        "off_sheet_mean": round(float(np.mean([r["off_sheet"] for r in traced])), 4) if traced else 0.0,
        "slices_any_jump": sum(r["jumped"] > 0.05 for r in traced),
        "mean_dist_target_px": round(float(np.mean([r["mean_dist_target"] for r in traced])), 3) if traced else None,
        "mean_length_px": round(float(np.mean([r["length_px"] for r in traced])), 1) if traced else 0.0,
    }
    (curves_dir / "score.json").write_text(json.dumps({"summary": summary, "slices": rows}, indent=2))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", required=True)
    ap.add_argument("--curves", required=True)
    ap.add_argument("--tol", type=float, default=2.0)
    ap.add_argument("--click", default="", help="x,y,z if not the meta.json default")
    a = ap.parse_args(argv)
    click = [int(v) for v in a.click.split(",")] if a.click else None
    print(json.dumps(score(a.synth, a.curves, a.tol, click), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
