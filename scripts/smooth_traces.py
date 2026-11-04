"""Steadiness filter: smooth each trace across slices, then re-score it.

  python scripts/smooth_traces.py --seed-set main|heldout|large [--sigma 2]

The ink drift test (scripts/letters_kaggle.py --shift-curve) found that
slice-to-slice wobble, not a steady offset, is what destroys the letters. Here
each fixed-tracer trace (curves_walk*) is stacked into a (slice x arc) grid
(scripts/unroll3d.grid_from_curves), smoothed along the slice axis with a
NaN-aware Gaussian (sigma in slices; light smoothing along the sheet), and
written back as per-slice curves to curves_smooth{sigma}*/ for scoring:
  python scripts/seeded_eval.py score --first-frame smooth2 --seed-set ...
No reference is used. Also reports a reference-free wobble measure: median
|second difference| of each grid column along the slices (voxels).
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.seeded_eval import OUT, Z0, _seeds, _suffix  # noqa: E402
from scripts.unroll3d import grid_from_curves, smooth_grid  # noqa: E402


def wobble(G):
    """Median |x[z+1] - 2x[z] + x[z-1]| over grid points (xy distance, voxels)."""
    d2 = G[2:, :, :2] - 2 * G[1:-1, :, :2] + G[:-2, :, :2]
    v = np.hypot(d2[..., 0], d2[..., 1])
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else None


def load(folder):
    curves = {}
    for f in sorted(folder.glob("[0-9]*.jsonl")):
        rec = json.loads(f.read_text().splitlines()[0])
        if len(rec["points"]) >= 2:
            curves[int(rec["source_z"])] = rec["points"]
    return curves


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-set", choices=["main", "heldout", "large"], default="main")
    ap.add_argument("--sigma", type=float, default=2.0, help="Gaussian sigma along the slices")
    a = ap.parse_args(argv)
    mode = f"smooth{a.sigma:g}"
    src = OUT / f"curves{_suffix(a, 'walk')}"
    dst = OUT / f"curves{_suffix(a, mode)}"
    if dst.exists():
        shutil.rmtree(dst)
    stats = []
    for r in _seeds(a):
        key = r.get("key", r["id"])
        curves = load(src / key)
        d = dst / key
        d.mkdir(parents=True, exist_ok=True)
        if len(curves) < 3:                      # nothing to smooth: keep the trace as it is
            if (src / key).exists():
                for f in (src / key).glob("[0-9]*.jsonl"):
                    shutil.copy2(f, d / f.name)
            continue
        G, zs = grid_from_curves(curves, r["seed"][:2])
        S = smooth_grid(G, sigma_rows=a.sigma, sigma_cols=1.0)
        stats.append({"key": key, "wobble_before": wobble(G), "wobble_after": wobble(S)})
        for f in (src / key).glob("[0-9]*.jsonl"):   # untraced slices stay untraced
            rec = json.loads(f.read_text().splitlines()[0])
            z = int(rec["source_z"])
            if z in curves:
                row = S[zs.index(z)]
                rec["points"] = np.round(row[np.isfinite(row[:, 0])][:, :2], 2).tolist()
                rec["tracker"] = f"walk+{mode}"
            (d / f.name).write_text(json.dumps(rec) + "\n")
    wb = [s["wobble_before"] for s in stats if s["wobble_before"] is not None]
    wa = [s["wobble_after"] for s in stats if s["wobble_after"] is not None]
    summary = {"sigma_slices": a.sigma, "traces": len(stats),
               "wobble_median_before": round(float(np.median(wb)), 3),
               "wobble_median_after": round(float(np.median(wa)), 3)}
    (OUT / f"wobble{_suffix(a, mode)}.json").write_text(json.dumps({"summary": summary, "traces": stats}, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
