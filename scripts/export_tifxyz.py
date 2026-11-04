"""Export an Unscroll trace as a tifxyz surface that VC3D and the Vesuvius tools can open.

  python scripts/export_tifxyz.py --curves reports/seeded_eval/curves_walk/20231210121321 \
      --click 3309,2497 --out exports/tifxyz/20231210121321 [--sigma 2]

Format (ScrollPrize/villa lasagna/tifxyz_format.md): a folder with x.tif,
y.tif, z.tif (float32, identical size; one grid vertex per pixel, volume voxel
coordinates) and meta.json with "format": "tifxyz" and "scale" (grid cells per
voxel). Rows are CT slices (1 voxel apart), columns are 1 voxel of arc along
the sheet, so scale = [1, 1]. Invalid vertices are (-1, -1, -1).
--sigma smooths along the slices first (scripts/smooth_traces.py's filter).
"""
import argparse
import json
import sys
import uuid
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.smooth_traces import load  # noqa: E402
from scripts.unroll3d import grid_from_curves, smooth_grid  # noqa: E402

VOLUME = "20230205180739"


def export(curves_dir, click_xy, out, sigma=0.0, name=None):
    curves = load(Path(curves_dir))
    if not curves:
        raise SystemExit(f"no curves in {curves_dir}")
    G, zs = grid_from_curves(curves, click_xy)
    if sigma > 0:
        z = G[..., 2].copy()
        G = smooth_grid(G, sigma_rows=sigma, sigma_cols=1.0)
        G[..., 2] = z                        # each row stays on its CT slice; only x, y are smoothed
    valid = np.isfinite(G).all(-1) & (G[..., 2] > 0)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for i, ax in enumerate("xyz"):
        a = np.where(valid, G[..., i], -1.0).astype(np.float32)
        tifffile.imwrite(out / f"{ax}.tif", a)
    P = G[valid]
    meta = {"format": "tifxyz", "type": "seg", "uuid": name or out.name,
            "scale": [1.0, 1.0],
            "bbox": [P.min(0).round(2).tolist(), P.max(0).round(2).tolist()],
            "volume": VOLUME,
            "source": "Unscroll SAM 3.1 trace (predicted surface, not released reference geometry)",
            "slices": [min(zs), max(zs)], "smoothing_sigma_slices": sigma,
            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"unscroll/{out.name}/{sigma}"))}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return G.shape[:2], int(valid.sum())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", required=True)
    ap.add_argument("--click", required=True, help="x,y of the click on the first slice")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigma", type=float, default=0.0)
    a = ap.parse_args(argv)
    shape, n = export(a.curves, [float(v) for v in a.click.split(",")[:2]], a.out, a.sigma)
    print(f"tifxyz {a.out}: grid {shape[0]} x {shape[1]}, {n} valid vertices")


if __name__ == "__main__":
    main()
