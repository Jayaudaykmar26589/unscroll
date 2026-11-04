"""Flatten one traced sheet into an image: rows = slices, columns = voxels along the sheet.

  python scripts/flatten_trace.py --curves reports/flatten/curves_20231210121321_kaggle \
      --click 3493,2426 --out reports/flatten/sheet_20231210121321

Each slice's trace is resampled at 1-voxel arc spacing. Columns are aligned on
an anchor carried from slice to slice (the trace point nearest the previous
anchor, starting at the click), so a column is the same physical place on the
sheet in every row and one pixel is one voxel (7.91 um) in both directions.
Intensities are sampled bilinearly along the local normal at offsets
-W..+W voxels, giving a thin surface volume. Writes surface.npy (rows, cols,
layers; NaN outside the trace), flat_mean.png, flat_layers.png and meta.json.
NumPy only.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import tifffile  # noqa: E402
from PIL import Image  # noqa: E402

SLICES = ROOT / "data" / "cache" / "scroll1_6100_6200"


def resample(points, step=1.0):
    p = np.asarray(points, float)
    seg = np.hypot(*np.diff(p, axis=0).T)
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] < 2:
        return np.empty((0, 2))
    q = np.arange(0.0, s[-1], step)
    return np.column_stack([np.interp(q, s, p[:, 0]), np.interp(q, s, p[:, 1])])


def normals(p):
    t = np.gradient(p, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    return np.column_stack([-t[:, 1], t[:, 0]])


def bilinear(img, x, y):
    h, w = img.shape
    x0 = np.clip(np.floor(x).astype(int), 0, w - 2)
    y0 = np.clip(np.floor(y).astype(int), 0, h - 2)
    fx, fy = np.clip(x - x0, 0, 1), np.clip(y - y0, 0, 1)
    a, b = img[y0, x0], img[y0, x0 + 1]
    c, d = img[y0 + 1, x0], img[y0 + 1, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def _png(a, path):
    v = a[np.isfinite(a)]
    lo, hi = np.percentile(v, (1, 99)) if len(v) else (0, 1)
    u8 = np.clip((np.nan_to_num(a, nan=lo) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(u8).save(path)


def flatten(curve_dir, click, out, half_width=6, slices=SLICES):
    curve_dir, out = Path(curve_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows, anchor = [], np.asarray(click, float)
    for f in sorted(curve_dir.glob("[0-9]*.jsonl")):
        rec = json.loads(f.read_text().splitlines()[0])
        p = resample(rec["points"]) if len(rec["points"]) > 1 else np.empty((0, 2))
        if not len(p):
            rows.append((rec["source_z"], None, None))
            continue
        i = int(np.argmin(np.hypot(*(p - anchor).T)))
        anchor = p[i]
        rows.append((rec["source_z"], p, i))
    left = max(i for _, p, i in rows if p is not None)
    right = max(len(p) - i for _, p, i in rows if p is not None)
    offsets = np.arange(-half_width, half_width + 1)
    vol = np.full((len(rows), left + right, len(offsets)), np.nan, np.float32)
    for r, (z, p, i) in enumerate(rows):
        if p is None:
            continue
        img = tifffile.imread(slices / f"{z:05d}.tif").astype(np.float32)
        n = normals(p)
        cols = np.arange(len(p)) - i + left
        for k, o in enumerate(offsets):
            q = p + o * n
            vol[r, cols, k] = bilinear(img, q[:, 0], q[:, 1])
    np.save(out / "surface.npy", vol)
    traced = [r for r, (_, p, _) in enumerate(rows) if p is not None]
    first, last = traced[0], traced[-1]          # images cover traced slices only
    vol = vol[first:last + 1]
    with warnings.catch_warnings():             # all-NaN columns at ragged trace ends
        warnings.simplefilter("ignore", RuntimeWarning)
        _png(np.nanmean(vol, axis=2), out / "flat_mean.png")
    # every 3rd layer stacked vertically: ink, if any, sits on one face of the sheet
    picks = list(range(0, len(offsets), 3))
    gap = np.full((3, vol.shape[1]), np.nan, np.float32)
    _png(np.vstack([np.vstack([vol[:, :, k], gap]) for k in picks]), out / "flat_layers.png")
    meta = {"curves": str(curve_dir), "click": list(map(float, click)), "rows": len(rows),
            "rows_traced": sum(p is not None for _, p, _ in rows), "columns": int(vol.shape[1]),
            "anchor_column": int(left), "normal_offsets_vx": offsets.tolist(),
            "layers_in_flat_layers_png": [int(offsets[k]) for k in picks],
            "z_range": [rows[0][0], rows[-1][0]],
            "z_range_traced": [rows[first][0], rows[last][0]], "image_rows": last - first + 1,
            "voxel_um": 7.91}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", required=True)
    ap.add_argument("--click", required=True, help="x,y of the seed click (volume coordinates)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--half-width", type=int, default=6)
    a = ap.parse_args(argv)
    print(json.dumps(flatten(a.curves, [float(v) for v in a.click.split(",")[:2]], a.out, a.half_width)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
