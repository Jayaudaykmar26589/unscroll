"""Normalize + region-split writer. Numpy (+ tifffile optional).

  python -m src.dataio.preprocess --input vol.npy --stats-out DIR
Splits: regions are {region_id, bbox}; 80/20 by region, seed 0.
Scroll 1 eval crop must not spatially overlap train regions.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import tifffile  # noqa: F401
    HAS_TIF = True
except ImportError:
    HAS_TIF = False


def load_volume(path):
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    if HAS_TIF and path.suffix in (".tif", ".tiff"):
        import tifffile
        return tifffile.imread(path)
    raise ValueError(f"unsupported format (tifffile present={HAS_TIF}): {path}")


def normalize(vol):
    vol = np.asarray(vol, dtype=np.float64)
    mean, std = float(vol.mean()), float(vol.std()) or 1.0
    return ((vol - mean) / std).astype(np.float32), {"mean": mean, "std": std}


def bbox_overlap(a, b):
    """BBoxes as ((z0,z1),(y0,y1),(x0,x1)); True if volumes intersect."""
    return all(lo1 < hi2 and lo2 < hi1
               for (lo1, hi1), (lo2, hi2) in zip(a, b))


def make_splits(regions, seed=0, train_frac=0.8):
    """regions: list of {region_id, bbox}. Deterministic by seed."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(regions))
    rng.shuffle(idx)
    cut = int(len(regions) * train_frac)
    out = []
    for k, i in enumerate(idx):
        r = dict(regions[i])
        r["split"] = "train" if k < cut else "val"
        out.append(r)
    return out


def check_eval_overlap(eval_bbox, train_regions):
    bad = [r["region_id"] for r in train_regions
           if bbox_overlap(eval_bbox, r["bbox"])]
    if bad:
        raise ValueError(f"eval crop overlaps train regions: {bad}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--stats-out", required=True)
    ap.add_argument("--regions", default="",
                    help="JSON list of {region_id, bbox} for split writer")
    a = ap.parse_args(argv)
    vol = load_volume(a.input)
    norm, stats = normalize(vol)
    out = Path(a.stats_out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "norm_stats.json").write_text(json.dumps(stats, indent=1))
    np.save(out / (Path(a.input).stem + "_norm.npy"), norm)
    print(f"normalized {vol.shape} {vol.dtype} -> {out}")
    if a.regions:
        regions = json.loads(a.regions)
        splits = make_splits(regions)
        (out / "splits.json").write_text(json.dumps(splits, indent=1))
        print(f"wrote splits for {len(splits)} regions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
