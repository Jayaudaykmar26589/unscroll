"""Day-1 fragment prep: streaming norm stats + region splits.

Fragments are whole-volume regions (frag1=train, frag3=val). Stats via
streaming slice reads (full float64 volume would exceed RAM). Training
normalizes patches on the fly; no normalized copy is written.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import tifffile  # noqa: E402

from src.dataio.preprocess import make_splits  # noqa: E402


def streaming_stats(tifs):
    n, mean, m2 = 0, 0.0, 0.0
    shape = None
    for t in tifs:
        a = tifffile.imread(t).astype(np.float64).ravel()
        shape = tifffile.imread(t).shape if shape is None else shape
        for x in (a.mean(),):  # batch update via moments
            b_n, b_mean = a.size, float(x)
            b_m2 = float(((a - b_mean) ** 2).sum())
            delta = b_mean - mean
            tot = n + b_n
            mean += delta * b_n / tot
            m2 += b_m2 + delta * delta * n * b_n / tot
            n = tot
    return {"mean": mean, "std": (m2 / n) ** 0.5, "n_vox": n,
            "slice_shape": list(shape)}


def main():
    cache = ROOT / "data" / "cache"
    stats = {}
    for frag in ("frag1", "frag3"):
        tifs = sorted((cache / frag / "surface_volume").glob("*.tif"))
        assert tifs, f"no slices for {frag}"
        print(f"{frag}: {len(tifs)} slices...", flush=True)
        stats[frag] = streaming_stats(tifs)
        stats[frag]["n_slices"] = len(tifs)
        print(f"  mean={stats[frag]['mean']:.1f} "
              f"std={stats[frag]['std']:.1f}", flush=True)
    (ROOT / "data" / "norm_stats.json").write_text(
        json.dumps(stats, indent=1))
    regions = [
        {"region_id": "frag1", "bbox": [[0, 65], [0, 8181], [0, 6330]]},
        {"region_id": "frag3", "bbox": [[0, 65], [0, 7606], [0, 5249]]},
    ]
    # fixed assignment (not random): frag1 train, frag3 val
    splits = [dict(regions[0], split="train"),
              dict(regions[1], split="val")]
    _ = make_splits  # schema-compatible; assignment fixed by design
    (ROOT / "src" / "dataio" / "splits.json").write_text(
        json.dumps(splits, indent=1))
    print("wrote data/norm_stats.json + src/dataio/splits.json")


if __name__ == "__main__":
    main()
