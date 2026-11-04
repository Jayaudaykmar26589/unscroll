"""Crop a slice stack around one seed for cheap upload (e.g. to a Kaggle GPU).

  python scripts/make_crop.py --seed-id 20231210121321 --n 100 --size 2560

Writes data/cache/crop_<seed-id>/NNNNN.tif (same physical z names, uint16) and
crop.json with the XY offset. Curves traced on the crop are in crop pixels;
add the offset (`shift_curves`) to return to volume coordinates, which is what
OBJ meshes and the flattener use.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import tifffile  # noqa: E402

SLICES = ROOT / "data" / "cache" / "scroll1_6100_6200"
SEEDS = ROOT / "reports" / "seeded_eval" / "seeds.json"


def shift_curves(src, dst, offset):
    """Copy curve JSONL files, adding the crop offset to every point."""
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    ox, oy = offset
    for f in sorted(src.glob("[0-9]*.jsonl")):
        rec = json.loads(f.read_text().splitlines()[0])
        rec["points"] = [[round(x + ox, 1), round(y + oy, 1)] for x, y in rec["points"]]
        if rec.get("origin"):
            rec["origin"] = [round(rec["origin"][0] + ox, 1), round(rec["origin"][1] + oy, 1)]
        (dst / f.name).write_text(json.dumps(rec) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-id", required=True)
    ap.add_argument("--z0", type=int, default=6100)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--size", type=int, default=2560)
    a = ap.parse_args(argv)
    seed = next(r["seed"] for r in json.loads(SEEDS.read_text()) if r["id"] == a.seed_id)
    out = ROOT / "data" / "cache" / f"crop_{a.seed_id}"
    out.mkdir(parents=True, exist_ok=True)
    first = tifffile.memmap(SLICES / f"{a.z0:05d}.tif")
    h, w = first.shape
    del first
    ox = int(min(max(seed[0] - a.size // 2, 0), w - a.size))
    oy = int(min(max(seed[1] - a.size // 2, 0), h - a.size))
    for z in range(a.z0, a.z0 + a.n):
        img = tifffile.imread(SLICES / f"{z:05d}.tif")   # full read also validates the file
        tifffile.imwrite(out / f"{z:05d}.tif", np.ascontiguousarray(img[oy:oy + a.size, ox:ox + a.size]))
    meta = {"seed_id": a.seed_id, "seed": seed, "offset_xy": [ox, oy], "size": a.size,
            "z0": a.z0, "n": a.n, "source": str(SLICES.relative_to(ROOT)),
            "click_in_crop": [seed[0] - ox, seed[1] - oy, seed[2]],
            "volume": "20230205180739"}
    (out / "crop.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
