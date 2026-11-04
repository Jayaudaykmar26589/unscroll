"""Test all .obj->volume axis permutations for frame alignment."""
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import tifffile  # noqa: E402

from src.eval.meshref import load_obj  # noqa: E402


def main():
    img = tifffile.imread(
        ROOT / "data/cache/scroll1_6100_6200/06103.tif").astype(float)
    H, W = img.shape
    verts, _ = load_obj(
        ROOT / "data/cache/scroll1_segments/20230518181521.obj")
    for perm in itertools.permutations([0, 1, 2]):
        # perm maps obj axes -> (col=x, row=y, slice=z)
        c, r, s = (verts[:, perm[0]], verts[:, perm[1]], verts[:, perm[2]])
        sel = (np.abs(s - 6103) < 3) & (c >= 20) & (c < W - 20) & \
            (r >= 0) & (r < H)
        if sel.sum() < 20:
            print(f"{perm}: n={sel.sum()} SKIP")
            continue
        xs, ys = c[sel].astype(int), r[sel].astype(int)
        v = img[ys, xs]
        neigh = np.maximum(img[ys, xs - 15], img[ys, xs + 15])
        frac = (v > neigh).mean()
        print(f"{perm}: n={sel.sum()} localmax_frac={frac:.2f} "
              f"med={np.median(v):.0f}")


if __name__ == "__main__":
    main()
