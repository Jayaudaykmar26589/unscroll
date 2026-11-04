"""Fragment surface-depth maps: per-pixel z of papyrus sheet.

z* = intensity-weighted mean z over the column (weights = max(0, v-mean)).
Computed at stride 4, cached to data/cache/{frag}/zsurf.npy (float32, H/4,W/4).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import tifffile  # noqa: E402


def zsurface(frag_dir, stride=4):
    frag_dir = Path(frag_dir)
    out = frag_dir / "zsurf.npy"
    if out.exists():
        return np.load(str(out))
    stats = json.loads((ROOT / "data" / "norm_stats.json").read_text())
    mean = stats[frag_dir.name]["mean"]
    tifs = sorted((frag_dir / "surface_volume").glob("*.tif*"))
    acc_w = acc_wz = None
    for z, t in enumerate(tifs):
        img = tifffile.imread(t).astype(np.float32)[::stride, ::stride]
        w = np.maximum(0.0, img - mean)
        if acc_w is None:
            acc_w = w
            acc_wz = w * z
        else:
            acc_w += w
            acc_wz += w * z
    zs = np.where(acc_w > 0, acc_wz / np.maximum(acc_w, 1e-6),
                  len(tifs) // 2).astype(np.float32)
    np.save(str(out), zs)
    print(f"zsurface {frag_dir.name}: {zs.shape} zrange "
          f"[{zs.min():.1f},{zs.max():.1f}]", flush=True)
    return zs


if __name__ == "__main__":
    for frag in ("frag1", "frag3"):
        zsurface(ROOT / "data" / "cache" / frag)
