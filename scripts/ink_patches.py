"""Visualize ink training patches: depth-mean projection + label overlay."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from src.ink.dataset import InkPatches  # noqa: E402

d = InkPatches(ROOT / "data/cache/frag1", n=8, seed=11)
out = ROOT / "reports" / "ink"
for i in range(4):
    x, y = d[i]
    proj = x.numpy().mean(axis=0)
    lo, hi = np.percentile(proj, (1, 99))
    u8 = np.clip((proj - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    rgb = np.stack([u8] * 3, -1)
    rgb[y.numpy() > 0] = (255, 0, 0)
    Image.fromarray(rgb).save(out / f"patch_{i}.png")
    print(f"patch {i}: inkfrac={y.numpy().mean():.3f}", flush=True)
