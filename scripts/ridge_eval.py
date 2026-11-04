"""Ridge-coincidence eval + overlay renderer (no aligned GT available).

Metric: fraction of traced points within 2 voxels of a ridge pixel
(ridge = row-local-maximum above median+1.5std). Overlays to
reports/overlays/ for author spot-verification.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import tifffile  # noqa: E402
from PIL import Image  # noqa: E402
from src.trace.slices import resolve_slice  # noqa: E402

def dilate2(m):
    """2px binary dilation via rolls (two 1px passes)."""
    for _ in range(2):
        m = (m | np.roll(m, 1, 0) | np.roll(m, -1, 0)
             | np.roll(m, 1, 1) | np.roll(m, -1, 1)
             | np.roll(np.roll(m, 1, 0), 1, 1)
             | np.roll(np.roll(m, 1, 0), -1, 1)
             | np.roll(np.roll(m, -1, 0), 1, 1)
             | np.roll(np.roll(m, -1, 0), -1, 1))
    return m


def ridge_mask(img, k=1.5):
    thr = np.median(img) + k * img.std()
    left = np.roll(img, 1, 1)
    right = np.roll(img, -1, 1)
    m = (img > thr) & (img >= left) & (img >= right)
    m[:, [0, -1]] = False
    return m


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", required=True)
    ap.add_argument("--slices", required=True)
    ap.add_argument("--z0", type=int, required=True)
    a = ap.parse_args()
    out = ROOT / "reports" / "overlays"
    out.mkdir(parents=True, exist_ok=True)
    tifs = sorted(Path(a.slices).glob("*.tif*"))
    fracs = []
    for f in sorted(Path(a.curves).glob("*.jsonl")):
        rec = json.loads(f.read_text())
        i = int(f.stem)
        pts = np.array(rec["points"])
        img = tifffile.imread(resolve_slice(rec, tifs)).astype(float)
        rm = dilate2(ridge_mask(img))
        if len(pts) == 0:
            print(f"{i}: EMPTY")
            continue
        qx = np.clip(pts[:, 0].astype(int), 0, rm.shape[1] - 1)
        qy = np.clip(pts[:, 1].astype(int), 0, rm.shape[0] - 1)
        frac = float(rm[qy, qx].mean())
        fracs.append(frac)
        # overlay at 1/8 scale
        small = img[::8, ::8]
        lo, hi = np.percentile(small, (1, 99))
        u8 = np.clip((small - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        rgb = np.stack([u8] * 3, -1)
        px, py = (pts[:, 0] / 8).astype(int), (pts[:, 1] / 8).astype(int)
        ok = (px >= 0) & (px < rgb.shape[1]) & (py >= 0) & (py < rgb.shape[0])
        rgb[py[ok], px[ok]] = (255, 0, 0)
        Image.fromarray(rgb).save(out / f"{i:05d}.png")
        print(f"{i}: n={len(pts)} ridge_frac2={frac:.3f}", flush=True)
    print(f"MEAN ridge_frac2 = {np.mean(fracs):.3f} over {len(fracs)} slices")


if __name__ == "__main__":
    main()
