"""Render overlay frames from curves for demo video / viewer.

  python scripts/render_frames.py --curves curves_gono \\
      --slices data/cache/scroll1_6100_6200 --out demo/frames --max 50

Writes {i:05d}.png: 1/4-scale slice + red curve + z label. Also writes
demo/flat.png from sheet dir if --sheet is given.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import tifffile  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from src.trace.slices import resolve_slice  # noqa: E402


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", required=True)
    ap.add_argument("--slices", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sheet", default="")
    ap.add_argument("--max", type=int, default=50)
    ap.add_argument("--scale", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tifs = sorted(Path(a.slices).glob("*.tif*"))
    files = sorted(Path(a.curves).glob("*.jsonl"))[:a.max]
    for f in files:
        i = int(f.stem)
        rec = json.loads(f.read_text())
        img = tifffile.imread(resolve_slice(rec, tifs)).astype(np.float32)
        small = img[::a.scale, ::a.scale]
        lo, hi = np.percentile(small, (1, 99))
        u8 = np.clip((small - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        rgb = np.stack([u8] * 3, -1)
        pts = np.array(rec["points"])
        if len(pts):
            px = (pts[:, 0] / a.scale).astype(int)
            py = (pts[:, 1] / a.scale).astype(int)
            ok = (px >= 0) & (px < rgb.shape[1]) & (py >= 0) & (py < rgb.shape[0])
            rgb[py[ok], px[ok]] = (255, 0, 0)
        im = Image.fromarray(rgb)
        ImageDraw.Draw(im).text((10, 10), f"z={rec.get('source_z', rec['z'])} n={len(pts)}",
                                fill=(0, 255, 0))
        im.save(out / f"{i:05d}.png")
        print(f"frame {i}", flush=True)
    if a.sheet:
        fp = Path(a.sheet) / "flat_preview.png"
        if fp.exists():
            Image.open(fp).save(out / "flat.png")
            print("flat.png", flush=True)
    print(f"{len(files)} frames in {out}", flush=True)


if __name__ == "__main__":
    main()
