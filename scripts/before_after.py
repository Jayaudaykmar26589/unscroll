"""Before/after animations of the first-slice fix on real Scroll 1 CT.

  python scripts/before_after.py            # default seeds -> reports/before_after/*.gif

Left: global-DP first slice (curves/). Right: local walk (curves_walk/). Green:
the released mesh the seed was clicked on (human-made reference); yellow: click.
Each frame states the median distance of both traces to that mesh. One GIF per
seed over the 20 traced slices, plus a PNG of slice 6110 for documents.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import tifffile  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from scripts.seeded_eval import _window_mesh  # noqa: E402
from src.eval.meshref import SurfaceIndex, curve_xyz, sample_curve, slice_segments  # noqa: E402

EVAL = ROOT / "reports" / "seeded_eval"
OUT = ROOT / "reports" / "before_after"
SLICES = ROOT / "data" / "cache" / "scroll1_6100_6200"
Z0, N = 6100, 20
DEFAULT = ["20231210121321", "20230701020044", "20230627170800", "20230702185753"]
HALF, SIZE = 320, 560


def _record(folder, sid, z):
    f = EVAL / folder / sid / f"{z:05d}.jsonl"
    return json.loads(f.read_text().splitlines()[0]) if f.exists() else None


def _median(index, rec):
    if rec is None or not rec["points"]:
        return None
    q, _ = sample_curve(curve_xyz(rec), 1.0)
    return float(np.median(index.distances(q))) if len(q) else None


def _panel(img, seed, mesh_segs, rec, title, dist):
    sx, sy = seed[:2]
    x0, y0 = sx - HALF, sy - HALF
    crop = img[y0:y0 + 2 * HALF, x0:x0 + 2 * HALF].astype(float)
    lo, hi = np.percentile(crop, (1, 99))
    im = Image.fromarray(np.clip((crop - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)).convert("RGB")
    s = SIZE / (2 * HALF)
    im = im.resize((SIZE, SIZE), Image.BILINEAR)
    d = ImageDraw.Draw(im)
    for a, b in mesh_segs:
        d.line([((a[0] - x0) * s, (a[1] - y0) * s), ((b[0] - x0) * s, (b[1] - y0) * s)],
               fill=(40, 230, 90), width=3)
    if rec is not None and len(rec["points"]) > 1:
        d.line([((x - x0) * s, (y - y0) * s) for x, y in rec["points"]], fill=(255, 40, 40), width=2)
    d.ellipse([HALF * s - 6, HALF * s - 6, HALF * s + 6, HALF * s + 6], outline=(255, 230, 0), width=3)
    d.rectangle([0, 0, SIZE, 30], fill=(15, 15, 15))
    label = f"{title}   median {dist:.1f} vx from human layer" if dist is not None else f"{title}   no trace"
    d.text((8, 8), label, fill=(255, 255, 255))
    return im


def render(sid, seed):
    v, f = _window_mesh(sid)
    index = SurfaceIndex(v, f)
    frames = []
    for z in range(N):
        img = tifffile.imread(SLICES / f"{Z0 + z:05d}.tif")
        segs = slice_segments(v, f, Z0 + z)
        dp, walk = _record("curves", sid, z), _record("curves_walk", sid, z)
        left = _panel(img, seed, segs, dp, "BEFORE: global DP", _median(index, dp))
        right = _panel(img, seed, segs, walk, "AFTER: local walk", _median(index, walk))
        frame = Image.new("RGB", (2 * SIZE + 10, SIZE + 34), (15, 15, 15))
        frame.paste(left, (0, 0))
        frame.paste(right, (SIZE + 10, 0))
        ImageDraw.Draw(frame).text(
            (8, SIZE + 10), f"Scroll 1 (PHercParis4) slice {Z0 + z}   seed mesh {sid}   "
            "green = human-traced layer, red = Unscroll trace, yellow = click", fill=(200, 200, 200))
        frames.append(frame)
    OUT.mkdir(parents=True, exist_ok=True)
    frames[0].save(OUT / f"{sid}.gif", save_all=True, append_images=frames[1:], duration=350, loop=0)
    frames[10].save(OUT / f"{sid}_z{Z0 + 10}.png")
    return OUT / f"{sid}.gif"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", default=DEFAULT)
    a = ap.parse_args(argv)
    seeds = {r["id"]: r["seed"] for r in json.loads((EVAL / "seeds.json").read_text()) if r.get("covers")}
    for sid in a.ids:
        print(render(sid, seeds[sid]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
