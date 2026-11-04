"""Full-page ink view: the Grand Prize model's ink map over the whole released segment.

Input: reports/ink_gp_full/pred_full.png from `kaggle_run.py ... --ink-gp --full-page`
(segment 20230827161847, Scroll 1, every column on it). Rotated so text lines run
left to right, cropped to the segment, written for the Read page:

  python scripts/build_fullpage.py -> viewer/assets/fullpage-ink.webp (page view)
                                      viewer/assets/fullpage-ink-hi.webp (zoom view)
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "reports" / "ink_gp_full"
ROTATE = 270           # degrees counter-clockwise: text lines run left to right, right way up
MIRROR = True          # the segment is seen from the back of the sheet; flip so letters are not reversed (checked by eye)
Image.MAX_IMAGE_PIXELS = None


def main():
    im = Image.open(SRC / "pred_full.png").convert("L")
    a = np.asarray(im)
    ys, xs = np.nonzero(a > 8)
    im = im.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    im = im.rotate(ROTATE, expand=True)
    if MIRROR:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    out = ROOT / "viewer" / "assets"
    for name, width, q in (("fullpage-ink.webp", 1800, 70), ("fullpage-ink-hi.webp", 4000, 60)):
        w = min(width, im.width)
        im.resize((w, round(im.height * w / im.width)), Image.LANCZOS).save(out / name, "WEBP", quality=q, method=6)
        print(name, (out / name).stat().st_size // 1024, "KB")
    s = json.loads((SRC / "summary.json").read_text())
    s["view"] = {"rotate_ccw": ROTATE, "mirrored": MIRROR, "size": [im.width, im.height]}
    (SRC / "view.json").write_text(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
