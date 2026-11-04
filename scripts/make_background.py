"""Site background: real Scroll 1 CT slice 6100 with the on-layer development SAM 3.1 traces in teal.

  python scripts/make_background.py   -> viewer/assets/scroll1-z6100.webp
"""
import json
from pathlib import Path
import numpy as np, tifffile
from PIL import Image, ImageDraw, ImageFilter
ROOT = Path(r"D:\Meta\unscroll")
a = tifffile.imread(ROOT / "data/cache/scroll1_6100_6200/06100.tif").astype(np.float32)
F = 6                                              # block-mean downscale
h, w = (a.shape[0] // F) * F, (a.shape[1] // F) * F
s = a[:h, :w].reshape(h // F, F, w // F, F).mean((1, 3))
lo, hi = np.percentile(s, [35, 99.7])
g = np.clip((s - lo) / (hi - lo), 0, 1) ** 1.25
x0, x1, y0, y1 = 70, 1300, 100, 1030               # the scroll, without most of the foam mount
g = g[y0:y1, x0:x1]
print("crop px", g.shape, "slice vx origin", x0 * F, y0 * F)
U = 2                                              # draw traces supersampled
img = Image.fromarray((g * 255 * 0.3).astype(np.uint8)).convert("RGB").resize((g.shape[1] * U, g.shape[0] * U), Image.LANCZOS)
lay = Image.new("RGBA", img.size, (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
rows = json.loads((ROOT / "reports/triage/triage.json").read_text())["rows"]
ids = [r["id"] for r in rows if r["mode"] == "walk" and r["set"] == "dev" and r["on_layer"]]
for i in ids:
    r = json.loads((ROOT / f"reports/seeded_eval/curves_walk/{i}/00000.jsonl").read_text().splitlines()[0])
    assert r["source_z"] == 6100
    p = np.asarray(r["points"], float)
    q = [((x / F - x0) * U, (y / F - y0) * U) for x, y in p]
    d.line(q, fill=(125, 224, 194, 255), width=4 * U, joint="curve")
glow = lay.filter(ImageFilter.GaussianBlur(5 * U))
img = Image.alpha_composite(Image.alpha_composite(img.convert("RGBA"), glow), lay).convert("RGB")
W = 1100; img = img.resize((W, round(W * g.shape[0] / g.shape[1])), Image.LANCZOS)
out = ROOT / "viewer/assets/scroll1-z6100.webp"
img.save(out, "WEBP", quality=50, method=6)
print(out, img.size, out.stat().st_size, "traces", len(ids))
