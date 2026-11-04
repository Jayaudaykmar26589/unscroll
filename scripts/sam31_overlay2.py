"""Overlay each curve_sam31_c curve on its own frame (correct scale)."""
import json
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
out = ROOT / "curves_sam31_c"
cx, cy, crop = 4820, 3403, 2048
ox, oy = cx - crop // 2, cy - crop // 2
for z in (0, 4):
    img = Image.open(out / "frames3" / f"{z:05d}.jpg").convert("RGB")
    W, H = img.size
    sc = W / crop
    rec = json.loads((out / f"{z:05d}.jsonl").read_text())
    d = ImageDraw.Draw(img)
    for x, y in rec["points"]:
        d.point(((x - ox) * sc, (y - oy) * sc), fill=(255, 0, 0))
    img.save(out / f"overlay_f{z}.png")
    print("saved", z, len(rec["points"]))
