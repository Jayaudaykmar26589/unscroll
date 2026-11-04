"""Overlay frame-0 curve + click on crop for curves_sam31_p1."""
import json
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
out = ROOT / "curves_sam31_p1"
img = Image.open(out / "frames3" / "00000.jpg").convert("RGB")
rec = json.loads((out / "00000.jsonl").read_text())
# curve stored in full-slice coords; crop was centered on click (4820,3403)
cx, cy, crop = 4820, 3403, 2048
ox, oy = cx - crop // 2, cy - crop // 2
W, H = img.size
sc = W / crop
d = ImageDraw.Draw(img)
for x, y in rec["points"]:
    d.point(((x - ox) * sc, (y - oy) * sc), fill=(255, 0, 0))
d.ellipse([W / 2 - 8, H / 2 - 8, W / 2 + 8, H / 2 + 8], outline=(0, 255, 0), width=3)
img.save(out / "frame0_overlay.png")
print("saved", out / "frame0_overlay.png", "npts=", len(rec["points"]))
