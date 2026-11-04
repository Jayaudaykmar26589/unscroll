"""Muse Spark reads the readable lines of Unscroll's full-page ink map (unscored).

The ink map is the Grand Prize model run by Unscroll on the whole released segment
20230827161847 (scripts/ink_gp_kaggle.py --full, scripts/build_fullpage.py). No
published transcription is aligned to this sheet, so the reading cannot be scored;
the measured error on published columns (scripts/read_columns.py) is the guide.

  python scripts/read_fullpage.py -> viewer/assets/fullpage-level.webp, reports/ink_gp_full/reading.json
"""
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.agents.recognition import recognize  # noqa: E402

SRC = ROOT / "reports" / "ink_gp_full"
DESKEW = 6.0           # degrees: makes text rows sharpest (row-darkness variance), checked by eye
# readable lines on the levelled 4000 px view, placed by hand (reports/ink_gp_full/read_boxes.png)
BOXES = [[0, 540, 3900, 790], [0, 840, 3900, 1150], [0, 1200, 3900, 1570], [0, 1590, 3900, 1920]]


def main():
    lev = SRC / "read_level.png"
    if not lev.exists():
        import numpy as np
        a = np.asarray(Image.open(ROOT / "viewer/assets/fullpage-ink-hi.webp").convert("L")).copy()
        inv = 255 - a
        inv[a < 6] = 255                                   # outside the segment: blank paper
        Image.fromarray(inv).rotate(DESKEW, resample=Image.BICUBIC, fillcolor=255).save(lev)
    out = ROOT / "viewer" / "assets" / "fullpage-level.webp"
    Image.open(lev).save(out, "WEBP", quality=60, method=6)
    w, h = Image.open(out).size
    passage = {"source": {"image_path": "assets/fullpage-level.webp",
                          "image_sha256": hashlib.sha256(out.read_bytes()).hexdigest()},
               "lines": [{"id": f"r{i + 1}", "bbox": b} for i, b in enumerate(BOXES)]}
    r = recognize(variant="baseline", passage=passage, timeout=120.)
    rep = {"segment": "20230827161847", "image": "assets/fullpage-level.webp", "size": [w, h], "deskew_deg": DESKEW,
           "model": r["model"], "seconds": round(r["seconds"], 1), "reference_sent": False, "scored": False,
           "lines": [{"id": x["id"], "bbox": x["bbox"], "text": x["text"]} for x in r["lines"]],
           "note": ("No published transcription is aligned to this sheet, so this reading is unscored. On the four "
                    "published Scroll 1 columns the same model had 79% character error (reports/reading/columns).")}
    (SRC / "reading.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(rep["model"], rep["seconds"], "s;", [len(x["text"]) for x in rep["lines"]], "characters per line")


if __name__ == "__main__":
    main()
