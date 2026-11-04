"""Build Read & Translate passages for the published PHerc. Paris. 4 (Scroll 1) columns.

Source: Vesuvius Challenge 2023 Grand Prize page (cached as data/reading/source.html,
parsed to data/reading/published_columns.json), column images in viewer/assets,
content CC BY-NC 4.0. Line rectangles come from scripts/line_boxes.py and were
checked by eye (reports/reading/columns/*_boxes.png). Lines that are not visible
in the published image are left out and named in the editorial note.

Starting translations are Muse Spark drafts, marked unreviewed; they are made once
and kept, so the page works offline.

  python scripts/build_passages.py            -> data/reading/passages/*.json
  python scripts/build_passages.py --no-model (keep existing drafts, no API calls)
"""
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.line_boxes import line_bands  # noqa: E402

OUT = ROOT / "data" / "reading" / "passages"
COLUMNS = ROOT / "data" / "reading" / "published_columns.json"
PAGE = "https://scrollprize.org/grandprize#what-does-the-scroll-say"
ATTRIBUTION = ("Column image © Vesuvius Challenge (CC BY-NC 4.0); underlying scan: EduceLab-Scrolls / "
               "University of Kentucky; papyrus: Institut de France. Published preliminary transcription: "
               "Vesuvius papyrology team.")
# lines visible in each published image (col. −2's last entry is one carried-over letter of l. 9)
VISIBLE = {"-8": 13, "-7": 7, "-2": 7, "-1": 6}


def passage(col, spec):
    n = VISIBLE[col]
    name = f"published-col-{abs(int(col))}.webp"
    img = ROOT / "viewer" / "assets" / name
    w, h = Image.open(img).size
    boxes = line_bands(img, n)
    lines = [{"id": f"l{ln['label']}", "label": ln["label"], "text": ln["text"], "bbox": b}
             for ln, b in zip(spec["lines"][:n], boxes)]
    left_out = [ln["label"] for ln in spec["lines"][n:]]
    a, b = spec["published_range"]
    cid = f"pherc-paris4-col-minus{abs(int(col))}"
    note = ("Underdots mark uncertain letters in the published reading; brackets mark lost text. "
            "Line rectangles were fitted to the published image and checked by eye; they are not CT coordinates. ")
    if left_out:
        note += f"Line {', '.join(left_out)} begins after the image ends and is not shown. "
    note += "Starting translations are unreviewed Muse Spark drafts, not a scholarly translation."
    return {
        "schema_version": 1, "id": cid,
        "title": f"PHerc. Paris. 4 · column −{abs(int(col))}, lines {a}–{b}",
        "scroll": "Scroll 1 · PHerc. Paris. 4", "source_language": "grc",
        "source": {
            "title": "Vesuvius Challenge 2023 Grand Prize announcement", "url": PAGE,
            "published_date": "2024-02-05",
            "location": f"PHerc. Paris. 4, column −{abs(int(col))}, lines {a}–{b}",
            "transcription_status": "Published initial, rough draft transcription; not Unscroll OCR",
            "image_url": f"https://scrollprize.org/img/grandprize/col-{abs(int(col))}.webp",
            "image_path": f"assets/{name}", "image_width": w, "image_height": h,
            "image_sha256": hashlib.sha256(img.read_bytes()).hexdigest(),
            "attribution": ATTRIBUTION, "license_url": "https://scrollprize.org/data",
            "relationship_to_experiment": ("Published reading of a whole surviving column. Not text recovered "
                                           "by Unscroll's own tracing."),
            "region_method": "Evenly spaced line rectangles fitted to the image darkness profile, checked by eye.",
        },
        "lines": lines, "editorial_note": note,
    }


def main():
    use_model = "--no-model" not in sys.argv
    from src.agents.reading import translate_passage
    OUT.mkdir(parents=True, exist_ok=True)
    cols = json.loads(COLUMNS.read_text(encoding="utf-8"))
    for col, spec in cols.items():
        p = passage(col, spec)
        path = OUT / f"{p['id']}.json"
        old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        text = "\n".join(ln["text"] for ln in p["lines"])
        prior = old.get("translations")
        if prior and old.get("translation_input") == text:
            p["translations"] = prior
        elif use_model:
            r = translate_passage(text)
            p["translations"] = {k: {"text": r["translations"][k], "kind": "model_draft",
                                     "provenance": f"{r['model']} draft; unreviewed",
                                     "review_status": "Unreviewed model draft; not a scholarly reading. " + r["notes"]}
                                 for k in ("en", "ta")}
            print(f"{p['id']}: Spark translation drafted", flush=True)
        else:
            raise SystemExit(f"{p['id']}: no stored translation for this text; run without --no-model")
        p["translation_input"] = text
        path.write_text(json.dumps(p, ensure_ascii=False, indent=1), encoding="utf-8")
        print(path.name, len(p["lines"]), "lines")


if __name__ == "__main__":
    main()
