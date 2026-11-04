import hashlib
import json
from pathlib import Path

from src.agents.reading import ROOT, load_passage

PASSAGES = sorted((ROOT / "data" / "reading" / "passages").glob("*.json"))


def test_every_published_column_is_a_passage():
    assert [p.stem for p in PASSAGES] == [f"pherc-paris4-col-minus{c}" for c in (1, 2, 7, 8)]


def test_passages_are_image_linked_and_pinned():
    for path in PASSAGES:
        p = load_passage(path)
        img = ROOT / "viewer" / p["source"]["image_path"]
        assert hashlib.sha256(img.read_bytes()).hexdigest() == p["source"]["image_sha256"]
        tops = [ln["bbox"][1] for ln in p["lines"]]
        assert tops == sorted(tops) and len(set(tops)) == len(tops)       # one box per line, top to bottom
        assert "CC BY-NC 4.0" in p["source"]["attribution"]


def test_starting_translations_are_marked_model_drafts():
    for path in PASSAGES:
        p = load_passage(path)
        for k in ("en", "ta"):
            t = p["translations"][k]
            assert t["kind"] == "model_draft" and t["review_status"].startswith("Unreviewed")
            assert t["text"].strip()
        assert any("஀" <= c <= "௿" for c in p["translations"]["ta"]["text"])


def test_original_excerpt_keeps_editorial_drafts():
    p = load_passage()
    assert all(p["translations"][k]["kind"] == "editorial_draft" for k in ("en", "ta"))


def test_reading_bundle_lists_all_passages_original_first():
    js = (ROOT / "viewer" / "reading-data.js").read_text(encoding="utf-8")
    bundles = json.loads(js.split("window.UNSCROLL_READINGS=", 1)[1].split(";\n", 1)[0])
    assert bundles[0]["id"] == load_passage()["id"] and bundles[0]["recognition_supported"]
    assert len(bundles) == 1 + len(PASSAGES)
    assert not any(b["recognition_supported"] for b in bundles[1:])
    assert all("translation_input" not in b for b in bundles)


def test_full_page_reading_is_unscored_and_matches_its_image():
    js = (ROOT / "viewer" / "reading-data.js").read_text(encoding="utf-8")
    fp = json.loads(js.split("window.UNSCROLL_FULLPAGE=", 1)[1].rsplit(";\n", 1)[0])
    assert fp["scored"] is False and fp["reference_sent"] is False
    assert fp["ink"]["segment"] == fp["segment"] == "20230827161847"
    from PIL import Image
    w, h = Image.open(ROOT / "viewer" / fp["image"]).size
    assert [w, h] == fp["size"]
    assert all(0 <= l["bbox"][0] < l["bbox"][2] <= w and 0 <= l["bbox"][1] < l["bbox"][3] <= h for l in fp["lines"])
