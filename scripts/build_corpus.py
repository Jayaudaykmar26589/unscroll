"""Build the sentence corpus for the Predict page.

Source: Diogenes Laertius, Lives of Eminent Philosophers (Greek, Perseus
canonical-greekLit, CC BY-SA 4.0). Book 10 holds Epicurus' three letters and the
Principal Doctrines, the school Philodemus wrote for. The published PHerc. Paris. 4
passage on the Read page is added too.

  python scripts/build_corpus.py   (fetch the XML once into data/predict/raw/)
"""
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.agents.complete import key_map  # noqa: E402

SRC = ROOT / "data/predict/raw/tlg0004.tlg001.perseus-grc2.xml"
URL = "https://raw.githubusercontent.com/PerseusDL/canonical-greekLit/master/data/tlg0004/tlg001/tlg0004.tlg001.perseus-grc2.xml"
OUT = ROOT / "data/predict/corpus.json"
NS = "{http://www.tei-c.org/ns/1.0}"
SKIP = {NS + "note", NS + "head", NS + "bibl", NS + "label"}
END = re.compile(r"(?<=[.;;])\s+")


def _text(el, out):
    if el.tag in SKIP:
        return
    if el.text:
        out.append(el.text)
    for c in el:
        _text(c, out)
        if c.tail:
            out.append(c.tail)


def sentences():
    root = ET.parse(SRC).getroot()
    for book in root.iter(NS + "div"):
        if book.get("subtype") != "book":
            continue
        # Join the book in order, remembering where each section starts.
        parts, marks = [], []
        for sec in book.iter(NS + "div"):
            if sec.get("subtype") != "section":
                continue
            buf = []
            _text(sec, buf)
            marks.append((sum(len(p) for p in parts), f"{book.get('n')}.{sec.get('n')}"))
            parts.append(" ".join(buf) + " ")
        text = "".join(parts)
        pos = 0
        for s in END.split(text):
            start = text.find(s, pos)
            pos = start + len(s)
            s = re.sub(r"\s+", " ", s.replace("“", "").replace("”", "")).strip()
            if len(key_map(s)[0].replace(" ", "")) < 12:
                continue
            ref = [r for o, r in marks if o <= start][-1] if marks else book.get("n")
            yield {"ref": "D.L. " + ref, "text": s}


def main():
    if not SRC.exists():
        raise SystemExit(f"Fetch the source first:\n  curl -sfL -o {SRC} {URL}")
    out = [dict(id=f"dl{i}", **s) for i, s in enumerate(sentences())]
    p = json.loads((ROOT / "data/reading/passage.json").read_text(encoding="utf-8"))
    out.append({"id": "pherc-col-8", "ref": "PHerc. Paris. 4, col. −8, 10–14 (published preliminary reading)",
                "text": " ".join(l["text"] for l in p["lines"])})
    OUT.write_text(json.dumps({"source": {"title": "Diogenes Laertius, Lives of Eminent Philosophers (Greek)",
                                           "url": URL, "license": "CC BY-SA 4.0 (Perseus Digital Library)"},
                               "sentences": out}, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"{len(out)} sentences -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
