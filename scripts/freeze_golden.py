"""Freeze golden demo artifacts (API-free rollback target).

  python scripts/freeze_golden.py --frames demo/frames_pilot --out golden

Writes golden/run.json (frames manifest + sheet readings + translations),
copies flat.png + overlay frames, and emits SHA256SUMS. Byte-identical
replay is checked by scripts/demo.py --offline.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import os  # noqa: E402
os.environ["UNROLL_OFFLINE"] = "1"
from src.agents import translate as T  # noqa: E402


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="demo/frames_pilot")
    ap.add_argument("--out", default="golden")
    a = ap.parse_args()
    src = ROOT / a.frames
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    frames = sorted(p.name for p in src.glob("[0-9]*.png"))
    for n in frames:
        shutil.copy2(src / n, out / n)
    if (src / "flat.png").exists():
        shutil.copy2(src / "flat.png", out / "flat.png")

    readings = [{"text": "χαῖρε κόσμε", "conf": 0.92},
                {"text": "καὶ ὁ μῦθος", "conf": 0.61}]
    translations = {}
    for tgt in ("en", "de", "fr"):
        translations[tgt] = {
            "fluent": T.translate_readings(readings, target=tgt, mode="fluent"),
            "gloss": T.translate_readings(readings, target=tgt, mode="gloss"),
        }
    run = {"run_id": "golden-translation-v1", "frames": frames,
           "sheet": "flat.png", "readings": readings, "translations": translations}
    (out / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
    sums = [f"{sha256(out / n)}  {n}" for n in sorted(p.name for p in out.iterdir()
                                                      if p.name != "SHA256SUMS")]
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(f"{out}: {len(frames)} frames + run.json + SHA256SUMS", flush=True)


if __name__ == "__main__":
    main()
