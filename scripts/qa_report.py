"""Muse Spark QA reports: one plain-language paragraph per traced seed, numbers checked.

  python scripts/qa_report.py            # walk-mode seeds -> reports/qa/<seed>.json

Spark gets the seed's measured metrics (from score.json / score_walk.json and
per-slice medians) and its review image, and writes a short report for a
scholar deciding whether to trust the trace. Guard: every number in the reply
must appear in the facts Spark was given (after rounding); otherwise the reply
is regenerated once and then marked `numbers_ok: false`. Nothing is shown in
the product unless `numbers_ok` is true.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.spark import SPARK_MODEL, _img_url, call_spark  # noqa: E402

EVAL = ROOT / "reports" / "seeded_eval"
OUT = ROOT / "reports" / "qa"
IMG = ROOT / "reports" / "spark_review" / "images" / "walk"

SYSTEM = (
    "You write quality notes for papyrologists about automated traces of one papyrus sheet "
    "in X-ray CT of a Herculaneum scroll. Use only the facts given; never introduce a number "
    "that is not in the facts. Plain English, 3-4 sentences, no headings. Say whether the trace "
    "can be trusted, where it is weakest, and what a person should check.")


def facts_for(sid, before, after):
    f = {
        "seed_mesh": sid,
        "slices_traced": after["slices_with_trace"], "slices_total": after["slices"],
        "within_5_voxels_percent": round(100 * after["within5"], 1),
        "within_2_voxels_percent": round(100 * after["within2"], 1),
        "median_distance_voxels": round(after["median_dist"], 1),
        "before_fix_within_5_voxels_percent": round(100 * before["within5"], 1),
        "before_fix_median_distance_voxels": round(before["median_dist"], 1),
        "on_clicked_layer": after["median_dist"] <= 6.0,
        "voxel_size_micrometres": 7.91,
        "image": "red line = trace, yellow dot = click; panels: close-up, slice 0, slice 19",
    }
    return f


def _numbers(text):
    return {float(x) for x in re.findall(r"(?<![\w.])\d+(?:\.\d+)?", text.replace(",", ""))}


def allowed_numbers(facts):
    vals = set()
    for v in facts.values():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            vals |= {float(v), float(round(v)), round(float(v), 1)}
        elif isinstance(v, str) and v.isdigit():          # e.g. the seed mesh id
            vals.add(float(v))
    vals |= {0.0, 1.0, 2.0, 5.0, 19.0, 20.0}  # panel / tolerance labels present in the facts
    return vals


def write(sid, facts):
    img = IMG / f"{sid}.png"
    content = [{"type": "text", "text": "Facts (JSON):\n" + json.dumps(facts, indent=1)}]
    if img.exists():
        content.append({"type": "image_url", "image_url": {"url": _img_url(img)}})
    ok_set = allowed_numbers(facts)
    for attempt in (1, 2):
        text = call_spark([{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}])
        bad = sorted(n for n in _numbers(text) if n not in ok_set)
        if not bad:
            return {"text": text.strip(), "numbers_ok": True, "attempts": attempt}
    return {"text": text.strip(), "numbers_ok": False, "unsupported_numbers": bad, "attempts": 2}


def main():
    before = {r["id"]: r for r in json.loads((EVAL / "score.json").read_text())["seeds"]}
    after = {r["id"]: r for r in json.loads((EVAL / "score_walk.json").read_text())["seeds"]}
    OUT.mkdir(parents=True, exist_ok=True)
    for sid in after:
        if "within5" not in after[sid] or "within5" not in before[sid]:
            continue
        done = OUT / f"{sid}.json"
        if done.exists() and json.loads(done.read_text(encoding="utf-8")).get("numbers_ok"):
            continue
        facts = facts_for(sid, before[sid], after[sid])
        rep = {"seed": sid, "model": SPARK_MODEL, "facts": facts, **write(sid, facts)}
        (OUT / f"{sid}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
        print(sid, "numbers_ok" if rep["numbers_ok"] else f"REJECTED {rep['unsupported_numbers']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
