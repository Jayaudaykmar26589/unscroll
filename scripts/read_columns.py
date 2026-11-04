"""Muse Spark reads every line of each published Scroll 1 column, scored against the published transcription.

Only the line crops and anonymous region ids are sent (no reference text). Scoring
uses the same character error rate as the five-line excerpt (src/agents/recognition.py).

  python scripts/read_columns.py   -> reports/reading/columns/{<id>.json, summary.json}
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.agents.reading import load_passage  # noqa: E402
from src.agents.recognition import evaluate, recognize  # noqa: E402

OUT = ROOT / "reports" / "reading" / "columns"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"model_input": "line crops of the published column image only; no reference text", "columns": []}
    for path in sorted((ROOT / "data" / "reading" / "passages").glob("*.json")):
        p = load_passage(path)
        r = recognize(variant="baseline", passage=p, timeout=180.)
        e = evaluate(r, p)
        (OUT / f"{p['id']}.json").write_text(json.dumps({"prediction": r, "evaluation": e}, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
        row = {"id": p["id"], "lines": len(p["lines"]), "seconds": round(r["seconds"], 1),
               "normalized_cer": round(e["normalized_cer"], 4), "strict_cer": round(e["strict_cer"], 4),
               "exact_lines": e["normalized_exact_lines"],
               "reference_letters": sum(x["reference_letters"] for x in e["lines"])}
        summary["columns"].append(row)
        print(row, flush=True)
    c = summary["columns"]
    letters = sum(x["reference_letters"] for x in c)
    summary["overall_normalized_cer"] = round(sum(x["normalized_cer"] * x["reference_letters"] for x in c) / letters, 4)
    summary["lines"] = sum(x["lines"] for x in c)
    summary["exact_lines"] = sum(x["exact_lines"] for x in c)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "columns"}))


if __name__ == "__main__":
    main()
