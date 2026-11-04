"""Translate Greek readings: demo CLI for the translation product.

  python scripts/translate_demo.py --text "χαῖρε κόσμε" --to de --mode gloss
  python scripts/translate_demo.py --file readings.json --to en --improve
  UNROLL_OFFLINE=1 python scripts/translate_demo.py --text "χαῖρε"

readings.json: [{"text": "...", "conf": 0.9}] or {"readings": [...]}.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents import harness as H  # noqa: E402
from src.agents import translate as T  # noqa: E402


def load_inputs(args):
    if args.file:
        obj = json.loads(Path(args.file).read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            obj = obj.get("readings", [])
        return [{"text": r.get("text", ""), "conf": r.get("conf", 0.0)} for r in obj]
    return [{"text": args.text, "conf": 1.0}]


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    import argparse
    ap = argparse.ArgumentParser(description="Unscroll translation demo")
    ap.add_argument("--text", default="χαῖρε κόσμε")
    ap.add_argument("--file", default="")
    ap.add_argument("--to", default="en", choices=sorted(T.SUPPORTED_TARGETS))
    ap.add_argument("--mode", default="fluent", choices=list(T.MODES))
    ap.add_argument("--improve", action="store_true",
                    help="run harness2-style Propose-Probe-Compose refinement")
    ap.add_argument("--budget", type=int, default=1)
    ap.add_argument("--json", action="store_true", help="emit raw JSON")
    a = ap.parse_args()

    readings = load_inputs(a)
    if a.improve:
        out = [H.improve(r["text"] or " ", target=a.to, mode=a.mode, budget=a.budget)
               for r in readings if r["text"].strip()]
    else:
        out = T.translate_readings(readings, target=a.to, mode=a.mode)
    if a.json:
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return
    for r, res in zip([x for x in readings if x["text"].strip()], out):
        if a.improve:
            t = res["result"]
            print(f"IN   [{r['conf']:.2f}] {t['input']}")
            print(f"OUT  [{t['target']}/{t['mode']} conf={t['conf']}] {t['translation']}")
            print(f"LOOP harness={res['harness']} score={res['score']} "
                  f"(base {res['base_score']})")
            for row in res["trace"]:
                print(f"  r{row.get('round', '?')} {row['harness']}: {row['score']}")
        else:
            print(f"IN   [{r['conf']:.2f}] {r['text']}")
            print(f"OUT  [{res['target']}/{res['mode']} tconf={res['tconf']}] {res['translation']}")


if __name__ == "__main__":
    main()
