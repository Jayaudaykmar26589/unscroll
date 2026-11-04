"""Measure how accurately the Predict page completes sentences.

A. Lookup (offline): masked fragments of corpus sentences, from the beginning,
   middle and end, typed cleanly or noisily (no accents, one typo, edge words cut).
   Correct = the top result is the source sentence.
B. Muse Spark conjecture (lookup switched off): the first half of a sentence;
   correct = the completed sentence equals the original (accents ignored).

  python scripts/eval_complete.py [--lookup-only] [--n-spark 40]
"""
import argparse
import json
import random
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
from src.agents import complete as C  # noqa: E402

OUT = ROOT / "reports/predict/eval.json"


def strip_accents(t):
    return unicodedata.normalize("NFC", "".join(c for c in unicodedata.normalize("NFD", t) if not unicodedata.combining(c)))


def noisy(ws, rng):
    ws = [strip_accents(w) for w in ws]
    i = rng.randrange(1, len(ws) - 1) if len(ws) > 2 else 0
    w = ws[i]
    if len(w) > 3:                                   # one typo: drop a letter
        k = rng.randrange(1, len(w) - 1)
        ws[i] = w[:k] + w[k + 1:]
    if len(ws[0]) > 3:
        ws[0] = ws[0][rng.randrange(1, len(ws[0]) - 2):]   # cut the first word
    if len(ws[-1]) > 3:
        ws[-1] = ws[-1][:rng.randrange(2, len(ws[-1]) - 1)]  # and the last
    return ws


def fragment(text, where, frac, rng):
    ws = [text[a:b] for _, a, b in C.words(text)]
    k = max(C.MIN_WORDS, round(len(ws) * frac))
    start = {"start": 0, "end": len(ws) - k, "middle": (len(ws) - k) // 2}[where]
    return ws[start:start + k]


def same(a, b):
    return C.key_map(a)[0] == C.key_map(b)[0]


def eval_lookup(rng, n=150):
    cor = C.corpus()
    pool = [s for s in cor.sents if 8 <= len(C.words(s["text"])) <= 60]
    sample = rng.sample(pool, n)
    rows = []
    t0 = time.time()
    for s in sample:
        for where in ("start", "middle", "end"):
            for mode in ("clean", "noisy"):
                ws = fragment(s["text"], where, 0.4, rng)
                if mode == "noisy":
                    ws = noisy(ws, rng)
                q = " ".join(ws)
                r = C.complete(q, use_model=False)
                top = r["matches"][0] if r["matches"] else None
                rows.append({"id": s["id"], "where": where, "mode": mode, "query": q, "status": r["status"],
                             "correct": bool(top and same(top["text"], s["text"])), "ambiguous": r.get("ambiguous", False),
                             "got": top["ref"] if top else None})
    dt = (time.time() - t0) / len(rows)
    by = {}
    for r in rows:
        for k in (f"{r['mode']}", f"{r['where']}_{r['mode']}", "all"):
            b = by.setdefault(k, [0, 0])
            b[0] += r["correct"]
            b[1] += 1
    return {"fragments": len(rows), "sentences": n, "fragment_words": "40% of the sentence, at least 3",
            "accuracy": {k: round(c / t, 4) for k, (c, t) in by.items()},
            "ambiguous_rate": round(sum(r["ambiguous"] for r in rows) / len(rows), 4),
            # Precision of the answers the page presents as certain (attested, single source).
            "confident_answers": sum(1 for r in rows if r["status"] == "attested" and not r["ambiguous"]),
            "confident_precision": round(sum(r["correct"] for r in rows if r["status"] == "attested" and not r["ambiguous"])
                                         / max(1, sum(1 for r in rows if r["status"] == "attested" and not r["ambiguous"])), 4),
            "not_found_rate": round(sum(r["status"] == "not_found" for r in rows) / len(rows), 4),
            "seconds_per_query": round(dt, 3), "failures": [r for r in rows if not r["correct"]][:25]}


def eval_spark(rng, n):
    from src.agents import spark
    cor = C.corpus()
    pool = [s for s in cor.sents if 8 <= len(C.words(s["text"])) <= 25]
    ep = [s for s in pool if s["ref"].startswith("D.L. 10.")]
    rest = [s for s in pool if not s["ref"].startswith("D.L. 10.")]
    sample = [("epicurus", s) for s in rng.sample(ep, n // 2)] + [("other_books", s) for s in rng.sample(rest, n - n // 2)]
    call = lambda m: spark.call_spark(m, timeout=90, max_retries=1)

    def one(item):
        group, s = item
        ws = [s["text"][a:b] for _, a, b in C.words(s["text"])]
        q = " ".join(ws[:max(C.MIN_WORDS, len(ws) // 2)])
        try:
            cands, dropped = C.conjecture(q, call, "start")
            err = None
        except Exception as e:  # provider/parse errors count as misses
            cands, dropped, err = [], 0, type(e).__name__
        missing = C.key_map(s["text"])[0].split()[len(C.words(q)):]

        def word_acc(c):
            # Words the model added after the fragment vs the words actually missing.
            got = C.key_map(c["after"])[0].split()
            return SequenceMatcher(None, missing, got, autojunk=False).ratio() if missing else 1.0
        by_agree = sorted(cands, key=lambda c: -(c["agreement"] or 0))
        return {"group": group, "ref": s["ref"], "query": q, "truth": s["text"], "error": err, "dropped": dropped,
                "top1": cands[0]["sentence"] if cands else None,
                "exact_top1": bool(cands and same(cands[0]["sentence"], s["text"])),
                "exact_top1_by_agreement": bool(by_agree and same(by_agree[0]["sentence"], s["text"])),
                "exact_any": any(same(c["sentence"], s["text"]) for c in cands),
                "word_acc_top1": round(word_acc(cands[0]), 3) if cands else 0.0,
                "source_named_top1": cands[0]["source"] if cands else None,
                "agreement_top1": cands[0]["agreement"] if cands else None}
    with ThreadPoolExecutor(4) as ex:
        rows = list(ex.map(one, sample))

    def summ(rs):
        k = len(rs) or 1
        return {"n": len(rs), "exact_top1": round(sum(r["exact_top1"] for r in rs) / k, 3),
                "exact_top1_by_agreement": round(sum(r["exact_top1_by_agreement"] for r in rs) / k, 3),
                "exact_any_of_5": round(sum(r["exact_any"] for r in rs) / k, 3),
                "mean_word_accuracy_top1": round(sum(r["word_acc_top1"] for r in rs) / k, 3),
                "errors": sum(1 for r in rs if r["error"])}
    return {"model": spark.SPARK_MODEL, "fragment": "first half of the sentence (at least 3 words), position given as start",
            "all": summ(rows), "epicurus": summ([r for r in rows if r["group"] == "epicurus"]),
            "other_books": summ([r for r in rows if r["group"] == "other_books"]), "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookup-only", action="store_true")
    ap.add_argument("--n-spark", type=int, default=40)
    a = ap.parse_args()
    old = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    res = {"corpus": C.corpus().source, "corpus_sentences": len(C.corpus().sents),
           "lookup": eval_lookup(random.Random(0))}
    print("lookup", json.dumps(res["lookup"]["accuracy"]), {k: res["lookup"][k] for k in ("ambiguous_rate", "confident_answers", "confident_precision", "not_found_rate")})
    if a.lookup_only:
        if "spark" in old:
            res["spark"] = old["spark"]
    else:
        res["spark"] = eval_spark(random.Random(1), a.n_spark)
        print("spark", json.dumps({k: res["spark"][k] for k in ("all", "epicurus", "other_books")}))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
