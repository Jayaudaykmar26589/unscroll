"""Paired comparison of tracers on the same clicks, with bootstrap confidence intervals.

  python scripts/compare_methods.py      -> reports/seeded_eval/compare.json

Reads score*.json from scripts/seeded_eval.py score. Per click the measure is
the share of the trace within 5 voxels (40 um) of the clicked human-traced
layer; a click with no trace counts as 0, so stopping early is not rewarded.
Groups: development (15 clicks used while building the fix) and new clicks
(held-out 15 + large set, never used in development).
"""
import json
import sys
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "seeded_eval"
METHODS = {"before": "dp", "after": "walk", "flow": "flow", "flowridge": "flowridge"}
LABELS = {"before": "SAM 3.1, original first slice", "after": "SAM 3.1 + first-slice fix (Unscroll)",
          "flow": "Optical-flow baseline", "flowridge": "Optical flow + ridge snap baseline"}
SETS = {"development": ["main"], "new_clicks": ["heldout", "large"]}
B, RNG = 10000, np.random.default_rng(0)


def scores(method, seed_set):
    mode = METHODS[method]
    name = "score" + ("" if mode == "dp" else f"_{mode}") + ("" if seed_set == "main" else f"_{seed_set}") + ".json"
    path = OUT / name
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return {r.get("key", r["id"]): (r.get("within5", 0.0), r["slices_with_trace"], r["slices"]) for r in d["seeds"]}


def ci(x, stat):
    idx = RNG.integers(0, len(x), (B, len(x)))
    v = np.array([stat(x[i]) for i in idx])
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


def sign_p(wins, losses):
    n = wins + losses
    k = min(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def main():
    report = {"measure": "share of trace within 5 voxels of the clicked layer, per click; no trace = 0",
              "labels": LABELS, "groups": {}}
    for group, sets in SETS.items():
        per = {m: {} for m in METHODS}
        for s in sets:
            for m in METHODS:
                got = scores(m, s)
                if got:
                    per[m].update({f"{s}:{k}": v for k, v in got.items()})
        common = sorted(set.intersection(*(set(v) for v in per.values() if v))) if any(per.values()) else []
        if not common:
            continue
        g = {"clicks": len(common), "sets": sets, "methods": {}, "vs_after": {}}
        for m in METHODS:
            if not per[m]:
                continue
            x = np.array([per[m][k][0] for k in common])
            g["methods"][m] = {"median": round(float(np.median(x)), 4), "median_ci95": ci(x, np.median),
                               "mean": round(float(x.mean()), 4), "mean_ci95": ci(x, np.mean),
                               "slices_traced": int(sum(per[m][k][1] for k in common)),
                               "slices_total": int(sum(per[m][k][2] for k in common))}
        a = np.array([per["after"][k][0] for k in common])
        for m in METHODS:
            if m == "after" or not per[m]:
                continue
            d = a - np.array([per[m][k][0] for k in common])
            wins, losses = int((d > 0.01).sum()), int((d < -0.01).sum())
            g["vs_after"][m] = {"mean_gain": round(float(d.mean()), 4), "mean_gain_ci95": ci(d, np.mean),
                                "after_better": wins, "after_worse": losses, "ties": len(d) - wins - losses,
                                "sign_test_p": round(sign_p(wins, losses), 5)}
        report["groups"][group] = g
    (OUT / "compare.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    sys.exit(main())
