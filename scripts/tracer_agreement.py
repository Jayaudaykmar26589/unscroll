"""Two tracers as a reference-free error flag: where SAM 3.1 and optical flow disagree.

  python scripts/tracer_agreement.py     -> reports/seeded_eval/agreement.json

Both start from the same first-slice curve. Per click, disagreement = median
over slices of the distance from each SAM 3.1 (fixed) point to the nearest
optical-flow point. No reference is used. Scored afterwards against the
human-traced layer: a trace is wrong-layer when its median distance to the
clicked layer exceeds 6 voxels (the Evaluation page's on-layer rule). AUC is
computed separately for the development clicks and the new clicks.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.seeded_eval import OUT, _suffix  # noqa: E402

ON_LAYER_VX = 6.0
SETS = {"development": ["main"], "new_clicks": ["heldout", "large"]}


def curves(folder):
    out = {}
    for f in sorted(folder.glob("[0-9]*.jsonl")):
        rec = json.loads(f.read_text().splitlines()[0])
        if len(rec["points"]) >= 2:
            out[int(rec["source_z"])] = np.asarray(rec["points"], float)
    return out


def disagreement(a, b):
    ds = []
    for z, p in a.items():
        if z in b:
            q = b[z]
            d = np.sqrt(((p[:, None, :] - q[None, ::2, :]) ** 2).sum(-1)).min(1)
            ds.append(np.median(d))
    return float(np.median(ds)) if ds else None


def auc(score, bad):
    pos, neg = [s for s, b in zip(score, bad) if b], [s for s, b in zip(score, bad) if not b]
    if not pos or not neg:
        return None
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def main():
    report = {"rule": f"wrong layer = median distance to the clicked layer > {ON_LAYER_VX} voxels",
              "signal": "median distance between SAM 3.1 (fixed) and optical-flow traces, voxels", "groups": {}}
    for group, sets in SETS.items():
        rows = []
        for s in sets:
            a = type("A", (), {"seed_set": s})()
            score = json.loads((OUT / f"score{_suffix(a, 'walk')}.json").read_text())
            for r in score["seeds"]:
                key = r.get("key", r["id"])
                if r.get("median_dist") is None:
                    continue
                d = disagreement(curves(OUT / f"curves{_suffix(a, 'walk')}" / key),
                                 curves(OUT / f"curves{_suffix(a, 'flow')}" / key))
                if d is not None:
                    rows.append({"key": f"{s}:{key}", "disagreement_vx": round(d, 2),
                                 "median_dist": r["median_dist"], "wrong_layer": r["median_dist"] > ON_LAYER_VX})
        bad = [r["wrong_layer"] for r in rows]
        sig = [r["disagreement_vx"] for r in rows]
        order = np.argsort(sig)[::-1]
        nbad = sum(bad)
        caught, review_all = 0, None
        for i, j in enumerate(order, 1):
            caught += bad[j]
            if nbad and caught == nbad and review_all is None:
                review_all = i / len(rows)
        flag = [s > 3.0 for s in sig]
        report["groups"][group] = {
            "traces": len(rows), "wrong_layer": nbad, "auc": auc(sig, bad),
            "review_to_catch_all": review_all,
            "flag_over_3vx": {"flagged": int(sum(flag)), "wrong_layer_caught": int(sum(f and b for f, b in zip(flag, bad))),
                              "good_traces_flagged": int(sum(f and not b for f, b in zip(flag, bad)))},
            "rows": rows}
    (OUT / "agreement.json").write_text(json.dumps(report, indent=2))
    for g, v in report["groups"].items():
        print(g, {k: v[k] for k in ("traces", "wrong_layer", "auc", "review_to_catch_all", "flag_over_3vx")})


if __name__ == "__main__":
    main()
