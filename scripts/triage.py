"""Slice triage: rank traces by reference-free risk so a reviewer checks the riskiest first.

For every seeded trace (development and held-out clicks, both first-slice modes)
we compute signals that need no human-traced mesh:
  click_offset    distance from the user's click to the trace on the first slice
  ridge_contrast  is the trace on a bright sheet (vs the gap beside it)?
  mode_disagree   distance between the two first-slice methods' traces
  coverage        share of slices where tracking kept going
  length_jump     largest slice-to-slice change in curve length
The human-traced meshes are used only to label each trace on/off layer and to
score the ranking. The rule is fixed on the development clicks, then reported
on the held-out clicks.

  python scripts/triage.py   -> reports/triage/triage.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
SE = ROOT / "reports/seeded_eval"
CT = ROOT / "data/cache/scroll1_6100_6200"
OUT = ROOT / "reports/triage"
ON_LAYER_VX = 6.0          # same threshold as the Evaluation page
SETS = {"dev": ("", "seeds.json"), "heldout": ("_heldout", "seeds_heldout.json")}
MODES = {"dp": "", "walk": "_walk"}


def curves(folder):
    out = []
    for f in sorted(folder.glob("[0-9]*.jsonl")):
        r = json.loads(f.read_text().splitlines()[0])
        out.append((r["source_z"], np.asarray(r["points"], float).reshape(-1, 2)))
    return out


def arc_length(p):
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


_slices = {}


def ct(z):
    if z not in _slices:
        _slices[z] = tifffile.memmap(CT / f"{z:05d}.tif")
    return _slices[z]


def bilinear(img, x, y):
    x0 = np.clip(np.floor(x).astype(int), 0, img.shape[1] - 2)
    y0 = np.clip(np.floor(y).astype(int), 0, img.shape[0] - 2)
    fx, fy = np.clip(x - x0, 0, 1), np.clip(y - y0, 0, 1)
    a, b = img[y0, x0].astype(float), img[y0, x0 + 1].astype(float)
    c, d = img[y0 + 1, x0].astype(float), img[y0 + 1, x0 + 1].astype(float)
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def ridge_contrast(p, z, off=5.0):
    """Median of (I(curve) - mean I(±off along the normal)) / I(curve): high on a sheet's bright core."""
    if len(p) < 9:
        return float("nan")
    t = np.gradient(p, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    img = ct(z)
    i0 = bilinear(img, p[:, 0], p[:, 1])
    side = 0.5 * (bilinear(img, *(p + off * n).T) + bilinear(img, *(p - off * n).T))
    return float(np.median((i0 - side) / (i0 + 1e-9)))


def nearest(a, b):
    """Median distance from points a to curve b (subsampled)."""
    if not len(a) or not len(b):
        return float("nan")
    a, b = a[::4], b[::2]
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min(1)
    return float(np.median(d))


def features(set_name, mode, sid, seed):
    sfx, _ = SETS[set_name]
    cs = curves(SE / f"curves{MODES[mode]}{sfx}" / sid)
    other = curves(SE / f"curves{MODES['walk' if mode == 'dp' else 'dp']}{sfx}" / sid)
    traced = [(z, p) for z, p in cs if len(p)]
    first = traced[0] if traced else (seed[2], np.zeros((0, 2)))
    lens = [arc_length(p) for _, p in cs]
    jumps = [abs(np.log((b + 1) / (a + 1))) for a, b in zip(lens, lens[1:]) if a and b]
    click = np.asarray(seed[:2], float)
    p0 = cs[0][1] if cs else np.zeros((0, 2))
    return {
        "click_offset": float(np.sqrt(((p0 - click) ** 2).sum(1)).min()) if len(p0) else 99.0,
        "ridge_contrast": ridge_contrast(first[1], first[0]),
        "mode_disagree": nearest(p0, other[0][1]) if other and len(p0) else 99.0,
        "coverage": len(traced) / max(1, len(cs)),
        "length_jump": float(max(jumps)) if jumps else 0.0,
    }


def risk(f):
    """Fixed rule, chosen on the development clicks (see RESULTS in triage.json).
    Each term is scaled to roughly 0..1 'risk'; the sum ranks traces."""
    r = min(f["click_offset"], 12.0) / 12.0
    r += min(f["mode_disagree"], 12.0) / 12.0
    rc = f["ridge_contrast"]
    r += 1.0 if not np.isfinite(rc) else float(np.clip((0.15 - rc) / 0.15, 0, 1))
    r += 1.0 - f["coverage"]
    r += min(f["length_jump"], 1.0)
    return r


DEV_SELECTED = ("click_offset", "ridge_contrast", "mode_disagree")   # signals with dev AUC > 0.6 (v1 run)
HIGH_IS_SAFE = ("ridge_contrast", "coverage")


def rank_risk(rows, signals):
    """v2: mean percentile rank of the chosen signals within the batch under review."""
    n = len(rows)
    score = np.zeros(n)
    for k in signals:
        v = np.array([r[k] if np.isfinite(r[k]) else (np.inf if k in HIGH_IS_SAFE else -np.inf) for r in rows], float)
        v = -v if k in HIGH_IS_SAFE else v
        score += np.argsort(np.argsort(v, kind="stable"), kind="stable") / max(1, n - 1)
    return score / len(signals)


def curve(rows):
    """Reviewing the riskiest k traces: share of wrong-layer traces caught, for k = 0..n."""
    order = sorted(rows, key=lambda r: -r["risk"])
    bad = sum(not r["on_layer"] for r in rows) or 1
    pts, caught = [(0.0, 0.0)], 0
    for k, r in enumerate(order, 1):
        caught += not r["on_layer"]
        pts.append((k / len(rows), caught / bad))
    return pts


def auc(rows):
    bad = [r["risk"] for r in rows if not r["on_layer"]]
    good = [r["risk"] for r in rows if r["on_layer"]]
    if not bad or not good:
        return None
    return float(np.mean([(b > g) + 0.5 * (b == g) for b in bad for g in good]))


def caught_at(pts, frac):
    return max(c for f, c in pts if f <= frac + 1e-9)


def review_needed(pts, target):
    return min(f for f, c in pts if c >= target - 1e-9)


def main():
    rows = []
    for set_name, (sfx, seeds_file) in SETS.items():
        seeds = {s["id"]: s["seed"] for s in json.loads((SE / seeds_file).read_text()) if s.get("seed")}
        for mode, msfx in MODES.items():
            score = json.loads((SE / f"score{msfx}{sfx}.json").read_text())["seeds"]
            for s in score:
                if s["id"] not in seeds or s.get("median_dist") is None:
                    continue
                f = features(set_name, mode, s["id"], seeds[s["id"]])
                rows.append({"set": set_name, "mode": mode, "id": s["id"], **{k: round(v, 4) for k, v in f.items()},
                             "median_dist": s["median_dist"], "on_layer": s["median_dist"] <= ON_LAYER_VX})
    for r in rows:
        r["risk_v1"] = round(risk(r), 4)
    for name in ("dev", "heldout"):
        rs = [r for r in rows if r["set"] == name]
        for r, v in zip(rs, rank_risk(rs, DEV_SELECTED)):
            r["risk"] = round(float(v), 4)
    out = {"on_layer_vx": ON_LAYER_VX, "signals": ["click_offset", "ridge_contrast", "mode_disagree", "coverage", "length_jump"],
           "rule": "v2: mean rank of the signals with development AUC > 0.6 (click offset, ridge contrast, method disagreement), ranked within each batch",
           "rule_v1": "a-priori sum of all five clipped risk terms; written before any results",
           "note": "v1 was evaluated first and failed on held-out (AUC 0.52); v2's signal choice used development numbers only, but was made after seeing v1's held-out result."}
    for name in ("dev", "heldout", "all"):
        rs = [r for r in rows if name == "all" or r["set"] == name]
        pts = curve(rs)
        per_signal = {}
        for k in out["signals"]:
            sign = -1 if k in ("ridge_contrast", "coverage") else 1
            v = [dict(r, risk=sign * (r[k] if np.isfinite(r[k]) else 0)) for r in rs]
            per_signal[k] = round(auc(v), 3) if auc(v) is not None else None
        if name == "all":   # ranks are per batch, so pooled ranking re-ranks the union
            for r, v in zip(rs, rank_risk(rs, DEV_SELECTED)):
                r["risk_all"] = round(float(v), 4)
            pts = curve([dict(r, risk=r["risk_all"]) for r in rs])
        v1 = [dict(r, risk=r["risk_v1"]) for r in rs]
        out[name] = {"traces": len(rs), "wrong_layer": sum(not r["on_layer"] for r in rs),
                     "auc": round(auc([dict(r, risk=r.get("risk_all", r["risk"])) for r in rs] if name == "all" else rs), 3),
                     "auc_v1": round(auc(v1), 3),
                     "caught_at_25": round(caught_at(pts, 0.25), 3), "caught_at_50": round(caught_at(pts, 0.5), 3),
                     "review_for_80": round(review_needed(pts, 0.8), 3), "review_for_100": round(review_needed(pts, 1.0), 3),
                     "auc_per_signal": per_signal, "curve": [[round(a, 4), round(b, 4)] for a, b in pts]}
    out["rows"] = rows
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "triage.json").write_text(json.dumps(out, indent=1))
    for name in ("dev", "heldout", "all"):
        print(name, {k: v for k, v in out[name].items() if k != "curve"})


if __name__ == "__main__":
    sys.exit(main())
