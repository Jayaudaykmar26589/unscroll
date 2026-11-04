"""Build the data behind viewer/explore.html (click explorer, review planner) and viewer/agent.html (agent replay).

  .venv\\Scripts\\python.exe scripts/build_explorer.py

Reads only what the evaluation already produced: the 89 seeded clicks (development, held-out
and the 59-click test set), their traces before the fix (dp), after it (walk) and from the
classic optical-flow tracer, the released hand-traced meshes, the triage ranking and the
Muse Wonder agent logs. Cuts a 512 x 256 voxel CT window around each click from the local
slice cache (Scroll 1, CC BY-NC 4.0) and scores every slice exactly as scripts/seeded_eval.py
does. Writes:
  viewer/explore-data.js            index of clicks, their scores, the triage ranking
  viewer/explore/<key>.js           per-click traces and reference, loaded on demand
  viewer/explore/<key>/<z>.webp     CT window for each slice
  viewer/explore/<key>.tifxyz.zip   Unscroll's trace as a tifxyz surface (VC3D format)
  viewer/agent-data.js              the three agent sessions, tool list and seam images
No GPU and no API call.
"""
import io
import json
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.meshref import SurfaceIndex, curve_xyz, sample_curve, slice_segments  # noqa: E402
from scripts.export_tifxyz import export  # noqa: E402

EVAL = ROOT / "reports" / "seeded_eval"
SLICES = ROOT / "data" / "cache" / "scroll1_6100_6200"
OUT = ROOT / "viewer"
EXP = OUT / "explore"
Z0, NZ = 6100, 20
WIN_W, WIN_H, SCALE = 512, 256, 1.5           # CT window around the click, in voxels; rendered 768 x 384
STEP = 3                                      # draw every 3rd voxel of a trace
SETS = [("main", "development", ""), ("heldout", "held-out", "_heldout"), ("large", "test", "_large")]
METHODS = [("before", "curves", "score"), ("after", "curves_walk", "score_walk"), ("flow", "curves_flow", "score_flow")]


def mesh(sid):
    d = np.load(EVAL / "mesh_cache" / f"{sid}.npz")
    return d["v"], d["f"]


def load_curves(folder):
    out = {}
    for f in sorted(folder.glob("[0-9]*.jsonl")) if folder.exists() else []:
        rec = json.loads(f.read_text().splitlines()[0])
        out[int(rec["source_z"])] = rec
    return out


def clip_line(pts, x0, y0, step=STEP):
    """Points inside the window (with margin), in window coordinates, split where they leave it."""
    out, cur = [], []
    for p in pts[::step]:
        x, y = p[0] - x0, p[1] - y0
        if -20 <= x <= WIN_W + 20 and -20 <= y <= WIN_H + 20:
            cur.append([round(float(x), 1), round(float(y), 1)])
        elif cur:
            out.append(cur); cur = []
    if len(cur) > 1:
        out.append(cur)
    return [c for c in out if len(c) > 1]


def main():
    EXP.mkdir(parents=True, exist_ok=True)
    clicks = []
    for set_name, label, suffix in SETS:
        seeds = [r for r in json.loads((EVAL / f"seeds{'' if set_name == 'main' else '_' + set_name}.json").read_text()) if r.get("covers")]
        scores = {}
        for m, _, sname in METHODS:
            f = EVAL / f"{sname}{suffix}.json"
            if f.exists():
                scores[m] = {r.get("key", r["id"]): r for r in json.loads(f.read_text())["seeds"]}
        for r in seeds:
            key = r.get("key", r["id"]) if set_name == "large" else (r["id"] if set_name == "main" else r["id"] + "_h")
            src_key = r.get("key", r["id"])
            clicks.append({"key": key, "src_key": src_key, "id": r["id"], "set": label, "suffix": suffix, "seed": r["seed"],
                           "score": {m: {k: scores[m][src_key].get(k) for k in ("within5", "median_dist", "slices_with_trace")}
                                     for m in scores if src_key in scores[m]}})
    print(len(clicks), "clicks")

    # CT windows: one pass over the 20 slices, cropping every click
    for z in range(Z0, Z0 + NZ):
        img = tifffile.imread(SLICES / f"{z:05d}.tif")
        H, W = img.shape
        for c in clicks:
            x0 = int(np.clip(c["seed"][0] - WIN_W // 2, 0, W - WIN_W)); y0 = int(np.clip(c["seed"][1] - WIN_H // 2, 0, H - WIN_H))
            c["win"] = [x0, y0]
            crop = img[y0:y0 + WIN_H, x0:x0 + WIN_W].astype(np.float32)
            if z == Z0:
                c["lo"], c["hi"] = np.percentile(crop, [1, 99.6])
            a = np.clip((crop - c["lo"]) / max(1.0, c["hi"] - c["lo"]), 0, 1) * 255
            im = Image.fromarray(a.astype(np.uint8)).resize((int(WIN_W * SCALE), int(WIN_H * SCALE)), Image.LANCZOS)
            d = EXP / c["key"]; d.mkdir(exist_ok=True)
            im.save(d / f"{z - Z0:02d}.webp", "WEBP", quality=50, method=6)
        print("slice", z, flush=True)

    index = []
    for c in clicks:
        v, f = mesh(c["id"])
        target = SurfaceIndex(v, f)
        x0, y0 = c["win"]
        per = {"ref": [], "before": [], "after": [], "flow": []}
        dist = {"before": [], "after": [], "flow": []}
        curves = {m: load_curves(EVAL / f"{folder}{c['suffix']}" / c["src_key"]) for m, folder, _ in METHODS}
        for k in range(NZ):
            z = Z0 + k
            segs = slice_segments(v, f, z)
            ref = []
            for s in segs:
                a, b = s[0] - [x0, y0], s[1] - [x0, y0]
                if (-20 <= a[0] <= WIN_W + 20 and -20 <= a[1] <= WIN_H + 20) or (-20 <= b[0] <= WIN_W + 20 and -20 <= b[1] <= WIN_H + 20):
                    ref.append([round(float(a[0]), 1), round(float(a[1]), 1), round(float(b[0]), 1), round(float(b[1]), 1)])
            per["ref"].append(ref)
            for m in dist:
                rec = curves[m].get(z)
                pts = np.asarray(rec["points"], float).reshape(-1, 2) if rec else np.empty((0, 2))
                per[m].append(clip_line(pts, x0, y0) if len(pts) else [])
                if len(pts) >= 2:
                    q, w = sample_curve(curve_xyz(rec), 1.0)
                    d_t = target.distances(q)
                    wn = w / w.sum()
                    dist[m].append([round(float(np.median(d_t)), 2), round(float(wn[d_t <= 5].sum()), 3)])
                else:
                    dist[m].append(None)
        # tifxyz of Unscroll's trace (after the fix)
        tz = None
        after_dir = EVAL / f"curves_walk{c['suffix']}" / c["src_key"]
        if curves["after"]:
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / f"unscroll_{c['key']}"
                export(after_dir, c["seed"][:2], out, sigma=0.0, name=f"unscroll_{c['key']}")
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
                    for fn in ("x.tif", "y.tif", "z.tif", "meta.json"):
                        zf.write(out / fn, f"unscroll_{c['key']}/{fn}")
                (EXP / f"{c['key']}.tifxyz.zip").write_bytes(buf.getvalue())
                tz = f"explore/{c['key']}.tifxyz.zip"
        (EXP / f"{c['key']}.js").write_text("window.UNSCROLL_CLICK=window.UNSCROLL_CLICK||{};window.UNSCROLL_CLICK[" + json.dumps(c["key"]) + "]="
                                            + json.dumps({"lines": per, "dist": dist}, separators=(",", ":")) + ";\n", encoding="utf-8")
        index.append({"key": c["key"], "id": c["id"], "set": c["set"], "seed": c["seed"], "win": c["win"], "score": c["score"], "tifxyz": tz})
        print(c["key"], flush=True)

    # review planner: the triage ranking, recomputed and checked against reports/triage/triage.json
    t = json.loads((ROOT / "reports/triage/triage.json").read_text())
    triage = {}
    for s in ("dev", "heldout"):
        rows = sorted((r for r in t["rows"] if r["set"] == s), key=lambda r: -r["risk"])
        wrong = sum(not r["on_layer"] for r in rows)
        curve = [[round(k / len(rows), 4), round(sum(not r["on_layer"] for r in rows[:k]) / wrong, 4)] for k in range(len(rows) + 1)]
        if s == "heldout":
            assert all(abs(a[1] - b[1]) < 1e-3 for a, b in zip(curve, t["heldout"]["curve"])), "triage curve does not match triage.json"
        triage[s] = {"rows": [{"id": r["id"], "mode": r["mode"], "risk": r["risk"], "on_layer": r["on_layer"], "median_dist": r["median_dist"],
                               "key": r["id"] if s == "dev" else r["id"] + "_h"} for r in rows],
                     "wrong": wrong, "auc": t[s]["auc"]}
    triage["rule"], triage["note"], triage["on_layer_vx"] = t["rule"], t["note"], t["on_layer_vx"]
    (OUT / "explore-data.js").write_text("window.UNSCROLL_EXPLORE=" + json.dumps({"clicks": index, "triage": triage, "z0": Z0, "win": [WIN_W, WIN_H], "scale": SCALE},
                                         separators=(",", ":")) + ";\n", encoding="utf-8")

    # agent replay
    src = (ROOT / "scripts/grow_agent.py").read_text(encoding="utf-8")
    tools = [{"name": n, "description": d} for n, d in re.findall(r'\{"name": "([a-z_]+)", "description": "([^"]+)"', src)]
    sessions = []
    for f, title in (("20231210121321_20261030T113950.jsonl", "Session 1 · extend the trace"),
                     ("20231210121321_20261030T120843.jsonl", "Session 2 · spend the last GPU run"),
                     ("20231210121321_20261031T180912.jsonl", "Session 3 · check every seam")):
        events = [json.loads(l) for l in (ROOT / "reports/agent" / f).read_text(encoding="utf-8").splitlines() if l.strip()]
        for e in events:
            img = (e.get("result") or {}).get("image") if isinstance(e.get("result"), dict) else None
            if img:
                name = Path(img.replace("\\", "/")).name
                shutil.copyfile(ROOT / "reports/agent" / name, OUT / "eval" / f"agent_{name}")
                e["result"]["image"] = f"eval/agent_{name}"
        sessions.append({"file": f"reports/agent/{f}", "title": title, "events": events})
    ex = json.loads((ROOT / "reports/flatten/excluded_20231210121321.json").read_text())
    (OUT / "agent-data.js").write_text("window.UNSCROLL_AGENT=" + json.dumps({"tools": tools, "sessions": sessions, "excluded": ex}, ensure_ascii=False) + ";\n", encoding="utf-8")
    size = sum(p.stat().st_size for p in EXP.rglob("*") if p.is_file())
    print(f"explore: {len(index)} clicks, {size / 1e6:.1f} MB; agent: {len(tools)} tools, {sum(len(s['events']) for s in sessions)} events")


if __name__ == "__main__":
    main()
