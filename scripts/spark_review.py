"""Muse Spark as a trace reviewer: does an automated trace stay on the clicked sheet?

  python scripts/spark_review.py render   # reviewer images (CT + trace + click; no mesh)
  python scripts/spark_review.py ask      # Muse Spark verdicts -> responses.jsonl (replayable)
  python scripts/spark_review.py score    # verdicts vs mesh-based truth -> summary.json

Reviews every seed of both first-slice modes (`curves/` = DP, `curves_walk/` =
walk). The model only sees CT crops, the red trace and a yellow click marker;
released meshes are used afterwards, for scoring only. Truth: a trace is on
the clicked layer when its median distance to that seed's mesh is <= 6 voxels
(`score.json`, `score_walk.json` from scripts/seeded_eval.py).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import tifffile  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from src.agents.spark import SPARK_MODEL, _img_url, call_spark  # noqa: E402

EVAL = ROOT / "reports" / "seeded_eval"
OUT = ROOT / "reports" / "spark_review"
SLICES = ROOT / "data" / "cache" / "scroll1_6100_6200"
Z0, LAST = 6100, 19
MODES = {"dp": ("curves", "score.json"), "walk": ("curves_walk", "score_walk.json")}
ON_LAYER_VX = 6.0
ZOOM_HALF = 160          # voxels around the click in the close-up panel

SYSTEM = (
    "You review automated tracing of papyrus layers in X-ray CT of a carbonized "
    "Herculaneum scroll. In each grayscale panel, bright bands are papyrus sheets "
    "and dark bands are gaps between sheets. A red line is an automated trace that "
    "must follow exactly ONE sheet: the sheet under the yellow dot, which a person "
    "clicked. Judge only what is visible. Reply with JSON only.")
QUESTION = (
    "Panel A: close-up around the click (slice 0). Panel B: whole trace, slice 0. "
    "Panel C: whole trace, slice 19 (the same sheet, deeper in the scroll).\n"
    "Does the red line stay on the same bright sheet as the yellow dot along its whole "
    "length in all panels, or does it cross a dark gap onto a neighbouring sheet or "
    "leave the sheet?\n"
    'Reply as {"verdict": "same_sheet" | "switches_sheet" | "unclear", '
    '"confidence": 0-1, "where": "short location of any switch", "reason": "one sentence"}')


def _seeds():
    return [r for r in json.loads((EVAL / "seeds.json").read_text()) if r.get("covers")]


def _curve(mode, sid, z):
    f = EVAL / MODES[mode][0] / sid / f"{z:05d}.jsonl"
    if not f.exists():
        return np.empty((0, 2))
    return np.asarray(json.loads(f.read_text().splitlines()[0])["points"], float).reshape(-1, 2)


def _panel(img, box, curve, click=None, size=640):
    x0, y0, x1, y1 = box
    crop = img[y0:y1, x0:x1].astype(float)
    lo, hi = np.percentile(crop, (1, 99))
    im = Image.fromarray(np.clip((crop - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)).convert("RGB")
    s = size / max(im.size)
    im = im.resize((max(1, int(im.size[0] * s)), max(1, int(im.size[1] * s))), Image.BILINEAR)
    d = ImageDraw.Draw(im)
    pts = [((x - x0) * s, (y - y0) * s) for x, y in curve]
    if len(pts) > 1:
        d.line(pts, fill=(255, 40, 40), width=2)
    if click is not None:
        cx, cy = (click[0] - x0) * s, (click[1] - y0) * s
        d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], outline=(255, 230, 0), width=3)
    return im


def _box(curves, h, w, margin=60):
    pts = np.concatenate([c for c in curves if len(c)])
    x0, y0 = np.floor(pts.min(0) - margin).astype(int)
    x1, y1 = np.ceil(pts.max(0) + margin).astype(int)
    return max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)


def cmd_render(_a):
    imgs = {z: tifffile.imread(SLICES / f"{Z0 + z:05d}.tif") for z in (0, LAST)}
    h, w = imgs[0].shape
    rows = []
    for mode in MODES:
        for r in _seeds():
            sid, (sx, sy, _) = r["id"], r["seed"]
            c0, c1 = _curve(mode, sid, 0), _curve(mode, sid, LAST)
            if not len(c0) and not len(c1):
                rows.append({"mode": mode, "id": sid, "image": None})
                continue
            zoom = (sx - ZOOM_HALF, sy - ZOOM_HALF, sx + ZOOM_HALF, sy + ZOOM_HALF)
            a = _panel(imgs[0], zoom, c0, click=(sx, sy))
            box = _box([c0, c1], h, w)
            b = _panel(imgs[0], box, c0, click=(sx, sy), size=960)
            c = _panel(imgs[LAST], box, c1, size=960)
            canvas = Image.new("RGB", (max(a.width, b.width, c.width), a.height + b.height + c.height + 60), (20, 20, 20))
            d = ImageDraw.Draw(canvas)
            y = 0
            for label, p in (("A", a), ("B", b), ("C", c)):
                d.text((6, y + 4), f"Panel {label}", fill=(255, 255, 255))
                canvas.paste(p, (0, y + 20))
                y += p.height + 20
            path = OUT / "images" / mode / f"{sid}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(path)
            rows.append({"mode": mode, "id": sid, "image": str(path.relative_to(ROOT))})
            print(mode, sid, canvas.size, flush=True)
    (OUT / "images.json").write_text(json.dumps(rows, indent=1))


TILE_HALF, TILES = 110, 6
QUESTION_TILES = (
    "Six close-ups (tiles 1-6) at full resolution along ONE automated trace on the "
    "same CT slice, ordered along the trace. Tile containing the yellow dot is where "
    "a person clicked the intended sheet. In every tile, does the red line run along "
    "the middle of a single bright sheet (the same sheet continuing from tile to "
    "tile), or does it cross a dark gap onto a neighbouring sheet?\n"
    'Reply as {"tiles": [{"tile": 1, "verdict": "same_sheet" | "switches_sheet" | '
    '"unclear", "confidence": 0-1}, ...], "verdict": "same_sheet" | "switches_sheet" | '
    '"unclear", "confidence": 0-1, "reason": "one sentence"}')


def cmd_render_tiles(_a):
    """v2: full-resolution close-ups spaced along the slice-0 trace."""
    img = tifffile.imread(SLICES / f"{Z0:05d}.tif")
    rows = []
    for mode in MODES:
        for r in _seeds():
            sid, (sx, sy, _) = r["id"], r["seed"]
            c0 = _curve(mode, sid, 0)
            if len(c0) < 2:
                rows.append({"mode": mode, "id": sid, "image": None})
                continue
            arc = np.r_[0, np.cumsum(np.hypot(*np.diff(c0, axis=0).T))]
            centres = np.interp(np.linspace(0, arc[-1], TILES), arc, np.arange(len(c0)))
            tiles = []
            for i, c in enumerate(centres):
                cx, cy = c0[int(round(c))]
                box = (int(cx) - TILE_HALF, int(cy) - TILE_HALF, int(cx) + TILE_HALF, int(cy) + TILE_HALF)
                near_click = abs(cx - sx) <= TILE_HALF and abs(cy - sy) <= TILE_HALF
                tiles.append(_panel(img, box, c0, click=(sx, sy) if near_click else None, size=440))
            canvas = Image.new("RGB", (3 * 440, 2 * (440 + 20)), (20, 20, 20))
            d = ImageDraw.Draw(canvas)
            for i, t in enumerate(tiles):
                x, y = (i % 3) * 440, (i // 3) * 460
                d.text((x + 6, y + 4), f"Tile {i + 1}", fill=(255, 255, 255))
                canvas.paste(t, (x, y + 20))
            path = OUT / "images_tiles" / mode / f"{sid}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(path)
            rows.append({"mode": mode, "id": sid, "image": str(path.relative_to(ROOT))})
    (OUT / "images_tiles.json").write_text(json.dumps(rows, indent=1))
    print(f"rendered {sum(r['image'] is not None for r in rows)} tile sheets")


def _parse(text):
    try:
        out = json.loads(text[text.index("{"):text.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return {"verdict": "unclear", "confidence": 0.0, "where": "", "reason": "unparseable reply"}
    if out.get("verdict") not in ("same_sheet", "switches_sheet", "unclear"):
        out["verdict"] = "unclear"
    try:
        out["confidence"] = float(out.get("confidence", 0.0))
    except (TypeError, ValueError):
        out["confidence"] = 0.0
    return out


VARIANTS = {"v1": ("images.json", "QUESTION", "responses.jsonl", "summary.json"),
            "v2": ("images_tiles.json", "QUESTION_TILES", "responses_tiles.jsonl", "summary_tiles.json")}


def cmd_ask(a):
    images, qname, resp_name, _ = VARIANTS[a.variant]
    question = globals()[qname]
    done = set()
    resp_file = OUT / resp_name
    if resp_file.exists() and not a.fresh:
        done = {(r["mode"], r["id"]) for r in map(json.loads, resp_file.read_text(encoding="utf-8").splitlines())}
    elif resp_file.exists():
        resp_file.unlink()
    for row in json.loads((OUT / images).read_text()):
        key = (row["mode"], row["id"])
        if row["image"] is None or key in done:
            continue
        img = ROOT / row["image"]
        t0 = time.time()
        text = call_spark([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [{"type": "text", "text": question},
                                         {"type": "image_url", "image_url": {"url": _img_url(img)}}]}])
        rec = {**row, "model": SPARK_MODEL, "seconds": round(time.time() - t0, 2),
               "image_sha256": hashlib.sha256(img.read_bytes()).hexdigest(),
               "prompt_sha256": hashlib.sha256((SYSTEM + question).encode()).hexdigest(), "variant": a.variant,
               "raw": text, "parsed": _parse(text)}
        with open(resp_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(row["mode"], row["id"], rec["parsed"]["verdict"], rec["parsed"]["confidence"],
              f"{rec['seconds']}s", flush=True)


def cmd_score(a):
    truth = {}
    for mode, (_, score_file) in MODES.items():
        for r in json.loads((EVAL / score_file).read_text())["seeds"]:
            if "median_dist" in r:
                truth[(mode, r["id"])] = {"on_layer": r["median_dist"] <= ON_LAYER_VX,
                                          "median_dist": r["median_dist"]}
    _, _, resp_name, summary_name = VARIANTS[a.variant]
    rows = []
    for rec in map(json.loads, (OUT / resp_name).read_text(encoding="utf-8").splitlines()):
        t = truth.get((rec["mode"], rec["id"]))
        if t is None:
            continue
        p = rec["parsed"]
        tile_switch = any(isinstance(t, dict) and t.get("verdict") == "switches_sheet"
                          for t in p.get("tiles", []) or [])
        accept = p["verdict"] == "same_sheet" and p["confidence"] >= a.accept_conf and not tile_switch
        rows.append({"mode": rec["mode"], "id": rec["id"], "on_layer": t["on_layer"],
                     "median_dist": t["median_dist"], "verdict": p["verdict"],
                     "confidence": p["confidence"], "auto_accept": accept,
                     "where": p.get("where", ""), "reason": p.get("reason", "")})
    n = len(rows)
    tp = sum(r["auto_accept"] and r["on_layer"] for r in rows)        # correctly accepted
    fp = sum(r["auto_accept"] and not r["on_layer"] for r in rows)    # wrong layer let through
    fn = sum(not r["auto_accept"] and r["on_layer"] for r in rows)    # good trace sent to review
    tn = sum(not r["auto_accept"] and not r["on_layer"] for r in rows)
    summary = {
        "variant": a.variant, "traces": n, "on_layer": tp + fn, "wrong_layer": fp + tn, "accept_conf": a.accept_conf,
        "auto_accepted": tp + fp, "sent_to_review": fn + tn,
        "accepted_and_correct": tp, "wrong_layer_let_through": fp,
        "wrong_layer_caught": tn, "good_sent_to_review": fn,
        "precision_of_accept": round(tp / (tp + fp), 3) if tp + fp else None,
        "wrong_layer_catch_rate": round(tn / (tn + fp), 3) if tn + fp else None,
        "review_load": round((fn + tn) / n, 3) if n else None,
    }
    (OUT / summary_name).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    for r in rows:
        print(f"{r['mode']:4} {r['id']} truth={'ON ' if r['on_layer'] else 'OFF'} "
              f"({r['median_dist']:5.1f}vx)  spark={r['verdict']:<15} {r['confidence']:.2f}"
              f"  {'ACCEPT' if r['auto_accept'] else 'review'}")
    print(json.dumps(summary, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["render", "render-tiles", "ask", "score"])
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="v1",
                    help="v1: overview panels; v2: full-resolution tiles along the trace")
    ap.add_argument("--fresh", action="store_true", help="ask: discard previous responses")
    ap.add_argument("--accept-conf", type=float, default=0.7)
    a = ap.parse_args(argv)
    {"render": cmd_render, "render-tiles": cmd_render_tiles, "ask": cmd_ask, "score": cmd_score}[a.step](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
