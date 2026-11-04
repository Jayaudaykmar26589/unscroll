"""Muse Wonder patch-growing agent: extend a traced sheet by re-seeding where tracking stopped.

  python scripts/grow_agent.py --seed-id 20231210121321 --max-runs 3
  python scripts/grow_agent.py --seed-id 20231210121321 --status     # tools only, no model

Muse Wonder (tool calling, Meta Model API) decides what to do; the tools act:
  get_patch_status  traced / failed slice ranges, reasons, lengths, provenance
  propose_reseed    deterministic click at the arc-midpoint of a good curve
  run_trace         Kaggle T4 crop job from that click (blocks ~10-15 min)
  score_patch       exact distance to the seed's released mesh where it exists
  flatten_patch     unroll the merged surface (scripts/flatten_trace.py)
  finish            end the session with a summary
Hard limits live in code, not in the prompt: at most --max-runs Kaggle jobs,
clicks must lie inside the crop, and the tracker's guards apply per slice.
Merge rule: each slice comes from the earliest run that traced it with status
"ok"; provenance is recorded. Every message and tool result is logged to
reports/agent/<seed>_<time>.jsonl for replay.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

FLAT = ROOT / "reports" / "flatten"
AGENT = ROOT / "reports" / "agent"
MODEL = "muse-wonder-1.0"
PY = sys.executable


# ---------------------------------------------------------------- patch state
def _crop_meta(seed_id):
    return json.loads((ROOT / "data" / "cache" / f"crop_{seed_id}" / "crop.json").read_text())


def _runs(seed_id):
    """[(tag, curves dir, {z: status})] in creation order."""
    out = []
    for d in sorted(FLAT.glob(f"curves_{seed_id}_*"), key=lambda p: p.stat().st_mtime):
        tag = d.name[len(f"curves_{seed_id}_"):]
        if tag == "merged":                     # flatten_patch output, not a tracing run
            continue
        status = {}
        log = FLAT / f"trace_{seed_id}_{tag}.log"
        if log.exists():                                   # stdout form: "z=6159 pts=.. ok"
            text = log.read_text(encoding="utf-8", errors="replace")
            for m in re.finditer(r"^z=(\d+) pts=\d+ (\w+)", text, re.M):
                status[int(m.group(1))] = m.group(2)
        for f in (FLAT / f"runlogs_{seed_id}_{tag}").glob("run_*.jsonl"):
            meta = _crop_meta(seed_id)
            for line in f.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                if "status" in rec:
                    reason = rec.get("checks", {}).get("extraction", {}).get("reason")
                    status[meta["z0"] + rec["z"]] = rec["status"] if rec["status"] != "failed" else f"failed:{reason}"
        out.append((tag, d, status))
    return out


def _curve(path):
    rec = json.loads(path.read_text().splitlines()[0])
    return rec, np.asarray(rec["points"], float).reshape(-1, 2)


def _excluded(seed_id):
    f = FLAT / f"excluded_{seed_id}.json"
    return json.loads(f.read_text()) if f.exists() else {}


def merged(seed_id):
    """{z: (tag, record, points)} using the earliest run with an ok curve."""
    best = {}
    excluded = _excluded(seed_id)
    for tag, d, status in _runs(seed_id):
        if tag in excluded:
            continue
        for f in sorted(d.glob("[0-9]*.jsonl")):
            rec, pts = _curve(f)
            z = rec["source_z"]
            ok = status.get(z, "ok" if len(pts) > 1 else "failed") in ("ok", "reprompted")
            if ok and len(pts) > 1 and z not in best:
                best[z] = (tag, rec, pts)
    return best


def _ranges(zs):
    zs = sorted(zs)
    out, start = [], None
    for i, z in enumerate(zs):
        if start is None:
            start = z
        if i == len(zs) - 1 or zs[i + 1] != z + 1:
            out.append([start, z])
            start = None
    return out


# ---------------------------------------------------------------------- tools
def get_patch_status(seed_id):
    meta = _crop_meta(seed_id)
    z_all = list(range(meta["z0"], meta["z0"] + meta["n"]))
    m = merged(seed_id)
    reasons = {}
    for tag, _, status in _runs(seed_id):
        for z, s in status.items():
            if s.startswith("failed") and z not in m:
                reasons[s] = reasons.get(s, 0) + 1
    lengths = {z: round(float(np.hypot(*np.diff(p, axis=0).T).sum())) for z, (_, _, p) in m.items()}
    return {
        "seed_id": seed_id, "slices_in_crop": [z_all[0], z_all[-1]],
        "traced_ranges": _ranges(m), "untraced_ranges": _ranges(set(z_all) - set(m)),
        "traced_slices": len(m), "total_slices": len(z_all),
        "failure_reasons": reasons,
        "last_good_slice": max(m) if m else None,
        "curve_length_vx_at": {str(z): lengths[z] for z in sorted(lengths)[::10]},
        "runs": [{"tag": t, "slices_with_status": len(s)} for t, _, s in _runs(seed_id)],
        "kaggle_runs_left": STATE["max_runs"] - STATE["runs_used"],
    }


def propose_reseed(seed_id, z):
    m = merged(seed_id)
    z = int(z)
    if z not in m:
        return {"error": f"slice {z} has no good curve; choose one of {_ranges(m)}"}
    _, _, p = m[z]
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    i = int(np.argmin(abs(s - s[-1] / 2)))
    return {"click": [int(round(p[i, 0])), int(round(p[i, 1])), z],
            "note": "arc-midpoint of the last good curve; the tracker re-anchors here"}


def run_trace(seed_id, click, window=0, tag=""):
    if STATE["runs_used"] >= STATE["max_runs"]:
        return {"error": "Kaggle run budget for this session is used up"}
    tag = tag or f"agent{STATE['runs_used'] + 1}"
    click_s = ",".join(str(int(c)) for c in click)
    base = [PY, str(ROOT / "scripts" / "kaggle_run.py")]
    args = ["--crop", seed_id, "--click", click_s, "--tag", tag] + (["--window", str(window)] if window else [])
    STATE["runs_used"] += 1
    t0 = time.time()
    for step in ("stage", "push-kernel"):
        for _ in range(45):                     # Kaggle allows 2 GPU sessions; wait up to ~90 min for a slot
            r = subprocess.run(base + [step] + args, cwd=ROOT, capture_output=True, text=True)
            if "Maximum batch GPU session" not in r.stdout + r.stderr:
                break
            time.sleep(120)
        if r.returncode or "push error" in r.stdout.lower():
            return {"error": f"{step} failed", "detail": (r.stdout + r.stderr)[-600:]}
    time.sleep(60)
    while True:
        r = subprocess.run(base + ["status"] + args, cwd=ROOT, capture_output=True, text=True)
        if not re.search(r"RUNNING|QUEUED", r.stdout):
            break
        time.sleep(60)
    final = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "unknown"
    r = subprocess.run(base + ["fetch"] + args, cwd=ROOT, capture_output=True, text=True)
    out_log = ROOT / "kaggle" / "output" / "unscroll-crop-trace" / "trace.log"
    if out_log.exists():
        (FLAT / f"trace_{seed_id}_{tag}.log").write_text(out_log.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    return {"tag": tag, "kaggle_status": final, "minutes": round((time.time() - t0) / 60, 1),
            "status_after": get_patch_status(seed_id)}


def score_patch(seed_id):
    from scripts.seeded_eval import Z1, _window_mesh
    from src.eval.meshref import SurfaceIndex, curve_xyz, sample_curve
    idx = SurfaceIndex(*_window_mesh(seed_id))
    zmax = Z1   # cached mesh triangles are complete only through the evaluated block
    ds = []
    for z, (_, rec, _) in sorted(merged(seed_id).items()):
        if z > zmax:
            continue
        q, _ = sample_curve(curve_xyz(rec), 1.0)
        if len(q):
            ds.append(idx.distances(q))
    if not ds:
        return {"error": "no traced slices inside the mesh window"}
    d = np.concatenate(ds)
    return {"scored_slices": f"<= {zmax}", "within5": round(float((d <= 5).mean()), 4),
            "within2": round(float((d <= 2).mean()), 4), "median_vx": round(float(np.median(d)), 2)}


def flatten_patch(seed_id):
    from scripts.flatten_trace import flatten
    out = FLAT / f"curves_{seed_id}_merged"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.jsonl"):
        f.unlink()
    prov = {}
    for z, (tag, rec, _) in sorted(merged(seed_id).items()):
        (out / f"{z:05d}.jsonl").write_text(json.dumps(rec) + "\n")
        prov[z] = tag
    (FLAT / f"provenance_{seed_id}_merged.json").write_text(json.dumps(prov, indent=1))
    meta = _crop_meta(seed_id)
    m = flatten(out, meta["seed"][:2], FLAT / f"sheet_{seed_id}_merged")
    return {"image_rows": m["image_rows"], "columns": m["columns"], "rows_traced": m["rows_traced"],
            "z_range_traced": m["z_range_traced"], "runs_used_per_slice": {t: list(prov.values()).count(t) for t in set(prov.values())}}


def _render_pair(seed_id, z_prev, p_prev, z_new, p_new, path):
    import tifffile
    from PIL import Image, ImageDraw
    meta = _crop_meta(seed_id)
    ox, oy = meta["offset_xy"]
    crop_dir = ROOT / "data" / "cache" / f"crop_{seed_id}"
    pts = np.vstack([p_prev, p_new]) - [ox, oy]
    x0, y0 = (pts.min(0) - 60).astype(int).clip(0)
    x1, y1 = (pts.max(0) + 60).astype(int)
    panels = []
    for z, p, color in ((z_prev, p_prev, (60, 220, 90)), (z_new, p_new, (255, 50, 50))):
        img = tifffile.imread(crop_dir / f"{z:05d}.tif")[y0:y1, x0:x1].astype(float)
        lo, hi = np.percentile(img, (1, 99))
        im = Image.fromarray(np.clip((img - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)).convert("RGB")
        d = ImageDraw.Draw(im)
        d.line([(x - ox - x0, y - oy - y0) for x, y in p], fill=color, width=2)
        d.text((6, 4), f"slice {z}", fill=(255, 255, 0))
        panels.append(im)
    canvas = Image.new("RGB", (sum(i.width for i in panels) + 10, max(i.height for i in panels)), (20, 20, 20))
    canvas.paste(panels[0], (0, 0))
    canvas.paste(panels[1], (panels[0].width + 10, 0))
    scale = 1100 / max(canvas.size)
    if scale < 1:
        canvas = canvas.resize((int(canvas.width * scale), int(canvas.height * scale)))
    canvas.save(path)
    return path


SEAM_QUESTION = ("Left: green trace on the earlier slice. Right: red trace on the next traced slice. "
                 "Do they follow the same sheet, or does the red one sit on a neighbouring sheet or "
                 "extend onto other material? Reply as JSON: "
                 '{"same_sheet": true or false, "confidence": 0-1, "reason": "one sentence"}')


def review_seam(seed_id, tag):
    """Continuity where run `tag` joins the patch: numbers across the seam + Muse Spark's visual verdict."""
    from src.agents.spark import _img_url, call_spark
    runs = {t: (d, st) for t, d, st in _runs(seed_id)}
    if tag not in runs:
        return {"error": f"unknown run {tag}; runs: {list(runs)}"}
    d, status = runs[tag]
    new = {}
    for f in sorted(d.glob("[0-9]*.jsonl")):
        rec, pts = _curve(f)
        if len(pts) > 1 and status.get(rec["source_z"], "ok") in ("ok", "reprompted"):
            new[rec["source_z"]] = pts
    if not new:
        return {"error": f"run {tag} has no good slices"}
    z_new = min(new)
    others = {}
    for t, (dd, st) in runs.items():
        if t == tag or t in _excluded(seed_id):
            continue
        for f in sorted(dd.glob("[0-9]*.jsonl")):
            rec, pts = _curve(f)
            z = rec["source_z"]
            if len(pts) > 1 and z < z_new and st.get(z, "ok") in ("ok", "reprompted"):
                others.setdefault(z, pts)
    if not others:
        return {"error": "no earlier slice from another run to compare against"}
    z_prev = max(others)
    a, b = others[z_prev], new[z_new]
    d_ab = np.sqrt(((b[:, None, :] - a[None, :, :]) ** 2).sum(-1)).min(1)
    len_a = float(np.hypot(*np.diff(a, axis=0).T).sum())
    len_b = float(np.hypot(*np.diff(b, axis=0).T).sum())
    numbers = {"slice_before": z_prev, "slice_after": z_new, "gap_slices": z_new - z_prev,
               "median_distance_vx": round(float(np.median(d_ab)), 1),
               "share_within_3vx": round(float((d_ab <= 3).mean()), 3),
               "length_before_vx": round(len_a), "length_after_vx": round(len_b),
               "length_ratio": round(len_b / max(len_a, 1), 2)}
    img = _render_pair(seed_id, z_prev, a, z_new, b, AGENT / f"seam_{seed_id}_{tag}.png")
    text = call_spark([
        {"role": "system", "content": "You check whether two traces in consecutive X-ray CT slices of a carbonized "
                                      "papyrus scroll follow the same papyrus sheet. Bright bands are sheets. Reply JSON only."},
        {"role": "user", "content": [{"type": "text", "text": SEAM_QUESTION},
                                     {"type": "image_url", "image_url": {"url": _img_url(img)}}]}])
    try:
        verdict = json.loads(text[text.index("{"):text.rindex("}") + 1])
    except ValueError:
        verdict = {"same_sheet": None, "confidence": 0, "reason": "unparseable reply"}
    return {"run": tag, "numbers": numbers, "spark": verdict, "image": str(img.relative_to(ROOT))}


def exclude_run(seed_id, tag, reason):
    ex = _excluded(seed_id)
    ex[tag] = reason
    (FLAT / f"excluded_{seed_id}.json").write_text(json.dumps(ex, indent=1))
    return {"excluded": ex, "status_after": get_patch_status(seed_id)}


TOOLS = {"get_patch_status": get_patch_status, "propose_reseed": propose_reseed,
         "run_trace": run_trace, "score_patch": score_patch, "flatten_patch": flatten_patch,
         "review_seam": review_seam, "exclude_run": exclude_run}
STATE = {"max_runs": 3, "runs_used": 0}

SCHEMAS = [
    {"name": "get_patch_status", "description": "Traced and untraced slice ranges, failure reasons, curve lengths and remaining Kaggle runs for a seed's patch.",
     "parameters": {"type": "object", "properties": {"seed_id": {"type": "string"}}, "required": ["seed_id"]}},
    {"name": "propose_reseed", "description": "Compute a re-seed click at the middle of the good curve on slice z (use the last good slice before a gap).",
     "parameters": {"type": "object", "properties": {"seed_id": {"type": "string"}, "z": {"type": "integer"}}, "required": ["seed_id", "z"]}},
    {"name": "run_trace", "description": "Trace on a Kaggle T4 GPU from a click [x, y, z] (volume coordinates) through the remaining slices of the crop. Takes 10-15 minutes. Limited budget.",
     "parameters": {"type": "object", "properties": {"seed_id": {"type": "string"}, "click": {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3},
                                                     "window": {"type": "integer", "description": "slices to trace; 0 = to the end of the crop"}, "tag": {"type": "string"}},
                    "required": ["seed_id", "click"]}},
    {"name": "score_patch", "description": "Exact distance of the merged trace to the human-traced released mesh, on slices the mesh covers.",
     "parameters": {"type": "object", "properties": {"seed_id": {"type": "string"}}, "required": ["seed_id"]}},
    {"name": "flatten_patch", "description": "Unroll the merged trace into a surface image and report its size.",
     "parameters": {"type": "object", "properties": {"seed_id": {"type": "string"}}, "required": ["seed_id"]}},
    {"name": "review_seam", "description": "Check the seam where a re-seeded run joins the patch: curve distance and length ratio across the seam, plus Muse Spark's visual verdict on whether both slices show the same sheet.",
     "parameters": {"type": "object", "properties": {"seed_id": {"type": "string"}, "tag": {"type": "string"}}, "required": ["seed_id", "tag"]}},
    {"name": "exclude_run", "description": "Drop a run from the merged patch (for example it jumped to another sheet). Persisted to a file; reversible.",
     "parameters": {"type": "object", "properties": {"seed_id": {"type": "string"}, "tag": {"type": "string"}, "reason": {"type": "string"}}, "required": ["seed_id", "tag", "reason"]}},
    {"name": "finish", "description": "End the session with a short factual summary of what was done and measured.",
     "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}},
]

SYSTEM = (
    "You operate Unscroll, a tracer for papyrus sheets in X-ray CT of a Herculaneum scroll. "
    "Goal: extend the traced patch of one sheet across as many slices of its crop as possible "
    "while staying on the same sheet. Tracking can stop when SAM loses the sheet; re-seeding "
    "from the middle of the last good curve usually recovers it. Use tools to inspect, re-seed, "
    "run traces (each costs a limited Kaggle GPU run), score against the human-traced mesh where "
    "it exists, and unroll the result. Never claim a number you did not get from a tool. "
    "After every re-seeded run, call review_seam on it; exclude runs whose seam shows a different sheet "
    "(large distance across the seam, a sudden length change, or a negative visual verdict), and say why. "
    "Stop with finish when the crop is covered, when the budget is used, or when re-seeding stops helping.")


# ------------------------------------------------------------------ the loop
def run_agent(seed_id, max_steps=14, task=""):
    from openai import OpenAI
    from src.agents.spark import load_key
    client = OpenAI(base_url="https://api.meta.ai/v1", api_key=load_key(), timeout=300)
    AGENT.mkdir(parents=True, exist_ok=True)
    log = AGENT / f"{seed_id}_{datetime.now().strftime('%Y%m%dT%H%M%S')}.jsonl"
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": task or f"Extend the traced patch for seed {seed_id}."}]
    tools = [{"type": "function", "function": s} for s in SCHEMAS]

    def record(kind, payload):
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"t": datetime.now().isoformat(timespec="seconds"), "kind": kind, **payload},
                                ensure_ascii=False) + "\n")

    for step in range(max_steps):
        resp = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
        msg = resp.choices[0].message
        calls = msg.tool_calls or []
        record("model", {"content": msg.content, "tool_calls": [{"name": c.function.name, "arguments": c.function.arguments} for c in calls]})
        print(f"[{step}] wonder: {(msg.content or '').strip()[:160]}", flush=True)
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [{"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls]})
        if not calls:
            break
        for c in calls:
            args = json.loads(c.function.arguments or "{}")
            if c.function.name == "finish":
                record("finish", args)
                print("FINISH:", args.get("summary", ""), flush=True)
                return log
            fn = TOOLS.get(c.function.name)
            try:
                result = fn(**args) if fn else {"error": f"unknown tool {c.function.name}"}
            except Exception as e:  # noqa: BLE001  - surface tool errors to the model
                result = {"error": f"{type(e).__name__}: {e}"}
            record("tool", {"name": c.function.name, "arguments": args, "result": result})
            print(f"    {c.function.name}({args}) -> {json.dumps(result)[:220]}", flush=True)
            messages.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(result)})
    record("stop", {"reason": "step limit or no tool call"})
    return log


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-id", required=True)
    ap.add_argument("--max-runs", type=int, default=3)
    ap.add_argument("--status", action="store_true", help="print tool outputs without the model")
    ap.add_argument("--task", default="", help="instruction for the agent (default: extend the patch)")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # model text may contain ≤, —, µ
    STATE["max_runs"] = a.max_runs
    if a.status:
        print(json.dumps(get_patch_status(a.seed_id), indent=1))
        print(json.dumps(propose_reseed(a.seed_id, get_patch_status(a.seed_id)["last_good_slice"])))
        print(json.dumps(score_patch(a.seed_id)))
        return 0
    print("log:", run_agent(a.seed_id, task=a.task))
    return 0


if __name__ == "__main__":
    sys.exit(main())
