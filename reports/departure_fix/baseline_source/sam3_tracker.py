"""Local SAM 3.1 click tracker (point prompts, direct tracker propagation).

Same CLI + curve JSONL interface as sam_tracker.py (drop-in).
Needs gated weights: checkpoints/sam3.1_multiplex.pt (see README) or
HF cache auth (HF_HOME defaults to project .hf-cache).

Design notes (see scripts/sam3_debug*.py):
- multiplex handle_stream_request drops point-only objects after frame 0
  (detection-association scaffolding); we propagate via the tracker's native
  SAM2 video API instead (stable, verified on CT + synthetic controls).
- prompt = config C sweep winner: click + ridge-snapped tangent +/-60
  positives, tight normal negatives (+/-15/30/50).
- skeleton snapped jointly along the curve with hard continuity bounds.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external" / "sam3"))

import numpy as np  # noqa: E402

from src.eval.runlog import RunLog  # noqa: E402
from src.trace.guards import LayerGuard, load_cfg  # noqa: E402
from src.trace.sam_tracker import layer_normal, load_frame  # noqa: E402
from src.trace.thin import mask_to_curve  # noqa: E402
from src.trace.slices import select_slices  # noqa: E402
from src.trace.snap import continuous_snap  # noqa: E402


def build_prompt_points(gray, px, py):
    """Config-C prompt: snapped tangent positives + tight normal negatives.

    Returns (points_Nx2_normalized, labels) in frame pixels -> [0,1].
    """
    NH, NW = gray.shape
    nx, ny = layer_normal(gray, px, py)
    tx, ty = -ny, nx

    def snap(x, y, span=15):
        best, bv = (x, y), -1.0
        for s in range(-span, span + 1):
            xx, yy = int(round(x + s * nx)), int(round(y + s * ny))
            if 0 <= xx < NW and 0 <= yy < NH and gray[yy, xx] > bv:
                bv, best = gray[yy, xx], (xx, yy)
        return best

    pts, labs = [[px / NW, py / NH]], [1]
    for s in (60, -60):
        x, y = snap(px + s * tx, py + s * ty)
        pts.append([x / NW, y / NH])
        labs.append(1)
    for s in (15, 30, 50, -15, -30, -50):
        pts.append([(px + s * nx) / NW, (py + s * ny) / NH])
        labs.append(0)
    return pts, labs


def ridge_snap(curve, gray, span=10):
    """Joint brightness snapping with hard continuity/forward-progress bounds."""
    return continuous_snap(curve, gray, span=span)


def fullres_ridge_snap(curve, tif_path, span=12, k=1.5):
    """Joint ridge snapping; unsupported gaps retain a continuous bridge."""
    import tifffile
    if len(curve) < 3:
        return np.asarray(curve, dtype=float).reshape(-1, 2).copy()
    img = tifffile.imread(tif_path).astype(float)
    thr = np.median(img) + k * img.std()
    rm = ((img > thr) & (img >= np.roll(img, 1, 1))
          & (img >= np.roll(img, -1, 1)))
    rm[:, [0, -1]] = False
    return continuous_snap(curve, rm, span=span, ridge=True)


def postprocess_mask(mask, gray, transform, tif_path, seed=None, previous=None):
    """Shared live/replay path; previous is the unsnapped video-space curve."""
    xs, ys, ox, oy = transform
    curve_small = mask_to_curve(mask, seed=seed)
    if previous is not None and len(curve_small):
        forward = np.linalg.norm(curve_small[0] - previous[0]) + np.linalg.norm(curve_small[-1] - previous[-1])
        reverse = np.linalg.norm(curve_small[-1] - previous[0]) + np.linalg.norm(curve_small[0] - previous[-1])
        if reverse < forward:
            curve_small = curve_small[::-1].copy()
    raw = curve_small.copy()
    if len(curve_small):
        snapped = ridge_snap(curve_small, gray.astype(np.float32), span=10)
        curve = snapped * np.array([xs, ys]) + np.array([ox, oy])
        curve = fullres_ridge_snap(curve, tif_path)
    else:
        curve = curve_small
    return curve, raw


def extract_mask(outputs, frame_hw):
    """Defensive mask extraction across predictor output shapes."""
    import torch
    from PIL import Image
    h, w = frame_hw
    if isinstance(outputs, dict) and "out_binary_masks" in outputs:
        m = outputs["out_binary_masks"]
        m = m.detach().cpu().numpy() if isinstance(m, torch.Tensor) else np.asarray(m)
        if m.shape[0] == 0:
            return np.zeros((h, w), dtype=bool)
        a = m[0].astype(np.float32)
        if a.shape != (h, w):
            a = np.asarray(Image.fromarray(a).resize((w, h), Image.NEAREST))
        return a > 0.5
    cands = []
    if isinstance(outputs, dict):
        for k in ("masks", "mask", "pred_masks"):
            if k in outputs:
                cands.append(outputs[k])
        cands.extend(outputs.values())
    elif isinstance(outputs, (list, tuple)):
        cands = list(outputs)
    else:
        cands = [outputs]
    for c in cands:
        if isinstance(c, dict):
            for k in ("mask", "masks"):
                if k in c:
                    c = c[k]
                    break
        if isinstance(c, torch.Tensor):
            a = c.detach().cpu().numpy()
        elif isinstance(c, np.ndarray):
            a = c
        else:
            continue
        while a.ndim > 2:
            a = a[0]
        if a.shape == (h, w):
            if a.max() <= 1.0 and a.min() >= 0.0:
                return a > 0.5
            return a > 0.0
    raise ValueError(f"no (H,W) mask in outputs: {type(outputs)}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", required=True)
    ap.add_argument("--click", required=True)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", type=int, default=2048)
    ap.add_argument("--z0", type=int, default=0,
                    help="physical z of first TIFF for nonnumeric filenames")
    ap.add_argument("--save-masks", action="store_true",
                    help="save pre-postprocessing masks for reproducible validation")
    ap.add_argument("--ckpt", default="",
                    help="local sam3.1_multiplex.pt (else HF cache)")
    a = ap.parse_args(argv)

    import torch
    from PIL import Image
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    cx, cy, cz = (int(v) for v in a.click.split(","))
    refs = select_slices(a.slices, cz, a.window, a.z0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runlog = RunLog(ROOT, note=f"sam3-local {a.click}")
    cfg = load_cfg(ROOT / "src" / "trace" / "guards.yaml")
    guard = LayerGuard(cfg)

    kw = {"checkpoint_path": a.ckpt} if a.ckpt else {}
    kw["use_fa3"] = False  # no flash_attn wheels on Windows
    kw["async_loading_frames"] = False
    try:
        predictor = build_sam3_multiplex_video_predictor(**kw)
    except TypeError:
        predictor = build_sam3_multiplex_video_predictor()
    with torch.inference_mode(), torch.autocast(device, dtype=dtype):
        fdir = out / "frames3"
        if fdir.exists() and any(fdir.iterdir()):
            raise ValueError("frame directory is not empty; use a fresh --out")
        fdir.mkdir(parents=True, exist_ok=True)
        crop = (cx, cy, a.crop) if a.crop > 0 else None
        meta, NW, NH = None, 0, 0
        for i, ref in enumerate(refs):
            rgb, meta = load_frame(ref.path, crop=crop)
            Image.fromarray(rgb).save(fdir / f"{i:05d}.jpg")
            NH, NW = rgb.shape[:2]
        xs, ys, ox, oy = meta
        r = predictor.handle_request(dict(type="start_session",
                                          resource_path=str(fdir)))
        sid = r["session_id"]
        first = np.array(Image.open(fdir / "00000.jpg"))[:, :, 0]
        px, py = (cx - ox) / xs, (cy - oy) / ys
        pts, labs = build_prompt_points(first.astype(np.float32), px, py)
        predictor.handle_request(dict(
            type="add_prompt", session_id=sid, frame_index=0,
            points=torch.tensor(pts, dtype=torch.float32),
            point_labels=torch.tensor(labs, dtype=torch.int32),
            obj_id=0))
        model = predictor.model
        state = predictor._all_inference_states[sid]["state"]
        sam2_state = state["sam2_inference_states"][0]
        model.tracker.propagate_in_video_preflight(sam2_state,
                                                   run_mem_encoder=True)
        previous = None
        for z, ref in enumerate(refs):
            model._prepare_backbone_feats(state, z, False)
            got = None
            for res in model.tracker.propagate_in_video(
                    sam2_state, start_frame_idx=z, max_frame_num_to_track=0,
                    reverse=False, tqdm_disable=True, run_mem_encoder=True):
                got = res
            _fz, oids, _low, vres, _scores = got
            gray = np.array(Image.open(fdir / f"{z:05d}.jpg"))[:, :, 0]
            if len(oids):
                m = vres[0, 0].detach().cpu().numpy()
                if m.shape != (NH, NW):
                    m = np.asarray(Image.fromarray(m).resize((NW, NH)))
                mask = m > 0.0
            else:
                mask = np.zeros((NH, NW), dtype=bool)
            if a.save_masks:
                np.savez_compressed(out / f"mask_{ref.index:05d}.npz",
                                    mask=mask, transform=np.array([xs, ys, ox, oy]))
            seed = (px, py) if previous is None else previous.mean(axis=0)
            curve, raw = postprocess_mask(mask, gray, (xs, ys, ox, oy),
                                          ref.path, seed=seed, previous=previous)
            if len(raw):
                previous = raw
            rec = {**ref.metadata(),
                   "points": [[round(float(x), 1), round(float(y), 1)]
                              for x, y in curve],
                   "origin": ([round(float(curve[0][0]), 1),
                               round(float(curve[0][1]), 1)]
                              if len(curve) else None),
                   "shift": [0, 0]}
            (out / f"{ref.index:05d}.jsonl").write_text(json.dumps(rec) + "\n")
            assessment = guard.evaluate(np.asarray(rec["points"]).reshape(-1, 2), ref.source_z)
            status = assessment["status"]
            checks = {"winding": assessment["winding"]["pass"],
                      "kink": assessment["kink"]["pass"], "details": assessment}
            runlog.log_slice(ref.index, status, f"{ref.index:05d}.jsonl", checks, 0)
            print(f"z={ref.source_z} pts={len(curve)} {status}", flush=True)
    print(f"run {runlog.run_id}: curves in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
