"""Click-to-trace: slice stack as video (SAM 2.1 video predictor).

  python -m src.trace.sam_tracker --slices DIR --click X,Y,Z \\
      --window 32 --out curves/ --model small|large

Frames: 16-bit tif -> uint8 RGB (replicated gray). 32-slice chunks with
8-slice overlap; curve JSONL per slice {z, points, origin, shift}.
Needs venv: torch (CUDA), sam2 (external/sam2 on sys.path), tifffile.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "external" / "sam2"))

import numpy as np  # noqa: E402

from src.eval.runlog import RunLog  # noqa: E402
from src.trace.guards import (check_curvature, check_spacing,  # noqa: E402
                                check_winding, load_cfg)
from src.trace.thin import mask_to_curve  # noqa: E402
from src.trace.slices import select_slices  # noqa: E402

MODELS = {
    "small": "facebook/sam2-hiera-small",
    "large": "facebook/sam2-hiera-large",
}
CHUNK, OVERLAP = 32, 8


def layer_normal(gray, px, py, win=32):
    """Layer normal via structure tensor (eigenvector, top eigenvalue)."""
    h, w = gray.shape
    x0, x1 = max(0, int(px) - win), min(w, int(px) + win)
    y0, y1 = max(0, int(py) - win), min(h, int(py) + win)
    patch = gray[y0:y1, x0:x1].astype(float)
    gy, gx = np.gradient(patch)
    jxx, jyy, jxy = (gx * gx).mean(), (gy * gy).mean(), (gx * gy).mean()
    tr, det = jxx + jyy, jxx * jyy - jxy * jxy
    lmax = tr / 2 + np.sqrt(max(tr * tr / 4 - det, 0))
    if abs(jxy) < 1e-9:
        return (1.0, 0.0) if jxx >= jyy else (0.0, 1.0)
    vx, vy = lmax - jyy, jxy
    n = (vx * vx + vy * vy) ** 0.5 or 1.0
    return vx / n, vy / n


def load_frame(path, size=1024, crop=None):
    """crop = (cx, cy, box): full-res square before downscale."""
    import tifffile
    img = tifffile.imread(path).astype(np.float32)
    ox, oy = 0, 0
    if crop is not None:
        cx, cy, box = crop
        h0, w0 = img.shape
        ox = max(0, min(w0 - box, cx - box // 2))
        oy = max(0, min(h0 - box, cy - box // 2))
        img = img[oy:oy + box, ox:ox + box]
    lo, hi = np.percentile(img, (1, 99))
    u8 = np.clip((img - lo) / max(hi - lo, 1e-6) * 255, 0, 255)
    u8 = u8.astype(np.uint8)
    h, w = u8.shape
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    ys = (np.arange(nh) / scale).astype(int)
    xs = (np.arange(nw) / scale).astype(int)
    small = u8[ys[:, None], xs]
    rgb = np.stack([small] * 3, -1)
    return rgb, (w / nw, h / nh, ox, oy)  # xscale, yscale, offsets


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", required=True, help="dir of sorted *.tif")
    ap.add_argument("--click", required=True, help="X,Y,Z in full-res px")
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="small", choices=tuple(MODELS))
    ap.add_argument("--crop", type=int, default=2048,
                    help="full-res crop box around click (0 = full frame)")
    ap.add_argument("--z0", type=int, default=0,
                    help="physical z of first TIFF for nonnumeric filenames")
    a = ap.parse_args(argv)

    import torch
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARN: no CUDA, CPU fallback (plumbing only, slow)")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    cx, cy, cz = (int(v) for v in a.click.split(","))
    refs = select_slices(a.slices, cz, a.window, a.z0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runlog = RunLog(ROOT, note=f"track {a.click} {a.model}")
    cfg = load_cfg(ROOT / "src" / "trace" / "guards.yaml")

    from PIL import Image
    predictor = SAM2VideoPredictor.from_pretrained(MODELS[a.model])
    winding, gaps = [], []
    prev_curve = None
    with torch.inference_mode(), torch.autocast(device, dtype=dtype):
        for c0 in range(0, len(refs), CHUNK - OVERLAP):
            chunk = refs[c0:c0 + CHUNK]
            fdir = out / f"chunk_{c0:04d}_frames"
            if fdir.exists() and any(fdir.iterdir()):
                raise ValueError("frame directory is not empty; use a fresh --out")
            fdir.mkdir(parents=True, exist_ok=True)
            meta = None
            crop = (cx, cy, a.crop) if a.crop > 0 else None
            for i, ref in enumerate(chunk):
                rgb, meta = load_frame(ref.path, crop=crop)
                Image.fromarray(rgb).save(fdir / f"{i:05d}.jpg")
            xs, ys, ox, oy = meta
            state = predictor.init_state(str(fdir))
            px, py = (cx - ox) / xs, (cy - oy) / ys
            # negatives along the layer NORMAL only (toward adjacent
            # layers), leaving the tangent free so the mask extends
            # along the layer. Local-only; design delta recorded.
            first = np.array(Image.open(fdir / "00000.jpg"))[:, :, 0]
            nx, ny = layer_normal(first, px, py)
            negs = [(px + s * nx, py + s * ny)
                    for s in (30, 60, 90, -30, -60, -90)]
            pts = np.array([[px, py]] + negs)
            lab = np.array([1] + [0] * len(negs))
            _, _, _ = predictor.add_new_points_or_box(
                state, frame_idx=0, obj_id=0, points=pts, labels=lab)
            for fi, _oids, masks in predictor.propagate_in_video(state):
                frame = c0 + fi
                if frame >= len(refs):
                    break
                ref = refs[frame]
                z = ref.index
                m = masks[0][0].cpu().numpy() > 0.0
                curve_small = mask_to_curve(m, seed=(px, py))
                curve = (curve_small * np.array([xs, ys]) +
                         np.array([ox, oy])) if len(
                    curve_small) else curve_small
                rec = {**ref.metadata(),
                       "points": [[round(float(x), 1), round(float(y), 1)]
                                  for x, y in curve],
                       "origin": ([round(float(curve[0][0]), 1),
                                   round(float(curve[0][1]), 1)]
                                  if len(curve) else None),
                       "shift": [0, 0]}
                (out / f"{z:05d}.jsonl").write_text(json.dumps(rec) + "\n")
                status, checks, nrep = "ok", {}, 0
                if len(curve) >= 3:
                    ang = np.unwrap(np.arctan2(
                        np.diff(curve[:, 1]), np.diff(curve[:, 0])))
                    w = check_winding(np.degrees(ang).tolist(),
                                      cfg["winding_reversal_deg"])
                    k = check_curvature(curve, cfg["kink_deg"])
                    checks = {"winding": w["pass"], "kink": k["pass"]}
                    if not (w["pass"] and k["pass"]):
                        status = "suspect"
                else:
                    status, checks = "failed", {"empty": True}
                runlog.log_slice(z, status, f"{z:05d}.jsonl", checks, nrep)
                prev_curve = curve
                print(f"z={z} pts={len(curve)} {status}", flush=True)
    print(f"run {runlog.run_id}: curves in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
