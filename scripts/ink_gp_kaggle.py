"""Grand Prize (2023) TimeSformer ink model on Kaggle: validate on known ink, then run on our surface.

  python scripts/ink_gp_kaggle.py        (on a Kaggle GPU notebook with internet)

Model: scrollprize/ink_detection_pipeline model.safetensors (MIT), loaded into
timesformer-pytorch TimeSformer(dim 512, image 64, patch 16, frames 30,
classes 16, depth 8, heads 6, dim_head 64); every weight must match exactly.
Preprocessing as in the winners' inference script: surface-volume layers 17-42
(26), 8-bit, clipped to 0-200, /255; 64 px tiles, stride 21, 4x4 output per
tile upsampled x16, Gaussian-weighted blending, sigmoid.

1. Validation: released Scroll 1 segment 20230827161847, layers 17-42 streamed
   from dl.ash2txt.org (cropped to the region of the official example).
2. Ours: reports/flatten/surface65_*/surface.npy (layer 32 = Unscroll's trace).
Writes reports/ink_gp/{summary.json, val_pred.png, val_ct.png, ours_*.png}.

  python scripts/ink_gp_kaggle.py --full   whole segment (every column on it) at
  full resolution -> reports/ink_gp_full/{pred_full.png, pred_q4.png, ct_q4.png, summary.json}
"""
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "ink_gp"
LAYERS = list(range(17, 43))
TILE, STRIDE = 64, 21
SEGMENT = "20230827161847"
SEG_URL = "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/paths/{seg}/layers/{z:02d}.tif"
CROP = (slice(200, 5600), slice(1000, 4600))     # rows, cols of the official example
Image.MAX_IMAGE_PIXELS = None


def load_model(dev):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from timesformer_pytorch import TimeSformer
    net = TimeSformer(dim=512, image_size=64, patch_size=16, num_frames=30, num_classes=16,
                      channels=1, depth=8, heads=6, dim_head=64, attn_dropout=0.1, ff_dropout=0.1)
    sd = load_file(hf_hub_download("scrollprize/ink_detection_pipeline", "model.safetensors"))
    want = set(net.state_dict())
    for prefix in ("", "model.", "backbone.", "model.backbone.", "timesformer.", "model.model."):
        cand = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if want <= set(cand):
            net.load_state_dict({k: cand[k] for k in want}, strict=True)
            print(f"weights loaded with prefix '{prefix}' ({len(want)} tensors)", flush=True)
            return net.to(dev).eval()
    raise SystemExit("weight names do not match TimeSformer; first keys: " + ", ".join(list(sd)[:12]))


def gkern(n=TILE, sig=1.0):
    ax = np.linspace(-(n - 1) / 2.0, (n - 1) / 2.0, n) / (n / 2.0) * 3
    g = np.exp(-0.5 * (ax / sig) ** 2)
    k = np.outer(g, g)
    return (k / k.max()).astype(np.float32)


def to_input(stack8):
    """uint8 (26, H, W) -> float32 clipped/scaled as in training."""
    return np.clip(stack8, 0, 200).astype(np.float32) / 255.0


@torch.no_grad()
def predict(net, x, dev, valid=None, batch=256):
    """x: float32 (26, H, W). Returns ink probability (H, W)."""
    _, H, W = x.shape
    pad_h, pad_w = max(TILE - H, 0), max(TILE - W, 0)
    if pad_h or pad_w:
        x = np.pad(x, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    _, Hp, Wp = x.shape
    acc = np.zeros((Hp, Wp), np.float32)
    cnt = np.zeros((Hp, Wp), np.float32)
    k = gkern()
    ys = list(range(0, Hp - TILE + 1, STRIDE)) or [0]
    xs = list(range(0, Wp - TILE + 1, STRIDE)) or [0]
    if ys[-1] != Hp - TILE:
        ys.append(Hp - TILE)
    if xs[-1] != Wp - TILE:
        xs.append(Wp - TILE)
    coords = [(y, c) for y in ys for c in xs if valid is None or valid[y:y + TILE, c:c + TILE].all()]
    for i in range(0, len(coords), batch):
        cs = coords[i:i + batch]
        t = torch.from_numpy(np.stack([x[:, y:y + TILE, c:c + TILE] for y, c in cs])).to(dev)
        t = t[:, :, None]                                   # B, frames, 1, 64, 64
        with torch.autocast("cuda", dtype=torch.float16):
            o = net(t).float().view(-1, 1, 4, 4)
        o = torch.sigmoid(F.interpolate(o, scale_factor=16, mode="bilinear", align_corners=False))
        o = o[:, 0].cpu().numpy()
        for (y, c), p in zip(cs, o):
            acc[y:y + TILE, c:c + TILE] += p * k
            cnt[y:y + TILE, c:c + TILE] += k
    return (acc / np.maximum(cnt, 1e-6))[:H, :W]


def fetch_layer(z):
    import tifffile
    for attempt in range(4):
        try:
            with urllib.request.urlopen(SEG_URL.format(seg=SEGMENT, z=z), timeout=600) as r:
                data = r.read()
            img = tifffile.imread(io.BytesIO(data))
            break
        except Exception as e:  # noqa: BLE001
            print(f"layer {z} attempt {attempt + 1} failed: {e}", flush=True)
            time.sleep(10)
    else:
        raise SystemExit(f"could not fetch layer {z}")
    img = img[CROP]
    if img.dtype == np.uint16:
        img = (img // 257).astype(np.uint8)                 # what cv2 IMREAD_GRAYSCALE does to 16-bit
    return img


def png(a, path, lo=None, hi=None):
    lo = np.nanpercentile(a, 1) if lo is None else lo
    hi = np.nanpercentile(a, 99) if hi is None else hi
    Image.fromarray(np.clip((np.nan_to_num(a, nan=lo) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)).save(path)


def full_page():
    """GP ink on the whole released segment: the full page, not a window."""
    global CROP
    out = ROOT / "reports" / "ink_gp_full"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    net = load_model("cuda")
    CROP = (slice(None), slice(None))
    stack = np.stack([fetch_layer(z) for z in LAYERS])
    print(f"full stack {stack.shape} in {time.time() - t0:.0f}s", flush=True)
    valid = stack.max(axis=0) > 0
    pred = predict(net, to_input(stack), "cuda", valid=valid)
    pred[~valid] = 0
    im = Image.fromarray((np.clip(pred, 0, 1) * 255).astype(np.uint8))
    im.save(out / "pred_full.png", optimize=True)
    im.reduce(4).save(out / "pred_q4.png")
    png(stack[len(LAYERS) // 2].astype(np.float32), out / "ct_q4.png")
    Image.open(out / "ct_q4.png").reduce(4).save(out / "ct_q4.png")
    S = {"model": "scrollprize/ink_detection_pipeline (Grand Prize 2023 TimeSformer, MIT)",
         "segment": SEGMENT, "layers": [LAYERS[0], LAYERS[-1]], "tile": TILE, "stride": STRIDE,
         "shape": list(pred.shape), "valid_fraction": float(valid.mean()),
         "mean": float(pred[valid].mean()), "frac_above_0.5": float((pred[valid] > 0.5).mean()),
         "minutes": round((time.time() - t0) / 60, 1)}
    (out / "summary.json").write_text(json.dumps(S, indent=2))
    print(json.dumps(S, indent=2), flush=True)


def main():
    if "--full" in sys.argv:
        return full_page()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    t0 = time.time()
    net = load_model(dev)
    summary = {"model": "scrollprize/ink_detection_pipeline (Grand Prize 2023 TimeSformer, MIT)",
               "layers": [LAYERS[0], LAYERS[-1]], "tile": TILE, "stride": STRIDE}

    # 1. validation on a released segment with known letters
    stack = np.stack([fetch_layer(z) for z in LAYERS])      # uint8 26, H, W
    print(f"validation stack {stack.shape} in {time.time() - t0:.0f}s", flush=True)
    valid = stack.max(axis=0) > 0
    pred = predict(net, to_input(stack), dev, valid=valid)
    Image.fromarray((np.clip(pred, 0, 1) * 255).astype(np.uint8)).reduce(4).save(OUT / "val_pred.png")
    png(stack[len(LAYERS) // 2].astype(np.float32), OUT / "val_ct.png")
    Image.open(OUT / "val_ct.png").reduce(4).save(OUT / "val_ct.png")
    summary["validation"] = {"segment": SEGMENT, "crop_rows_cols": [[CROP[0].start, CROP[0].stop], [CROP[1].start, CROP[1].stop]],
                             "shape": list(pred.shape), "mean": float(pred[valid].mean()),
                             "p99": float(np.percentile(pred[valid], 99)), "frac_above_0.5": float((pred[valid] > 0.5).mean())}
    del stack

    # 2. Unscroll's own traced surface (same scroll, same voxel size)
    summary["ours"] = {}
    for f in sorted(ROOT.glob("reports/flatten/surface65_*/surface.npy")):
        v = np.load(f)                                       # rows, cols, 65 (uint16 values, NaN outside)
        rows = np.isfinite(v).any(axis=(1, 2))
        v = v[rows]
        finite = np.isfinite(v).all(axis=2)
        fill = np.nanmedian(v)
        v8 = (np.nan_to_num(v, nan=fill) // 257).clip(0, 255).astype(np.uint8)
        x = to_input(np.transpose(v8[:, :, LAYERS], (2, 0, 1)))
        p = predict(net, x, dev)
        p[~finite] = np.nan
        name = f.parent.name
        Image.fromarray((np.clip(np.nan_to_num(p), 0, 1) * 255).astype(np.uint8)).save(OUT / f"ours_{name}_pred.png")
        png(v8[:, :, 32].astype(np.float32), OUT / f"ours_{name}_ct.png")
        summary["ours"][name] = {"shape": list(p.shape), "mean": float(np.nanmean(p)),
                                 "p99": float(np.nanpercentile(p, 99)), "frac_above_0.5": float(np.nanmean(p > 0.5))}
    summary["minutes"] = round((time.time() - t0) / 60, 1)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
