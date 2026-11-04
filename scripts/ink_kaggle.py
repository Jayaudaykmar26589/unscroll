"""2.5D ink detection on the Vesuvius 2023 Kaggle fragments, run on a Kaggle GPU.

  python scripts/ink_kaggle.py --train 1 2 --val 3 --tag f12      (on Kaggle)
  python scripts/ink_kaggle.py --train 1 --val 3 --tag f1         (baseline-comparable)

Input: the middle 16 of 65 surface-volume layers as channels. Model: U-Net with
an ImageNet-pretrained ResNet-34 encoder (timm). Balanced random 224 px patches
inside the fragment mask, flips/rot90, AdamW, BCE + Dice, fp16. Validation on a
held-out fragment with full sliding-window prediction: AUC, best F0.5 (the
competition metric) and F1. Then the model is applied, unchanged, to the
unrolled Scroll 1 surface (reports/flatten/surface65_*/surface.npy). Writes
reports/ink_kaggle/<tag>/{metrics.json, val_pred.png, scroll_pred.png, model.pt}.
"""
import argparse
import glob
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LAYERS = list(range(24, 40))   # middle 16 of 65
PATCH, STRIDE = 224, 112
Image.MAX_IMAGE_PIXELS = None


def frag_dir(fid):
    hits = glob.glob(f"/kaggle/input/**/train/{fid}/surface_volume", recursive=True)
    if not hits:
        raise SystemExit(f"fragment {fid} not found: accept the competition rules and attach "
                         "vesuvius-challenge-ink-detection to the notebook")
    return Path(hits[0]).parent


def load_fragment(fid):
    import tifffile
    d = frag_dir(fid)
    vol = np.stack([tifffile.imread(d / "surface_volume" / f"{z:02d}.tif") for z in LAYERS])  # uint16 C,H,W
    ink = np.array(Image.open(d / "inklabels.png").convert("L")) > 0
    mask = np.array(Image.open(d / "mask.png").convert("L")) > 0
    return vol, ink, mask


class UNet(nn.Module):
    def __init__(self, in_ch=len(LAYERS)):
        super().__init__()
        import timm
        self.enc = timm.create_model("resnet34", features_only=True, pretrained=True, in_chans=in_ch)
        chs = self.enc.feature_info.channels()          # [64, 64, 128, 256, 512]
        self.dec = nn.ModuleList()
        prev = chs[-1]
        for skip in reversed(chs[:-1]):
            self.dec.append(nn.Sequential(nn.Conv2d(prev + skip, skip, 3, padding=1), nn.BatchNorm2d(skip), nn.ReLU(inplace=True),
                                          nn.Conv2d(skip, skip, 3, padding=1), nn.BatchNorm2d(skip), nn.ReLU(inplace=True)))
            prev = skip
        self.head = nn.Conv2d(prev, 1, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        feats = self.enc(x)
        y = feats[-1]
        for blk, skip in zip(self.dec, reversed(feats[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = blk(torch.cat([y, skip], 1))
        return F.interpolate(self.head(y), size=(h, w), mode="bilinear", align_corners=False)


def norm(x):
    x = x.astype(np.float32) / 65535.0
    m, s = x.mean(), x.std() + 1e-6
    return (x - m) / s


def sample(vol, ink, mask, rng, pos_frac=0.5):
    _, H, W = vol.shape
    want_ink = rng.random() < pos_frac
    for _ in range(50):
        y, x = rng.randrange(0, H - PATCH), rng.randrange(0, W - PATCH)
        c = mask[y + PATCH // 2, x + PATCH // 2]
        if not c:
            continue
        has = ink[y:y + PATCH, x:x + PATCH].mean() > 0.02
        if has == want_ink:
            break
    a = norm(vol[:, y:y + PATCH, x:x + PATCH])
    t = ink[y:y + PATCH, x:x + PATCH].astype(np.float32)
    k = rng.randrange(4)
    a, t = np.rot90(a, k, (1, 2)), np.rot90(t, k)
    if rng.random() < 0.5:
        a, t = a[:, :, ::-1], t[:, ::-1]
    a = a * (0.8 + 0.4 * rng.random()) + (rng.random() - 0.5) * 0.2
    return np.ascontiguousarray(a), np.ascontiguousarray(t)


def dice(logits, t, eps=1.0):
    p = torch.sigmoid(logits)
    return 1 - (2 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps)


@torch.no_grad()
def predict(model, vol, mask, dev, batch=16):
    _, H, W = vol.shape
    acc = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    coords = [(y, x) for y in range(0, max(H - PATCH, 0) + 1, STRIDE) for x in range(0, max(W - PATCH, 0) + 1, STRIDE)
              if mask[y:y + PATCH, x:x + PATCH].any()]
    for i in range(0, len(coords), batch):
        cs = coords[i:i + batch]
        xb = torch.from_numpy(np.stack([norm(vol[:, y:y + PATCH, x:x + PATCH]) for y, x in cs])).to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            pb = torch.sigmoid(model(xb)).float().cpu().numpy()[:, 0]
        for (y, x), p in zip(cs, pb):
            acc[y:y + PATCH, x:x + PATCH] += p
            cnt[y:y + PATCH, x:x + PATCH] += 1
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), 0)


def metrics(pred, ink, mask, rng):
    p, t = pred[mask], ink[mask]
    idx = rng.choice(len(p), size=min(len(p), 2_000_000), replace=False)
    p, t = p[idx], t[idx]
    order = np.argsort(-p)
    tp = np.cumsum(t[order])
    fp = np.cumsum(~t[order])
    P = t.sum()
    trapezoid = getattr(np, "trapezoid", None) or np.trapz     # NumPy 2 renamed trapz
    auc = float(trapezoid(tp / max(P, 1), fp / max((~t).sum(), 1)))
    best = {"f05": 0.0}
    for thr in np.linspace(0.05, 0.95, 19):
        y = p >= thr
        tp_, fp_, fn_ = (y & t).sum(), (y & ~t).sum(), (~y & t).sum()
        prec, rec = tp_ / max(tp_ + fp_, 1), tp_ / max(tp_ + fn_, 1)
        f05 = 1.25 * prec * rec / max(0.25 * prec + rec, 1e-9)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f05 > best["f05"]:
            best = {"f05": float(f05), "f1_at_best_f05": float(f1), "threshold": float(thr),
                    "precision": float(prec), "recall": float(rec)}
    best_f1 = 0.0
    for thr in np.linspace(0.05, 0.95, 19):
        y = p >= thr
        tp_, fp_, fn_ = (y & t).sum(), (y & ~t).sum(), (~y & t).sum()
        best_f1 = max(best_f1, 2 * tp_ / max(2 * tp_ + fp_ + fn_, 1))
    return {"auc": auc, "best_f1": float(best_f1), **best, "pixels_scored": int(len(p))}


def scroll_surface(model, dev, out):
    hits = sorted(ROOT.glob("reports/flatten/surface65_*/surface.npy"))
    res = {}
    for h in hits:
        v = np.load(h)                                   # rows, cols, 65 (NaN outside the trace)
        rows = np.isfinite(v).any(axis=(1, 2))
        v = v[rows]
        fill = np.nanmedian(v)
        v = np.nan_to_num(v, nan=fill)
        x = np.transpose(v[:, :, LAYERS], (2, 0, 1)).astype(np.float32)   # C, rows, cols
        H, W = x.shape[1:]
        ph, pw = max(PATCH - H, 0), max(PATCH - W, 0)
        x = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="reflect")
        xs = (x - x.mean()) / (x.std() + 1e-6)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            # tile across columns if wider than a patch
            preds = np.zeros(x.shape[1:], np.float32)
            cnt = np.zeros_like(preds)
            for c0 in range(0, max(x.shape[2] - PATCH, 0) + 1, STRIDE):
                for r0 in range(0, max(x.shape[1] - PATCH, 0) + 1, STRIDE):
                    t = torch.from_numpy(xs[None, :, r0:r0 + PATCH, c0:c0 + PATCH]).to(dev)
                    preds[r0:r0 + PATCH, c0:c0 + PATCH] += torch.sigmoid(model(t)).float().cpu().numpy()[0, 0]
                    cnt[r0:r0 + PATCH, c0:c0 + PATCH] += 1
        p = (preds / np.maximum(cnt, 1))[:H, :W]
        name = h.parent.name
        Image.fromarray((np.clip(p, 0, 1) * 255).astype(np.uint8)).save(out / f"scroll_pred_{name}.png")
        res[name] = {"shape": [int(H), int(W)], "max": float(p.max()), "mean": float(p.mean()),
                     "frac_above_0.5": float((p > 0.5).mean())}
    return res


CONFIGS = {"f1": (["1"], "3"), "f12": (["1", "2"], "3")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", default=["1", "2"])
    ap.add_argument("--val", default="3")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tag", default="f12", help="output name; 'all' runs every entry of CONFIGS")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.tag == "all":
        for tag, (tr, va) in CONFIGS.items():
            a.tag, a.train, a.val = tag, tr, va
            run(a)
            torch.cuda.empty_cache()
        return
    run(a)


def run(a):
    out = ROOT / "reports" / "ink_kaggle" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    np_rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    dev = "cuda"
    t0 = time.time()
    train = [load_fragment(f) for f in a.train]
    print(f"loaded train {a.train} in {time.time() - t0:.0f}s", flush=True)
    model = UNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps)
    scaler = torch.cuda.amp.GradScaler()
    weights = [v.shape[1] * v.shape[2] for v, _, _ in train]
    model.train()
    for step in range(a.steps):
        xb, yb = [], []
        for _ in range(a.batch):
            v, ink, mask = rng.choices(train, weights=weights)[0]
            x, y = sample(v, ink, mask, rng)
            xb.append(x)
            yb.append(y)
        xb = torch.from_numpy(np.stack(xb)).to(dev)
        yb = torch.from_numpy(np.stack(yb))[:, None].to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb) + dice(logits.float(), yb)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % 250 == 0:
            print(f"step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    del train
    model.eval()
    v, ink, mask = load_fragment(a.val)
    pred = predict(model, v, mask, dev)
    m = metrics(pred, ink, mask, np_rng)
    small = Image.fromarray((np.clip(pred, 0, 1) * 255).astype(np.uint8)).reduce(4)
    small.save(out / "val_pred.png")
    Image.fromarray((ink * 255).astype(np.uint8)).reduce(4).save(out / "val_label.png")
    torch.save(model.state_dict(), out / "model.pt")
    result = {"tag": a.tag, "train": a.train, "val": a.val, "layers": [LAYERS[0], LAYERS[-1]],
              "steps": a.steps, "batch": a.batch, "patch": PATCH, "val_metrics": m,
              "scroll": scroll_surface(model, dev, out), "minutes": round((time.time() - t0) / 60, 1)}
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
