"""Train Ink3D on frag1, validate on frag3 (G3 = F1 + AUC).

  python -m src.ink.train --epochs 6 --batch 8 --n-train 2000 --n-val 400

fp16 autocast, AdamW, BCE+Dice. Writes reports/ink_train.json + best.pt.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.ink.dataset import InkPatches  # noqa: E402
from src.ink.model import Ink3D, dice_loss  # noqa: E402


def f1_auc(plog, t):
    p = torch.sigmoid(plog).cpu().numpy().ravel()
    t = t.cpu().numpy().ravel()
    pred = p > 0.5
    tp = float(((pred == 1) & (t == 1)).sum())
    fp = float(((pred == 1) & (t == 0)).sum())
    fn = float(((pred == 0) & (t == 1)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
    order = np.argsort(p)
    t_sorted = t[order]
    tp_c = np.cumsum(t_sorted[::-1])
    fp_c = np.cumsum((1 - t_sorted)[::-1])
    auc = float(np.trapz(tp_c, fp_c) / max(tp_c[-1] * fp_c[-1], 1e-9))
    return f1, auc


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="reports/ink")
    a = ap.parse_args(argv)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = DataLoader(InkPatches(ROOT / "data/cache/frag1", a.n_train),
                       batch_size=a.batch, shuffle=True, num_workers=0)
    val = DataLoader(InkPatches(ROOT / "data/cache/frag3", a.n_val, seed=7),
                     batch_size=a.batch, num_workers=0)
    net = Ink3D().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr)
    bce = torch.nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    best, hist = 0.0, []
    for ep in range(a.epochs):
        net.train()
        tl = 0.0
        for x, y in train:
            x, y = x[:, None].to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device, torch.float16):
                logits = net(x)
                loss = bce(logits, y) + dice_loss(logits, y).mean()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tl += float(loss)
        net.eval()
        f1s, aucs = [], []
        with torch.inference_mode():
            for x, y in val:
                x = x[:, None].to(device)
                with torch.autocast(device, torch.float16):
                    logits = net(x)
                f1, auc = f1_auc(logits, y)
                f1s.append(f1)
                aucs.append(auc)
        f1m, aucm = float(np.mean(f1s)), float(np.mean(aucs))
        hist.append({"ep": ep, "train_loss": tl / len(train),
                     "val_f1": f1m, "val_auc": aucm})
        print(f"ep={ep} loss={tl/len(train):.4f} val_f1={f1m:.4f} "
              f"val_auc={aucm:.4f}", flush=True)
        if f1m > best:
            best = f1m
            torch.save(net.state_dict(), out / "best.pt")
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    print(f"best val_f1={best:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
