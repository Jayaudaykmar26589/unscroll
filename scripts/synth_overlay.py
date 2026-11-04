"""Overlay a traced curve on a synthetic slice with the ground truth.

  python scripts/synth_overlay.py --synth data/synth/crush2 --curves curves_synth/crush2 --z 0 30 49

Green: target sheet centre line (within half a turn of the click). Red: trace. Writes overlay_zNNNNN.png
(full slice) and overlay_zNNNNN_zoom.png (around the trace) into --curves.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw


def render(synth, curves, z, theta_c, pad=60):
    img = tifffile.imread(synth / f"{z:05d}.tif").astype(np.float32)
    lo, hi = np.percentile(img, (1, 99))
    u8 = np.clip((img - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    im = Image.fromarray(u8).convert("RGB")
    draw = ImageDraw.Draw(im)
    truth = np.load(synth / "truth.npz")
    gx, gy, gt = truth[f"x_{z:05d}"], truth[f"y_{z:05d}"], truth[f"t_{z:05d}"]
    sel = np.abs(gt - theta_c) < np.pi
    draw.point(list(zip(gx[sel].tolist(), gy[sel].tolist())), fill=(40, 220, 90))
    rec = json.loads((curves / f"{z:05d}.jsonl").read_text().splitlines()[0])
    pts = [tuple(p) for p in rec["points"]]
    if len(pts) > 1:
        draw.line(pts, fill=(255, 60, 60), width=2)
    im.save(curves / f"overlay_z{z:05d}.png")
    if pts:
        a = np.asarray(pts)
        box = (int(a[:, 0].min()) - pad, int(a[:, 1].min()) - pad,
               int(a[:, 0].max()) + pad, int(a[:, 1].max()) + pad)
        zoom = im.crop(box)
        scale = 600 / max(zoom.size)
        zoom.resize((int(zoom.size[0] * scale), int(zoom.size[1] * scale)), Image.NEAREST).save(
            curves / f"overlay_z{z:05d}_zoom.png")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", required=True)
    ap.add_argument("--curves", required=True)
    ap.add_argument("--z", type=int, nargs="+", default=[0])
    a = ap.parse_args(argv)
    synth, curves = Path(a.synth), Path(a.curves)
    score = json.loads((curves / "score.json").read_text())["summary"] if (curves / "score.json").exists() else None
    theta_c = score["theta_c"] if score else json.loads((synth / "meta.json").read_text())["click_theta"]
    for z in a.z:
        render(synth, curves, z, theta_c)
    print(f"overlays written to {curves}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
