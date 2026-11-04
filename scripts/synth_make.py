"""Write a synthetic crushed-scroll slice stack with exact ground truth.

  python scripts/synth_make.py --out data/synth/crush2_open --crush 2 --view open

Each slice is a 1024 px window into a large scroll (radius 1500 px), centred
on the click. --view open looks at the side where gaps stay open; --view
touching looks at the flattened side where crushed layers press together.
Outputs: NNNNN.tif (uint16), truth.npz (per-slice centre lines and ink masks),
meta.json (config, click, click_theta), preview.png.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import tifffile  # noqa: E402

from src.synth.scroll import Scroll, SynthConfig  # noqa: E402

VIEWS = {"open": 0.0, "touching": np.pi / 2}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--crush", type=int, default=2, choices=range(5))
    ap.add_argument("--view", default="open", choices=sorted(VIEWS))
    ap.add_argument("--slices", type=int, default=64)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("*.tif")):
        raise SystemExit(f"{out} already has slices; use a fresh --out")
    cfg = SynthConfig(size=a.size, slices=a.slices, crush=a.crush,
                      view_angle=VIEWS[a.view], seed=a.seed)
    scroll = Scroll(cfg)
    t0 = time.time()
    truth = {}
    for z in range(cfg.slices):
        img, ink = scroll.render(z)
        u16 = np.clip(img * 40000, 0, 65535).astype(np.uint16)
        tifffile.imwrite(out / f"{z:05d}.tif", u16)
        for key, val in scroll.centerline(z).items():
            truth[f"{key}_{z:05d}"] = val
        truth[f"ink_{z:05d}"] = np.packbits(ink)
        if z == 0:
            from PIL import Image
            lo, hi = np.percentile(img, (1, 99))
            Image.fromarray(np.clip((img - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)).save(out / "preview.png")
    np.savez_compressed(out / "truth.npz", **truth)
    cx, cy, ct = scroll.click()
    meta = {**scroll.meta(), "view": a.view, "click": [cx, cy, 0], "click_theta": ct,
            "seconds": round(time.time() - t0, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"out": str(out), "click": meta["click"], "view": a.view,
                      "crush": a.crush, "seconds": meta["seconds"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
