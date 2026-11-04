"""Outer-boundary tracer: threshold + largest contour per slice.

Honest scope: traces the scroll's outer wrap (clean, separated by air),
NOT interior touching layers. Same curve JSONL interface as other
trackers. Used for the flattening demo + Tier A outer-wrap metrics.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import tifffile  # noqa: E402

from src.eval.runlog import RunLog  # noqa: E402
from src.trace.guards import LayerGuard, load_cfg  # noqa: E402


def outer_contour(img):
    import cv2
    lo, hi = np.percentile(img, (30, 99.5))
    mask = ((img > lo + 0.25 * (hi - lo)).astype(np.uint8)) * 255
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return np.zeros((0, 2))
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)
    return c


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", required=True)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    tifs = sorted(Path(a.slices).glob("*.tif*"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runlog = RunLog(ROOT, note="outer-boundary contour")
    cfg = load_cfg(ROOT / "src" / "trace" / "guards.yaml")
    guard = LayerGuard(cfg, curvature_stride=10)
    for i in range(min(a.window, len(tifs))):
        img = tifffile.imread(tifs[i]).astype(np.float32)
        curve = outer_contour(img)
        rec = {"z": i, "points": [[round(float(x), 1), round(float(y), 1)]
                                  for x, y in curve],
               "origin": ([round(float(curve[0][0]), 1),
                           round(float(curve[0][1]), 1)]
                          if len(curve) else None),
               "shift": [0, 0]}
        (out / f"{i:05d}.jsonl").write_text(json.dumps(rec) + "\n")
        assessment = guard.evaluate(np.asarray(rec["points"]).reshape(-1, 2), i)
        status = assessment["status"]
        checks = {"winding": assessment["winding"]["pass"],
                  "kink": assessment["kink"]["pass"], "details": assessment}
        runlog.log_slice(i, status, f"{i:05d}.jsonl", checks, 0)
        print(f"z={i} pts={len(curve)} {status}", flush=True)
    print(f"run {runlog.run_id}: curves in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
