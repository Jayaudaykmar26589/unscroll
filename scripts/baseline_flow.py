"""Classic baseline tracers for the seeded evaluation (CPU, no SAM).

  python scripts/baseline_flow.py --seed-set main|heldout|large

Both start from the same first-slice curve as Unscroll's fixed tracer
(curves_walk*/<seed>/00000.jsonl), so only slice-to-slice propagation differs:
  flow       dense optical flow (OpenCV Farneback) between consecutive CT slices
             moves every curve point; no other information.
  flowridge  flow, then each point snaps to the brightest pixel within +-3 px
             along the curve normal (papyrus is bright in CT), then light smoothing.
These are generic classic tracers, not a reimplementation of any community tool.
Writes curves_flow*/ and curves_flowridge*/ in the tracker's format; score with
  python scripts/seeded_eval.py score --first-frame flow --seed-set ...
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.seeded_eval import OUT, SLICES, WINDOW, Z0, _seeds, _suffix  # noqa: E402

MARGIN = 96


def to_u8(a):
    lo, hi = np.percentile(a, (1, 99.5))
    return np.clip((a.astype(np.float32) - lo) * (255.0 / max(hi - lo, 1)), 0, 255).astype(np.uint8)


def bilinear(img, x, y):
    return cv2.remap(img, x.astype(np.float32)[None], y.astype(np.float32)[None], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)[0]


def normals(p):
    t = np.gradient(p, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    return np.column_stack([-t[:, 1], t[:, 0]])


def ridge_snap(p, img, span=3):
    n = normals(p)
    offs = np.arange(-span, span + 1, dtype=np.float32)
    xs = p[:, 0, None] + offs[None] * n[:, 0, None]
    ys = p[:, 1, None] + offs[None] * n[:, 1, None]
    v = bilinear(img, xs.ravel(), ys.ravel()).reshape(xs.shape)
    k = v.argmax(1)
    q = p + offs[k][:, None] * n
    q = cv2.GaussianBlur(q.astype(np.float32)[:, None, :], (1, 9), 0)[:, 0, :]  # smooth along the curve
    return q.astype(float)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-set", choices=["main", "heldout", "large"], default="main")
    a = ap.parse_args(argv)
    a.first_frame = ["walk"]
    start_dir = OUT / f"curves{_suffix(a, 'walk')}"
    seeds = [r for r in _seeds(a) if (start_dir / r.get("key", r["id"]) / "00000.jsonl").exists()]
    print(f"{len(seeds)} seeds with a first-slice curve in {start_dir.name}", flush=True)
    state = {}
    for r in seeds:
        key = r.get("key", r["id"])
        rec = json.loads((start_dir / key / "00000.jsonl").read_text().splitlines()[0])
        p = np.asarray(rec["points"], float).reshape(-1, 2)
        if len(p) < 2:          # the fixed tracer produced no first curve: nothing to propagate
            continue
        state[key] = {"flow": p.copy(), "flowridge": p.copy()}
    outs = {m: OUT / f"curves{_suffix(a, m)}" for m in ("flow", "flowridge")}
    for d in outs.values():          # a fresh run: no curves left from an earlier one
        if d.exists():
            shutil.rmtree(d)

    def write(z):
        for key, st in state.items():
            for m, p in st.items():
                d = outs[m] / key
                d.mkdir(parents=True, exist_ok=True)
                rec = {"z": z - Z0, "source_z": z, "source_file": f"{z:05d}.tif",
                       "points": np.round(p, 2).tolist(), "tracker": f"baseline_{m}"}
                (d / f"{z - Z0:05d}.jsonl").write_text(json.dumps(rec) + "\n")

    t0 = time.time()
    write(Z0)
    prev = tifffile.imread(SLICES / f"{Z0:05d}.tif")
    for z in range(Z0 + 1, Z0 + WINDOW):
        cur = tifffile.imread(SLICES / f"{z:05d}.tif")
        for key, st in state.items():
            for m, p in st.items():
                x0 = int(max(p[:, 0].min() - MARGIN, 0)); x1 = int(min(p[:, 0].max() + MARGIN, cur.shape[1]))
                y0 = int(max(p[:, 1].min() - MARGIN, 0)); y1 = int(min(p[:, 1].max() + MARGIN, cur.shape[0]))
                a8, b8 = to_u8(prev[y0:y1, x0:x1]), to_u8(cur[y0:y1, x0:x1])
                f = cv2.calcOpticalFlowFarneback(a8, b8, None, 0.5, 3, 21, 5, 7, 1.5, 0)
                lx, ly = p[:, 0] - x0, p[:, 1] - y0
                q = p + np.column_stack([bilinear(f[..., 0], lx, ly), bilinear(f[..., 1], lx, ly)])
                if m == "flowridge":
                    sm = cv2.GaussianBlur(b8, (0, 0), 1.5)
                    q = ridge_snap(q - [x0, y0], sm) + [x0, y0]
                st[m] = q
        write(z)
        prev = cur
        print(f"slice {z} done, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
