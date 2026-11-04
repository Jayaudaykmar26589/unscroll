"""Classical ridge tracker (promoted fallback): click -> layer curve.

Frangi-style vesselness (structure tensor) + tangent walk from click.
Same CLI + curve JSONL interface as sam_tracker.py (drop-in replacement).
  python -m src.trace.ridge_tracker --slices DIR --click X,Y,Z \\
      --window 32 --out curves/ [--polarity bright|dark]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.eval.runlog import RunLog  # noqa: E402
from src.trace.guards import check_curvature, check_winding, load_cfg  # noqa: E402


def vesselness(img, sigma=2.0):
    """Frangi-like response for thin bright lines, float32 HxW."""
    from scipy.ndimage import gaussian_filter
    img = img.astype(np.float32)
    g = gaussian_filter(img, sigma)
    gy, gx = np.gradient(g)
    gyy, _ = np.gradient(gy)
    _, gxx = np.gradient(gx)
    gxy = np.gradient(gy, axis=1)
    tmp = np.sqrt((gxx - gyy) ** 2 + 4 * gxy ** 2)
    l1 = (gxx + gyy + tmp) / 2  # larger magnitude for bright lines <0?
    l2 = (gxx + gyy - tmp) / 2
    # bright thin line: l1 ~ 0 (along), l2 << 0 (across)
    rb = np.abs(l1) / (np.abs(l2) + 1e-6)
    s2 = l1 * l1 + l2 * l2
    beta, c = 0.5, 0.5 * np.sqrt(s2.max())
    resp = np.exp(-rb * rb / (2 * beta * beta)) * (
        1 - np.exp(-s2 / (2 * c * c)))
    resp[l2 > 0] = 0
    return resp


def tangent_field(img, sigma=2.0):
    """Unit tangent (along-line) field from structure tensor."""
    from scipy.ndimage import gaussian_filter
    g = gaussian_filter(img.astype(np.float32), sigma)
    gy, gx = np.gradient(g)
    jxx = gaussian_filter(gx * gx, sigma)
    jyy = gaussian_filter(gy * gy, sigma)
    jxy = gaussian_filter(gx * gy, sigma)
    tr = jxx + jyy
    disc = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4 * jxy * jxy, 0))
    lmin = (tr - disc) / 2  # minor eigenvalue -> tangent direction
    vx = lmin - jyy
    vy = jxy
    n = np.hypot(vx, vy) + 1e-9
    return vx / n, vy / n


def walk(resp, tx, ty, x0, y0, thresh, max_steps=4000, step=1.0):
    """Walk both directions along tangent while response > thresh."""
    h, w = resp.shape
    pts_fwd, pts_bwd = [], []
    for direction in (1.0, -1.0):
        x, y = float(x0), float(y0)
        prev = None
        seen = set()
        for _ in range(max_steps):
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= xi < w and 0 <= yi < h):
                break
            if resp[yi, xi] < thresh:
                break
            key = (xi, yi)
            if key in seen:
                break  # loop guard (exact-pixel revisit)
            seen.add(key)
            t = np.array([tx[yi, xi], ty[yi, xi]])
            if prev is not None and (t * prev).sum() < 0:
                t = -t
            prev = t
            pts_fwd.append((x, y)) if direction > 0 else pts_bwd.append(
                (x, y))
            x, y = x + direction * t[0] * step, y + direction * t[1] * step
    pts_bwd.reverse()
    return pts_bwd + [(float(x0), float(y0))] + pts_fwd


def trace_slice(img, cx, cy, polarity="bright", sigma=2.0, q=0.90,
                frac=0.30):
    if polarity == "dark":
        img = img.max() - img
    resp = vesselness(img, sigma)
    tx, ty = tangent_field(img, sigma)
    rclick = float(resp[int(cy), int(cx)])
    floor = float(np.quantile(resp, q))
    if rclick < floor:
        return np.zeros((0, 2))  # click not on a ridge: fail fast
    pts = walk(resp, tx, ty, cx, cy, rclick * frac)
    return np.array(pts) if len(pts) > 2 else np.zeros((0, 2))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", required=True)
    ap.add_argument("--click", required=True)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--polarity", default="bright",
                    choices=("bright", "dark"))
    ap.add_argument("--z0", type=int, default=0)
    a = ap.parse_args(argv)
    import tifffile
    tifs = sorted(Path(a.slices).glob("*.tif*"))
    cx, cy, _ = (int(v) for v in a.click.split(","))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runlog = RunLog(ROOT, note=f"ridge {a.click} {a.polarity}")
    cfg = load_cfg(ROOT / "src" / "trace" / "guards.yaml")
    px, py = cx, cy
    for i in range(min(a.window, len(tifs))):
        img = tifffile.imread(tifs[i]).astype(np.float32)
        curve = trace_slice(img, px, py, a.polarity)
        rec = {"z": i, "points": [[round(float(x), 1), round(float(y), 1)]
                                  for x, y in curve],
               "origin": ([round(float(curve[0][0]), 1),
                           round(float(curve[0][1]), 1)]
                          if len(curve) else None),
               "shift": [0, 0]}
        (out / f"{i:05d}.jsonl").write_text(json.dumps(rec) + "\n")
        status, checks = "ok", {}
        if len(curve) >= 3:
            ang = np.unwrap(np.arctan2(np.diff(curve[:, 1]),
                                       np.diff(curve[:, 0])))
            w = check_winding(np.degrees(ang).tolist(),
                              cfg["winding_reversal_deg"])
            k = check_curvature(curve, cfg["kink_deg"])
            checks = {"winding": w["pass"], "kink": k["pass"]}
            status = "ok" if (w["pass"] and k["pass"]) else "suspect"
            # seed next slice from nearest point to previous click
            d2 = ((curve - [px, py]) ** 2).sum(1)
            px, py = (int(curve[d2.argmin()][0]),
                      int(curve[d2.argmin()][1]))
        else:
            status = "failed"
        runlog.log_slice(i, status, f"{i:05d}.jsonl", checks, 0)
        print(f"z={i} pts={len(curve)} {status}", flush=True)
    print(f"run {runlog.run_id}: curves in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
