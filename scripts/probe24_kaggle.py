"""Probe on the Vesuvius Challenge's current scans: Unscroll on the 2.4 um Scroll 1 volume (Kaggle).

  python scripts/probe24_kaggle.py [--clicks 8] [--levels 0 1]      (Kaggle GPU notebook with internet)

Data, streamed from the public open data bucket (s3://vesuvius-challenge-open-data, CC BY-NC 4.0):
  volume  PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr (zarr v2, uint8,
          128^3 chunks, uncompressed; level 1 = 4.8 um)
  layer   the community's 2026 trace of segment 20231210121321 (v8) on that volume (tifxyz,
          20-voxel grid = 48 um), used only to place the clicks and to score.
Clicks sit on the layer's intersection with slice Z0 (where this sheet crosses the region the
7.91 um evaluation used). Each click gets its own 2048 x 2048 x 20 crop; the unchanged tracer
(src.trace.sam3_tracker) runs with the original first slice (dp) and the fix (walk), at 2.4 um
and at 4.8 um (the tracer was developed at 7.91 um). Scores are in level-0 voxels (2.4 um) and um.
Writes reports/probe24/{summary.json, overlay_*.png}.
"""
import argparse
import io
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.meshref import SurfaceIndex, curve_xyz, sample_curve, slice_segments  # noqa: E402

BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
VOL = f"{BUCKET}/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr"
MESH = (f"{BUCKET}/PHercParis4/segments/20260604223808-20231210121321_v8/mesh/"
        "20260604223808-on-20260411134726-2.4um.tifxyz")
Z0, WINDOW, CROP, CH = 49420, 20, 2048, 128
UM = 2.4
OUT = ROOT / "reports" / "probe24"
TMP = Path("/tmp/probe24")
Image.MAX_IMAGE_PIXELS = None


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def get(url, tries=5):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            time.sleep(2 * (k + 1))
        except Exception:
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"failed: {url}")


def load_mesh(src=MESH):
    """tifxyz -> (vertices, faces) in level-0 voxels (x, y, z); LZW TIFFs read with Pillow."""
    def read(name):
        p = Path(src) / name
        data = p.read_bytes() if p.exists() else get(f"{src}/{name}")
        return np.asarray(Image.open(io.BytesIO(data)), dtype=float)
    x, y, z = read("x.tif"), read("y.tif"), read("z.tif")
    ok = z > 0
    H, W = z.shape
    idx = np.arange(H * W).reshape(H, W)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    q = ok[:-1, :-1] & ok[:-1, 1:] & ok[1:, :-1] & ok[1:, 1:]
    f = np.concatenate([np.stack([a[q], b[q], c[q]], 1), np.stack([b[q], d[q], c[q]], 1)])
    return np.stack([x, y, z], -1).reshape(-1, 3), f


def window(v, f, z_lo, z_hi):
    zt = v[f][:, :, 2]
    f = f[(zt.max(1) >= z_lo) & (zt.min(1) <= z_hi)]
    used = np.unique(f)
    remap = np.full(len(v), -1)
    remap[used] = np.arange(len(used))
    return v[used], remap[f]


def pick_clicks(v, f, n, min_gap=600.0, seed=20261102):
    segs = slice_segments(v, f, Z0)
    mids = segs.mean(1)
    rng = np.random.default_rng(seed)
    out = []
    for i in rng.permutation(len(mids)):
        if len(out) >= n:
            break
        if all(np.hypot(*(mids[i] - c)) >= min_gap for c in out):
            out.append(mids[i])
    return [[int(round(c[0])), int(round(c[1])), Z0] for c in out]


def fetch_crop(level, cx, cy, z0, n=WINDOW, size=CROP):
    """uint8 (n, size, size) from zarr level `level`; cx, cy, z0 in that level's voxels."""
    x0, y0 = int(cx) - size // 2, int(cy) - size // 2
    out = np.zeros((n, size, size), np.uint8)
    keys = [(iz, iy, ix) for iz in range(z0 // CH, (z0 + n - 1) // CH + 1)
            for iy in range(y0 // CH, (y0 + size - 1) // CH + 1)
            for ix in range(x0 // CH, (x0 + size - 1) // CH + 1)]

    def one(k):
        iz, iy, ix = k
        raw = get(f"{VOL}/{level}/{iz}/{iy}/{ix}")
        if raw is None:
            return
        c = np.frombuffer(raw, np.uint8).reshape(CH, CH, CH)
        zs, ys, xs = iz * CH, iy * CH, ix * CH
        za, zb = max(z0, zs), min(z0 + n, zs + CH)
        ya, yb = max(y0, ys), min(y0 + size, ys + CH)
        xa, xb = max(x0, xs), min(x0 + size, xs + CH)
        out[za - z0:zb - z0, ya - y0:yb - y0, xa - x0:xb - x0] = c[za - zs:zb - zs, ya - ys:yb - ys, xa - xs:xb - xs]
    with ThreadPoolExecutor(32) as ex:
        list(ex.map(one, keys))
    return out, (x0, y0)


def score(curve_dir, offset, s, index):
    """Curves in crop pixels of level s -> distances to the layer in level-0 voxels."""
    pts, wts = [], []
    for c in sorted(Path(curve_dir).glob("[0-9]*.jsonl")):
        rec = json.loads(c.read_text().splitlines()[0])
        if len(rec["points"]) < 2:
            continue
        q, w = sample_curve(curve_xyz(rec), 1.0)
        q = np.column_stack([(q[:, 0] + offset[0]) * s, (q[:, 1] + offset[1]) * s, q[:, 2] * s])
        pts.append(q)
        wts.append(w * s)
    if not pts:
        return {"slices_with_trace": 0}
    p, w = np.concatenate(pts), np.concatenate(wts)
    d = index.distances(p)
    wn = w / w.sum()
    return {"slices_with_trace": len(pts),
            "within5_vx": round(float(wn[d <= 5].sum()), 4),                 # 12 um
            "within_40um": round(float(wn[d <= 40 / UM].sum()), 4),          # same physical tolerance as 5 vx at 7.91 um
            "within_80um": round(float(wn[d <= 80 / UM].sum()), 4),          # a sheet is ~100-150 um thick; the layer is its surface
            "median_um": round(float(np.median(d) * UM), 2)}


def overlay(ct, curve_dir, segs, offset, path):
    import cv2
    img = cv2.cvtColor(ct, cv2.COLOR_GRAY2BGR)
    for a, b in segs:
        cv2.line(img, tuple(int(v) for v in a - offset), tuple(int(v) for v in b - offset), (105, 173, 239), 2)
    rec = json.loads(sorted(Path(curve_dir).glob("[0-9]*.jsonl"))[0].read_text().splitlines()[0])
    p = np.asarray(rec["points"], np.int32)
    if len(p) > 1:
        cv2.polylines(img, [p.reshape(-1, 1, 2)], False, (194, 224, 125), 2)
    cv2.imwrite(str(path), cv2.resize(img, (1024, 1024)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clicks", type=int, default=8)
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--gpus", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    v, f = load_mesh()
    v, f = window(v, f, Z0 - 8, Z0 + 2 * WINDOW + 8)
    index = SurfaceIndex(v, f)
    clicks = pick_clicks(v, f, a.clicks)
    log(f"layer window: {len(f)} triangles; clicks {clicks}")
    S = {"volume": "20260411134726-2.400um-0.2m-78keV-masked", "layer": "20260604223808-20231210121321_v8",
         "z0": Z0, "window": WINDOW, "crop": CROP, "clicks": clicks, "runs": []}
    jobs = []
    for L in a.levels:
        s = 2 ** L
        for k, (cx, cy, cz) in enumerate(clicks):
            d = TMP / f"L{L}_{k}" / "slices"
            d.mkdir(parents=True, exist_ok=True)
            vol, (x0, y0) = fetch_crop(L, cx // s, cy // s, cz // s)
            import tifffile
            for i in range(WINDOW):
                tifffile.imwrite(d / f"{cz // s + i:05d}.tif", vol[i])
            np.save(TMP / f"L{L}_{k}" / "ct0.npy", vol[0])
            jobs += [(L, k, mode, (x0, y0), (cx // s - x0, cy // s - y0, cz // s)) for mode in ("dp", "walk")]
            log(f"level {L} click {k}: crop at {x0},{y0}, nonzero {float((vol > 0).mean()):.2f}")
    import queue
    import torch
    n = a.gpus or max(1, torch.cuda.device_count())
    free = queue.Queue()
    for g in range(n):
        free.put(g)

    def run(job):
        L, k, mode, off, click = job
        g = free.get()
        try:
            out = TMP / f"L{L}_{k}" / mode
            cmd = [sys.executable, "-m", "src.trace.sam3_tracker", "--slices", str(TMP / f"L{L}_{k}" / "slices"),
                   "--click", ",".join(map(str, click)), "--window", str(WINDOW), "--crop", str(CROP),
                   "--out", str(out), "--ckpt", str(ROOT / "checkpoints" / "sam3.1_multiplex.pt"),
                   "--trim-memory", "--first-frame", mode]
            env = dict(__import__("os").environ, CUDA_VISIBLE_DEVICES=str(g))
            t = time.time()
            with open(TMP / f"L{L}_{k}_{mode}.log", "w") as fh:
                code = subprocess.call(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env)
            log(f"level {L} click {k} {mode}: exit {code} in {time.time() - t:.0f}s")
            return {"level": L, "um_per_voxel": UM * 2 ** L, "click": k, "mode": mode, "exit": code,
                    **(score(out, off, 2 ** L, index) if code == 0 else {})}
        finally:
            free.put(g)
    with ThreadPoolExecutor(n) as ex:
        S["runs"] = list(ex.map(run, jobs))
    for L in a.levels:                                   # one picture per level: first click, fixed tracer
        out = TMP / f"L{L}_0" / "walk"
        if out.exists() and any(out.glob("*.jsonl")):
            s = 2 ** L
            off = next(j[3] for j in jobs if j[0] == L and j[1] == 0)
            segs = slice_segments(v, f, Z0) / s
            overlay(np.load(TMP / f"L{L}_0" / "ct0.npy"), out, segs, np.asarray(off, float), OUT / f"overlay_L{L}.png")
    agg = {}
    for L in a.levels:
        for mode in ("dp", "walk"):
            r = [x for x in S["runs"] if x["level"] == L and x["mode"] == mode and "within5_vx" in x]
            if r:
                agg[f"L{L}_{mode}"] = {"traces": len(r), **{k: round(float(np.median([x[k] for x in r])), 4)
                                                           for k in ("within5_vx", "within_40um", "within_80um", "median_um")}}
    S["median_by_level_and_mode"] = agg
    S["minutes"] = round((time.time() - t0) / 60, 1)
    (OUT / "summary.json").write_text(json.dumps(S, indent=2))
    print(json.dumps(agg, indent=2), flush=True)


if __name__ == "__main__":
    main()
