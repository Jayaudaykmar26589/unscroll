"""Letter-scale demo on Kaggle: Unscroll traces a sheet with known text; the Grand Prize model reads it.

  python scripts/letters_kaggle.py --nz 300 --width 700      (Kaggle GPU notebook with internet)

Data stay on Kaggle. Steps:
 1. Official reference: Grand Prize (GP) TimeSformer on layers 17-42 of released
    segment 20230827161847 inside its known-text region; pick the window
    (nz slices tall in z, `width` pixels along the sheet) with the most ink.
 2. Read that window of the segment's .ppm (pixel -> volume x, y, z, normal).
 3. Fetch the CT region around it from the Scroll 1 OME-Zarr (8-bit, 128^3 chunks)
    and write uint8 crop slices.
 4. Control: sample the Zarr along the official surface (+-32 along the ppm
    normals) and run GP. Contrast is calibrated by a linear fit of official
    layer-32 pixels against the Zarr at the same 3D points.
 5. Unscroll: one click on the sheet at the window's first slice; trace with the
    SAM 3.1 tracker (walk), re-seeding from the last good curve when it stops;
    unroll +-32 layers (scripts/flatten_trace.py); GP on our own surface.
 6. Distance of every traced point to the official surface (ppm points).
Writes reports/letters/{summary.json, official_*.png, control_*.png, ours_*.png}.
"""
import argparse
import io
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ink_gp_kaggle import LAYERS, load_model, predict, to_input  # noqa: E402
from scripts.unroll3d import grid_from_curves, normals3d, sample_layers, smooth_grid, snap  # noqa: E402

BASE = "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg"
SEG = "20230827161847"
SEG_URL = f"{BASE}/paths/{SEG}"
ZARR = f"{BASE}/volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr/0"
REGION = (200, 5600, 1000, 4600)          # rows, cols of the official example (known text)
OUT = ROOT / "reports" / "letters"
TMP = Path("/tmp/letters")
Image.MAX_IMAGE_PIXELS = None


def get(url, rng=None, tries=5):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
            with urllib.request.urlopen(req, timeout=600) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(3 * (t + 1))
        except Exception:  # noqa: BLE001
            time.sleep(3 * (t + 1))
    raise SystemExit(f"failed: {url}")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------- official layers
def official_layers(r0, r1, c0, c1, layers):
    """uint8 (len(layers), r1-r0, c1-c0) from the segment's 16-bit layer TIFFs."""
    def one(z):
        img = tifffile.imread(io.BytesIO(get(f"{SEG_URL}/layers/{z:02d}.tif")))[r0:r1, c0:c1]
        return (img // 257).astype(np.uint8) if img.dtype == np.uint16 else img.astype(np.uint8)
    with ThreadPoolExecutor(6) as ex:
        return np.stack(list(ex.map(one, layers)))


# ------------------------------------------------------------------- ppm map
def ppm_rows(r0, r1):
    head = get(f"{SEG_URL}/{SEG}.ppm", (0, 4095))
    end = head.index(b"<>") + 3
    hdr = dict(line.split(": ") for line in head[:end].decode().splitlines() if ": " in line)
    W = int(hdr["width"])
    row = W * 6 * 8
    raw = get(f"{SEG_URL}/{SEG}.ppm", (end + r0 * row, end + r1 * row - 1))
    return np.frombuffer(raw, "<f8").reshape(r1 - r0, W, 6), W, int(hdr["height"])


# --------------------------------------------------------------------- zarr
def zarr_region(z0, z1, y0, y1, x0, x1):
    import numcodecs
    meta = json.loads(get(f"{ZARR}/.zarray"))
    codec = numcodecs.get_codec(meta["compressor"])
    C = meta["chunks"][0]
    out = np.zeros((z1 - z0, y1 - y0, x1 - x0), np.uint8)
    keys = [(cz, cy, cx) for cz in range(z0 // C, (z1 - 1) // C + 1)
            for cy in range(y0 // C, (y1 - 1) // C + 1) for cx in range(x0 // C, (x1 - 1) // C + 1)]

    def fetch(k):
        b = get(f"{ZARR}/{k[0]}/{k[1]}/{k[2]}")
        return k, (None if b is None else np.frombuffer(codec.decode(b), np.uint8).reshape(C, C, C))
    with ThreadPoolExecutor(16) as ex:
        for (cz, cy, cx), a in ex.map(fetch, keys):
            if a is None:
                continue
            gz, gy, gx = cz * C, cy * C, cx * C
            sz, sy, sx = max(z0, gz), max(y0, gy), max(x0, gx)
            ez, ey, ex_ = min(z1, gz + C), min(y1, gy + C), min(x1, gx + C)
            out[sz - z0:ez - z0, sy - y0:ey - y0, sx - x0:ex_ - x0] = a[sz - gz:ez - gz, sy - gy:ey - gy, sx - gx:ex_ - gx]
    log(f"zarr region {out.shape} from {len(keys)} chunks")
    return out


def trilinear(vol, z, y, x):
    z = np.clip(z, 0, vol.shape[0] - 1.001)
    y = np.clip(y, 0, vol.shape[1] - 1.001)
    x = np.clip(x, 0, vol.shape[2] - 1.001)
    z0, y0, x0 = np.floor(z).astype(int), np.floor(y).astype(int), np.floor(x).astype(int)
    fz, fy, fx = z - z0, y - y0, x - x0
    v = 0.0
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                w = (fz if dz else 1 - fz) * (fy if dy else 1 - fy) * (fx if dx else 1 - fx)
                v = v + w * vol[z0 + dz, y0 + dy, x0 + dx]
    return v


def png(a, path, lo=None, hi=None):
    a = np.asarray(a, np.float32)
    lo = np.nanpercentile(a, 1) if lo is None else lo
    hi = np.nanpercentile(a, 99) if hi is None else hi
    Image.fromarray(np.clip((np.nan_to_num(a, nan=lo) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)).save(path)


# ---------------------------------------------------------------- tracing
def trace(crop_dir, click, n, tag):
    out = TMP / f"curves_{tag}"
    if out.exists():
        shutil.rmtree(out)
    cmd = [sys.executable, "-m", "src.trace.sam3_tracker", "--slices", str(crop_dir),
           "--click", ",".join(str(int(v)) for v in click), "--window", str(n), "--crop", "2048",
           "--out", str(out), "--ckpt", str(ROOT / "checkpoints" / "sam3.1_multiplex.pt"),
           "--trim-memory", "--first-frame", "walk"]
    env = {**__import__("os").environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    good = {}
    for f in sorted(out.glob("[0-9]*.jsonl")):
        rec = json.loads(f.read_text().splitlines()[0])
        if len(rec["points"]) > 1:
            good[rec["source_z"]] = rec
    log(f"trace {tag}: exit {r.returncode}, {len(good)} slices")
    if r.returncode:
        tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-25:])
        print(f"--- tracer output tail ({tag}) ---\n{tail}\n---", flush=True)
        TRACER_ERRORS.append({"tag": tag, "tail": tail[-3000:]})
    return good


TRACER_ERRORS = []


def plan_segments(click_zs, z_end):
    """Assisted mode: [(start_z, window, owned_lo, owned_hi)] for sorted click slices.

    Each segment traces from its click to one slice past the next click; it owns
    slices [click_z, next_click_z), the last one owns through z_end - 1.
    """
    zs = sorted(set(int(z) for z in click_zs))
    out = []
    for i, z in enumerate(zs):
        hi = zs[i + 1] if i + 1 < len(zs) else z_end
        out.append((z, min(hi + 1, z_end) - z, z, hi))
    return out


def official_click(xyz, zs, z, x_lo, y_lo):
    """Simulated scholar click: the official surface point mid-sheet on slice z (crop coords)."""
    r = int(np.argmin(np.abs(zs.mean(axis=1) - z)))
    mid = xyz.shape[1] // 2
    return [int(round(xyz[r, mid, 0] - x_lo)), int(round(xyz[r, mid, 1] - y_lo)), int(round(xyz[r, mid, 2]))]


def shift_curve(S, net, dev, volf, pz, py, px, nrm, cal, refw, zs, seed=0):
    """How fast the letters fade when the surface is off the true sheet.

    constant: the whole official surface moved d voxels along its normals.
    wobble: each CT slice (row of the window) moved by its own random offset,
    |offset| with median m, like a trace that is m voxels off on median.
    Score: correlation of the ink map with the official one (GP on official layers).
    """
    def ink(d_rows):
        d = d_rows[:, None]
        st = np.stack([trilinear(volf, pz + (L - 32 + d) * nrm[..., 2], py + (L - 32 + d) * nrm[..., 1],
                                 px + (L - 32 + d) * nrm[..., 0]) for L in LAYERS])
        pred = predict(net, to_input(cal(st)), dev)
        return pred, float(np.corrcoef(pred.ravel(), refw.ravel())[0, 1])
    rows = zs.shape[0]
    const = []
    for d in (0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 7, -7, 10, -10):
        pred, r = ink(np.full(rows, float(d)))
        const.append({"offset_vx": d, "corr_with_official_ink": r, "mean_ink": float(pred.mean())})
        if d in (0, 2, 4, 7):
            png(pred, OUT / f"shift_{d}vx_ink.png", 0, 1)
        log(f"constant offset {d}: r={r:.3f}")
    rng = np.random.default_rng(seed)
    wob = []
    for m in (1, 2, 3, 5, 7):
        rs = []
        for rep_ in range(3):
            # |N(0, s)| has median 0.674 s; smooth along z over ~5 slices like a real trace
            raw = rng.normal(0, m / 0.674, rows + 8)
            sm = np.convolve(raw, np.ones(5) / 5, mode="same")[4:-4]
            sm *= m / max(np.median(np.abs(sm)), 1e-6)
            rs.append(ink(sm)[1])
        wob.append({"median_offset_vx": m, "corr_with_official_ink": float(np.mean(rs)), "runs": rs})
        log(f"wobble median {m}: r={np.mean(rs):.3f}")
    S["shift_curve"] = {"constant": const, "wobble": wob,
                        "method": "official surface of segment 20230827161847 moved along its ppm normals; "
                                  "GP ink model; correlation with GP on the official layers"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nz", type=int, default=400)
    ap.add_argument("--width", type=int, default=1000)
    ap.add_argument("--margin", type=int, default=300)
    ap.add_argument("--max-reseeds", type=int, default=6)
    ap.add_argument("--assist-every", type=int, default=0,
                    help="assisted mode: one simulated scholar click (official surface) every N slices")
    ap.add_argument("--sigmas", type=float, nargs="+", default=[2.0],
                    help="slice-to-slice smoothing strengths for the 3D unrolling of our trace")
    ap.add_argument("--shift-curve", action="store_true",
                    help="only the control, with the official surface moved along its normals: ink vs offset")
    ap.add_argument("--tag", default="", help="suffix for the output folder, so a new run never overwrites another")
    a = ap.parse_args()
    global OUT
    if a.shift_curve:
        OUT = OUT / "shift"
    elif a.assist_every or a.sigmas != [2.0]:
        # assisted = a click every 40 slices (the original assisted run); others carry N in the name
        base = ("assisted" if a.assist_every == 40 else f"assisted{a.assist_every}") if a.assist_every else "unassisted"
        if a.sigmas != [2.0]:                   # a new run: keep the earlier results intact
            base += "_smooth"
        OUT = OUT / (base + (f"_{a.tag}" if a.tag else ""))
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    t0 = time.time()
    S = {"segment": SEG, "volume": "20230205180739", "zarr": "54keV_7.91um_Scroll1A.zarr level 0 (uint8)"}
    net = load_model(dev)

    # 1-2. official map + reference ink over the known-text region
    r0, r1, c0, c1 = REGION
    ppm, W, H = ppm_rows(r0, r1)
    ppm = ppm[:, c0:c1]
    valid = np.isfinite(ppm[..., 2]) & (np.abs(ppm[..., :3]).sum(-1) > 0)
    gr = np.gradient(np.where(valid, ppm[..., 2], np.nan), axis=0)
    gc = np.gradient(np.where(valid, ppm[..., 2], np.nan), axis=1)
    z_axis = 0 if np.nanmedian(np.abs(gr)) > np.nanmedian(np.abs(gc)) else 1
    log(f"ppm window {ppm.shape}, z runs along image {'rows' if z_axis == 0 else 'cols'}")
    lay = official_layers(r0, r1, c0, c1, LAYERS)
    ref = predict(net, to_input(lay), dev, valid=valid)
    # densest window: nz along the z axis (pixels ~ slices), width along the sheet
    hz = a.nz if z_axis == 0 else a.width
    wz = a.width if z_axis == 0 else a.nz
    best, bpos = -1, (0, 0)
    for rr in range(0, ref.shape[0] - hz, 50):
        for cc in range(0, ref.shape[1] - wz, 50):
            v = valid[rr:rr + hz, cc:cc + wz]
            if v.mean() < 0.98:
                continue
            s = ref[rr:rr + hz, cc:cc + wz].std()     # strokes against background = letters
            if s > best:
                best, bpos = s, (rr, cc)
    wr, wc = bpos
    win = (slice(wr, wr + hz), slice(wc, wc + wz))
    pw, lw, refw = ppm[win], lay[:, win[0], win[1]], ref[win]
    if z_axis == 1:                                    # make rows = z everywhere
        pw, lw, refw = pw.transpose(1, 0, 2), lw.transpose(0, 2, 1), refw.T
    png(refw, OUT / "official_ink.png", 0, 1)
    png(lw[len(LAYERS) // 2], OUT / "official_ct.png")
    xyz, nrm = pw[..., :3], pw[..., 3:]
    zs = xyz[..., 2]
    S["window"] = {"image_rows": [r0 + wr, r0 + wr + hz], "image_cols": [c0 + wc, c0 + wc + wz],
                   "official_mean_ink": float(refw.mean()), "official_ink_std": float(refw.std()), "z_range": [float(zs.min()), float(zs.max())]}
    log(f"window {S['window']}")

    # 3. CT region around the window
    z_lo, z_hi = int(np.floor(zs.min())) - 2, int(np.ceil(zs.max())) + 3
    y_lo, y_hi = int(xyz[..., 1].min()) - a.margin, int(xyz[..., 1].max()) + a.margin
    x_lo, x_hi = int(xyz[..., 0].min()) - a.margin, int(xyz[..., 0].max()) + a.margin
    y_lo, x_lo = max(y_lo, 0), max(x_lo, 0)
    vol = zarr_region(z_lo, z_hi, y_lo, y_hi, x_lo, x_hi)
    crop_dir = TMP / "crop"
    crop_dir.mkdir(exist_ok=True)
    for i in range(vol.shape[0]):
        tifffile.imwrite(crop_dir / f"{z_lo + i:05d}.tif", vol[i])
    S["crop"] = {"z": [z_lo, z_hi], "y": [y_lo, y_hi], "x": [x_lo, x_hi], "shape": list(vol.shape)}

    # contrast calibration: official layer 32 (index 15 in LAYERS 17..42) vs zarr at ppm points
    k32 = LAYERS.index(32)
    pz, py, px = xyz[..., 2] - z_lo, xyz[..., 1] - y_lo, xyz[..., 0] - x_lo
    volf = vol.astype(np.float32)                      # once; reused by the control sampling
    zv = trilinear(volf, pz, py, px)
    ov = lw[k32].astype(np.float32)
    sel = np.isfinite(zv) & (ov > 0)
    A = np.vstack([zv[sel], np.ones(sel.sum())]).T
    gain, offset = np.linalg.lstsq(A, ov[sel], rcond=None)[0]
    corr = float(np.corrcoef(zv[sel], ov[sel])[0, 1])
    S["calibration"] = {"gain": float(gain), "offset": float(offset), "corr_zarr_vs_official": corr}
    log(f"calibration official ~ {gain:.3f}*zarr + {offset:.1f} (r={corr:.3f})")
    cal = lambda a_: np.clip(gain * a_ + offset, 0, 255).astype(np.uint8)

    # 4. control: official surface sampled from the zarr
    ctrl = np.stack([trilinear(volf, pz + (L - 32) * nrm[..., 2], py + (L - 32) * nrm[..., 1],
                               px + (L - 32) * nrm[..., 0]) for L in LAYERS])
    ctrl_pred = predict(net, to_input(cal(ctrl)), dev)
    png(ctrl_pred, OUT / "control_ink.png", 0, 1)
    png(ctrl[k32], OUT / "control_ct.png")
    S["control"] = {"mean_ink": float(ctrl_pred.mean()),
                    "corr_with_official_ink": float(np.corrcoef(ctrl_pred.ravel(), refw.ravel())[0, 1])}
    log(f"control {S['control']}")
    if a.shift_curve:
        shift_curve(S, net, dev, volf, pz, py, px, nrm, cal, refw, zs)
        S["minutes"] = round((time.time() - t0) / 60, 1)
        (OUT / "summary.json").write_text(json.dumps(S, indent=2, default=float))
        print(json.dumps(S, indent=2, default=float), flush=True)
        return

    # 5. Unscroll: free the GPU for the SAM tracker subprocess (it needs ~10 GB of the T4)
    del net
    torch.cuda.empty_cache()
    # click at the middle of the first full row, trace, re-seed
    row0 = int(np.argmin(np.abs(zs.mean(axis=1) - (z_lo + 3))))
    mid = xyz.shape[1] // 2
    cx, cy, cz = xyz[row0, mid, 0] - x_lo, xyz[row0, mid, 1] - y_lo, int(round(xyz[row0, mid, 2]))
    click = [int(round(cx)), int(round(cy)), cz]
    S["click_crop"] = click
    merged, tag, attempt = {}, 0, 0
    if a.assist_every:
        targets = list(range(cz, z_hi - 2, a.assist_every))
        clicks = [official_click(xyz, zs, z, x_lo, y_lo) for z in targets]
        clicks = [c for c in clicks if z_lo <= c[2] < z_hi]
        by_z = {c[2]: c for c in clicks}
        plan = plan_segments(by_z, z_hi)
        S["assisted"] = {"every_slices": a.assist_every, "clicks_crop": [by_z[p[0]] for p in plan],
                         "source": "official segment surface (simulated scholar clicks)"}
        for k, (z0, win, own_lo, own_hi) in enumerate(plan):
            got = trace(crop_dir, by_z[z0], max(win, 2), f"a{k}")
            for z, rec in got.items():
                if own_lo <= z < own_hi:
                    merged[z] = rec
            for z, rec in got.items():            # fill a neighbour's gap only if still empty
                merged.setdefault(z, rec)
    start, n_left = click, z_hi - cz
    for attempt in range(0 if a.assist_every else a.max_reseeds + 1):
        got = trace(crop_dir, start, max(n_left, 2), f"r{attempt}")
        for z, rec in got.items():
            merged.setdefault(z, rec)
        last = max(merged) if merged else None
        if last is None or last >= z_hi - 2 or (TRACER_ERRORS and not got):
            break
        p = np.asarray(merged[last]["points"], float)
        s = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
        i = int(np.argmin(abs(s - s[-1] / 2)))
        start, n_left = [int(round(p[i, 0])), int(round(p[i, 1])), last], z_hi - last
    S["traced_slices"] = len(merged)
    S["traced_z"] = [min(merged), max(merged)] if merged else None
    S["reseeds"] = 0 if a.assist_every else attempt
    mdir = TMP / "curves_merged"
    if mdir.exists():
        shutil.rmtree(mdir)
    mdir.mkdir()
    for z, rec in merged.items():
        (mdir / f"{z:05d}.jsonl").write_text(json.dumps(rec) + "\n")

    # 6. distance to the official surface (ppm points within half a slice)
    d_all = []
    for z, rec in merged.items():
        q = np.asarray(rec["points"], float) + [x_lo, y_lo]
        m = np.abs(xyz[..., 2] - z) < 0.5
        ref_pts = xyz[m][:, :2]
        if len(ref_pts) and len(q):
            d = np.sqrt(((q[:, None, :] - ref_pts[None, ::4, :]) ** 2).sum(-1)).min(1)
            d_all.append(d)
    if d_all:
        d = np.concatenate(d_all)
        S["distance_to_official_surface"] = {"within5": float((d <= 5).mean()), "within2": float((d <= 2).mean()),
                                             "median_vx": float(np.median(d)), "points": int(len(d)),
                                             "note": "points outside the official window's extent count as far"}

    # unroll +-32 and read our own surface (reload the ink model now that SAM is done)
    net = load_model(dev)
    from scripts.flatten_trace import flatten
    m = flatten(mdir, click[:2], TMP / "sheet", half_width=32, slices=crop_dir)
    surf = np.load(TMP / "sheet" / "surface.npy")
    rows = np.isfinite(surf).any(axis=(1, 2))
    surf = surf[rows]
    fin = np.isfinite(surf).all(axis=2)
    fill = np.nanmedian(surf)
    x = np.transpose(cal(np.nan_to_num(surf, nan=fill))[:, :, LAYERS], (2, 0, 1))
    ours = predict(net, to_input(x), dev)
    ours[~fin] = np.nan
    png(ours, OUT / "ours_ink.png", 0, 1)
    png(cal(np.nan_to_num(surf[:, :, 32], nan=fill)), OUT / "ours_ct.png")
    S["ours"] = {"image": list(ours.shape), "mean_ink": float(np.nanmean(ours)),
                 "p99": float(np.nanpercentile(ours, 99)), "frac_above_0.5": float(np.nanmean(ours > 0.5)),
                 "flatten_meta": m}
    S["tracer_errors"] = TRACER_ERRORS
    # 3D-normal unrolling (scripts/unroll3d.py)
    offs65 = np.arange(-32, 33)
    fillv = float(np.nanmedian(vol))
    def ink_of(stack_u8):
        return predict(net, to_input(stack_u8), dev)
    # (a) control: official surface positions, but normals computed by our own code
    Gc = np.stack([xyz[..., 0] - x_lo, xyz[..., 1] - y_lo, xyz[..., 2]], -1)
    nc = normals3d(Gc)
    if np.nansum(nc * nrm) < 0:                       # same side as the official normals
        nc = -nc
    Lc = sample_layers(volf, Gc, nc, offs65, origin=(0, 0, z_lo))
    pc = ink_of(cal(np.nan_to_num(Lc, nan=fillv))[LAYERS])
    S["control_own_normals"] = {"corr_with_official_ink": float(np.corrcoef(pc.ravel(), refw.ravel())[0, 1]),
                                "normal_agreement": float(np.nanmean(np.abs((nc * nrm).sum(-1))))}
    png(pc, OUT / "control3d_ink.png", 0, 1)
    log(f"control with our normals {S['control_own_normals']}")
    # (b) Unscroll's trace, unrolled along smoothed 3D normals, snapped without reference;
    #     --sigmas: the same trace at several slice-to-slice smoothing strengths
    G0, gz = grid_from_curves({z: rec["points"] for z, rec in merged.items()}, click[:2])
    for sig in a.sigmas:
        key, sfx = ("ours3d", "") if sig == 2 else (f"ours3d_sigma{sig:g}", f"_sigma{sig:g}")
        G = smooth_grid(G0, sigma_rows=sig)
        n3 = normals3d(G)
        L3 = sample_layers(volf, G, n3, offs65, origin=(0, 0, z_lo))
        valid3 = np.isfinite(L3).all(axis=0)
        u8 = cal(np.nan_to_num(L3, nan=fillv))
        p3, shift, side, tried = snap(ink_of, u8, valid3)
        p3 = np.where(valid3, p3, np.nan)
        png(p3, OUT / f"ours3d{sfx}_ink.png", 0, 1)
        png(np.where(valid3, (u8 if side > 0 else u8[::-1])[32 + shift], np.nan), OUT / f"ours3d{sfx}_ct.png")
        # agreement with the official ink at matched surface points (scoring only)
        ov, pv, dd = [], [], []
        for r in range(G.shape[0]):
            row = G[r]
            ok = np.isfinite(row[:, 0]) & np.isfinite(p3[r])
            if not ok.any():
                continue
            zr = row[ok][0, 2]
            m = np.abs(xyz[..., 2] - zr) < 0.5
            if not m.any():
                continue
            ref_xy = np.stack([xyz[..., 0][m] - x_lo, xyz[..., 1][m] - y_lo], 1)
            ref_ink = refw[m]
            q = row[ok][:, :2]
            d2 = ((q[:, None, :] - ref_xy[None, ::2, :]) ** 2).sum(-1)
            j = d2.argmin(1)
            dd.append(np.sqrt(d2[np.arange(len(q)), j]))
            ov.append(ref_ink[::2][j])
            pv.append(p3[r][ok])
        dd, ov, pv = np.concatenate(dd), np.concatenate(ov), np.concatenate(pv)
        near = dd <= 5
        S[key] = {"sigma_rows": sig, "image": list(p3.shape), "snap_shift_vx": int(shift), "snap_side": int(side),
                       "snap_scores": tried, "mean_ink": float(np.nanmean(p3)),
                       "frac_above_0.5": float(np.nanmean(p3 > 0.5)),
                       "points_within_5vx_of_official": float(near.mean()),
                       "ink_corr_with_official_where_within_5vx": float(np.corrcoef(pv[near], ov[near])[0, 1]) if near.sum() > 100 else None,
                       "ink_corr_with_official_all_matched": float(np.corrcoef(pv, ov)[0, 1])}
        log(f"{key} {({k: v for k, v in S[key].items() if k != 'snap_scores'})}")
    S["minutes"] = round((time.time() - t0) / 60, 1)
    (OUT / "summary.json").write_text(json.dumps(S, indent=2, default=float))
    print(json.dumps(S, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
