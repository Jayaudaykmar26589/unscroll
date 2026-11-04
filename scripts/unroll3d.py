"""3D-normal unrolling of a traced sheet, with reference-free surface snapping.

Slice-by-slice unrolling (scripts/flatten_trace.py) samples along in-plane
normals, which smears a sheet that runs obliquely through the slices. Here:
  1. grid_from_curves: stack per-slice traces into a (rows=z, cols=arc) grid of
     volume points, columns anchored from slice to slice (1-voxel arc spacing);
  2. smooth_grid: NaN-aware Gaussian smoothing (removes slice-to-slice jitter);
  3. normals3d: unit normals from the cross product of the two grid tangents;
  4. sample_layers: trilinear samples at offsets along those normals;
  5. snap: choose the depth offset and sheet side of the ink model's 26-layer
     window by the sharpness (std) of the predicted ink map - no reference used.
NumPy only (the ink model is passed in).
"""
import numpy as np


def resample(points, step=1.0):
    p = np.asarray(points, float).reshape(-1, 2)
    if len(p) < 2:
        return np.empty((0, 2))
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    if s[-1] < 2:
        return np.empty((0, 2))
    q = np.arange(0.0, s[-1], step)
    return np.column_stack([np.interp(q, s, p[:, 0]), np.interp(q, s, p[:, 1])])


def grid_from_curves(curves, click_xy):
    """curves: {z: [[x, y], ...]} -> (G (R, C, 3) with NaN gaps, zs list)."""
    zs = sorted(curves)
    anchor = np.asarray(click_xy, float)
    rows = []
    for z in zs:
        p = resample(curves[z])
        if not len(p):
            rows.append((z, None, None))
            continue
        i = int(np.argmin(np.hypot(*(p - anchor).T)))
        anchor = p[i]
        rows.append((z, p, i))
    good = [r for r in rows if r[1] is not None]
    left = max(i for _, _, i in good)
    right = max(len(p) - i for _, p, i in good)
    G = np.full((len(rows), left + right, 3), np.nan)
    for r, (z, p, i) in enumerate(rows):
        if p is None:
            continue
        cols = np.arange(len(p)) - i + left
        G[r, cols, 0], G[r, cols, 1], G[r, cols, 2] = p[:, 0], p[:, 1], z
    return G, zs


def _gauss1d(a, sigma, axis):
    if sigma <= 0:
        return a
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    a = np.moveaxis(a, axis, 0)
    pad = np.pad(a, [(r, r)] + [(0, 0)] * (a.ndim - 1), mode="edge")
    out = sum(w * pad[i:i + a.shape[0]] for i, w in enumerate(k))
    return np.moveaxis(out, 0, axis)


def smooth_grid(G, sigma_rows=2.0, sigma_cols=1.0):
    """NaN-aware separable Gaussian on a (R, C, 3) grid; keeps NaN where input was NaN."""
    m = np.isfinite(G[..., 0]).astype(float)
    v = np.nan_to_num(G)
    num, den = v * m[..., None], m
    for sig, ax in ((sigma_rows, 0), (sigma_cols, 1)):
        num, den = _gauss1d(num, sig, ax), _gauss1d(den, sig, ax)
    out = num / np.maximum(den[..., None], 1e-9)
    out[m == 0] = np.nan
    return out


def normals3d(G):
    """Unit normals (R, C, 3) from grid tangents along columns and rows."""
    tu = np.gradient(G, axis=1)
    tz = np.gradient(G, axis=0)
    n = np.cross(tu, tz)
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)


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


def sample_layers(vol, G, n, offsets, origin=(0, 0, 0)):
    """(len(offsets), R, C) samples at G + o*n; G is (x, y, z) volume coords, vol is (z, y, x)
    starting at origin (x0, y0, z0). NaN where G is NaN."""
    x0, y0, z0 = origin
    bad = ~np.isfinite(G[..., 0]) | ~np.isfinite(n[..., 0])
    Gf, nf = np.nan_to_num(G), np.nan_to_num(n)
    out = np.stack([trilinear(vol, Gf[..., 2] + o * nf[..., 2] - z0, Gf[..., 1] + o * nf[..., 1] - y0,
                              Gf[..., 0] + o * nf[..., 0] - x0) for o in offsets])
    out[:, bad] = np.nan
    return out


def snap(predict_fn, layers65, valid, shifts=range(-8, 9, 2), base=range(17, 43)):
    """Pick the depth shift and sheet side whose 26-layer window gives the sharpest ink map.
    layers65: (65, R, C) uint8 with index 32 at the traced surface. Returns (pred, shift, side, scores)."""
    best, scores = None, []
    base = list(base)
    for side in (+1, -1):
        stack = layers65 if side > 0 else layers65[::-1]
        for d in shifts:
            idx = [i + d for i in base]
            if min(idx) < 0 or max(idx) > 64:
                continue
            p = predict_fn(stack[idx])
            s = float(np.nanstd(np.where(valid, p, np.nan)))
            scores.append({"side": side, "shift": d, "ink_std": s})
            if best is None or s > best[0]:
                best = (s, p, d, side)
    return best[1], best[2], best[3], scores
