"""G1/G2/G3 metric formulas (exact, per design Goals). Numpy only."""

import numpy as np


def _cdist_chunked(a, b, chunk=2048):
    """Min distance from each row of a to rows of b.

    Double-chunked (no scipy): safe for 1e5+ point clouds.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    out = np.full(len(a), np.inf)
    for i in range(0, len(a), chunk):
        ai = a[i:i + chunk]
        best = np.full(len(ai), np.inf)
        for j in range(0, len(b), chunk * 4):
            bj = b[j:j + chunk * 4]
            d2 = ((ai[:, None, :] - bj[None, :, :]) ** 2).sum(-1)
            best = np.minimum(best, d2.min(1))
        out[i:i + chunk] = np.sqrt(best)
    return out


def g1(traced_xyz, ref_xyz, tol=2.0, z_of_point=None,
       slice_pass_frac=0.8):
    """G1: fraction of traced points within tol voxels of reference.

    traced_xyz: (N,3) sampled surface points (1pt/voxel, every slice).
    ref_xyz: (M,3) released segment surface points.
    Returns dict(frac_within, p95, slice_pass_rate or None).
    Pass: frac_within >= 0.90 (plus slice_pass_rate >= 0.80 when z given).
    """
    d = _cdist_chunked(traced_xyz, ref_xyz)
    res = {"frac_within": float((d <= tol).mean()),
           "p95": float(np.percentile(d, 95)),
           "n": int(len(d))}
    if z_of_point is not None:
        z_of_point = np.asarray(z_of_point)
        ok_slices = 0
        for z in np.unique(z_of_point):
            dz = d[z_of_point == z]
            if (dz <= tol).mean() >= slice_pass_frac:
                ok_slices += 1
        res["slice_pass_rate"] = ok_slices / len(np.unique(z_of_point))
    else:
        res["slice_pass_rate"] = None
    res["pass"] = bool(res["frac_within"] >= 0.90 and
                        (res["slice_pass_rate"] is None or
                         res["slice_pass_rate"] >= 0.80))
    return res


def g2(surface_ids):
    """G2: jumps per 100 slices. ids[z] = segment-surface ID or -1 untraced.

    Jump = maximal run of >=3 consecutive slices on an ID different
    from the reference ID (first traced slice's ID).
    Returns dict(jumps, n_slices, per_100). Pass: per_100 <= 2.
    """
    ids = np.asarray(surface_ids)
    traced = ids[ids >= 0]
    if len(traced) == 0:
        return {"jumps": 0, "n_slices": int(len(ids)), "per_100": 0.0,
                "pass": False}
    ref = traced[0]
    jumps, run = 0, 0
    for v in ids:
        if v >= 0 and v != ref:
            run += 1
        else:
            if run >= 3:
                jumps += 1
            run = 0
    if run >= 3:
        jumps += 1
    per_100 = jumps / len(ids) * 100
    return {"jumps": jumps, "n_slices": int(len(ids)),
            "per_100": float(per_100), "pass": bool(per_100 <= 2.0)}


def _lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def g3(pred_glyphs, true_glyphs, excluded=None):
    """G3: exact-glyph accuracy + CER over included glyphs.

    pred/true: aligned lists of single-char strings. excluded: bool mask
    (illegible/damaged, counted separately). Pass: accuracy >= 0.70.
    """
    assert len(pred_glyphs) == len(true_glyphs)
    n = len(true_glyphs)
    excl = np.asarray(excluded if excluded is not None
                      else [False] * n, dtype=bool)
    idx = [i for i in range(n) if not excl[i]]
    if not idx:
        return {"accuracy": 0.0, "cer": 1.0, "n": 0,
                "n_excluded": int(excl.sum()), "pass": False}
    acc = sum(pred_glyphs[i] == true_glyphs[i] for i in idx) / len(idx)
    tot_dist = sum(_lev(pred_glyphs[i], true_glyphs[i]) for i in idx)
    tot_ch = sum(len(true_glyphs[i]) for i in idx) or 1
    return {"accuracy": float(acc), "cer": float(tot_dist / tot_ch),
            "n": len(idx), "n_excluded": int(excl.sum()),
            "pass": bool(acc >= 0.70)}
