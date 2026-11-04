"""Joint normal-offset snapping with explicit continuity constraints.

Dynamic programming selects a ridge path rather than independent point maxima.
The original (zero-offset) path is always feasible. Weak/missing evidence can
therefore be bridged without jumping to a neighboring layer. This constrains
geometry; it does not establish the anatomical identity of a ridge.
"""
import numpy as np


def _smooth(points, radius=2):
    """Symmetric mean with linear endpoint extrapolation (no endpoint shrink)."""
    if len(points) < 3:
        return points.copy()
    pad = np.arange(radius, 0, -1)[:, None]
    left = points[0] - pad * (points[1] - points[0])
    right = points[-1] + np.arange(1, radius + 1)[:, None] * (points[-1] - points[-2])
    extended = np.concatenate((left, points, right))
    return sum(extended[i:i + len(points)] for i in range(2 * radius + 1)) / (2 * radius + 1)


def continuous_snap(curve, signal, span=10, ridge=False):
    """Snap to brightness or a binary ridge map while retaining path continuity.

    Per-edge length is bounded by max(1.25*original_length+0.25, 1),
    and each edge must make at least 10% of its original forward progress.
    Offset differences carry a quadratic penalty. A small symmetric smoothing
    removes pixel staircasing; a feasible blend with the input enforces the
    same hard bounds and a span-sized displacement budget after smoothing.
    Counts and order are preserved. Infeasible evidence is ignored, not joined
    by a fabricated edge. Degenerate/short curves or absent evidence stay put.
    """
    p = np.asarray(curve, dtype=float).reshape(-1, 2)
    if span < 0 or int(span) != span:
        raise ValueError("span must be a nonnegative integer")
    if not np.isfinite(p).all():
        raise ValueError("curve must be finite")
    if len(p) < 3 or span == 0:
        return p.copy()
    image = np.asarray(signal)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("signal must be a finite 2D image")
    edges = np.diff(p, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths < 1e-9):
        return p.copy()  # a normal/forward direction is undefined here
    tangent = np.gradient(_smooth(p), axis=0)
    norm = np.linalg.norm(tangent, axis=1)
    if np.any(norm < 1e-9):
        return p.copy()
    normals = np.column_stack((-tangent[:, 1], tangent[:, 0])) / norm[:, None]
    offsets = np.arange(-span, span + 1, dtype=float)
    candidates = p[:, None, :] + normals[:, None, :] * offsets[None, :, None]
    xy = np.rint(candidates).astype(int)
    h, w = image.shape
    valid = (xy[..., 0] >= 0) & (xy[..., 0] < w) & (xy[..., 1] >= 0) & (xy[..., 1] < h)
    values = image[np.clip(xy[..., 1], 0, h - 1), np.clip(xy[..., 0], 0, w - 1)].astype(float)
    if not valid.any():
        return p.copy()
    observed = values[valid]
    if observed.max() == observed.min():
        return p.copy()
    if ridge:
        strength = values
    else:
        lo, hi = np.percentile(observed, (5, 95))
        strength = np.clip((values - lo) / max(hi - lo, 1e-9), 0, 1)
    emission = -4.0 * strength + 0.025 * offsets[None, :]**2
    emission[~valid] = np.inf
    # Preserve an out-of-frame original instead of inventing an entry path.
    emission[:, span] = np.where(valid[:, span], emission[:, span], 0.0)
    limit = np.maximum(1.25 * lengths + 0.25, 1.0)
    change_cost = 0.6 * (offsets[:, None] - offsets[None, :])**2
    cost = emission[0].copy()
    parents = np.zeros((len(p), len(offsets)), dtype=int)
    for i in range(1, len(p)):
        delta = candidates[i][None, :, :] - candidates[i - 1][:, None, :]
        feasible = ((np.linalg.norm(delta, axis=-1) <= limit[i - 1] + 1e-9)
                    & ((delta * edges[i - 1]).sum(-1) >= .1 * lengths[i - 1]**2))
        transition = cost[:, None] + change_cost
        transition[~feasible] = np.inf
        parents[i] = np.argmin(transition, axis=0)
        cost = transition[parents[i], np.arange(len(offsets))] + emission[i]
    if not np.isfinite(cost).any():
        return p.copy()
    chosen = np.empty(len(p), dtype=int)
    chosen[-1] = int(np.argmin(cost))
    for i in range(len(p) - 1, 0, -1):
        chosen[i - 1] = parents[i, chosen[i]]
    target = _smooth(candidates[np.arange(len(p)), chosen])
    # Feasibility of a global blend is convex in alpha. Backtracking keeps the
    # fallback explicit and covers smoothing, boundary and tight-corner cases.
    alpha = 1.0
    input_in_bounds = np.all((p >= [0, 0]) & (p <= [w - 1, h - 1]))
    while alpha >= 1 / 1024:
        out = p + alpha * (target - p)
        delta = np.diff(out, axis=0)
        if (np.all(np.linalg.norm(delta, axis=1) <= limit + 1e-9)
                and np.all((delta * edges).sum(1) >= .1 * lengths**2)
                and np.max(np.linalg.norm(out - p, axis=1)) <= span + 1e-9
                and (not input_in_bounds or np.all((out >= [0, 0]) & (out <= [w - 1, h - 1])))):
            return out
        alpha *= .5
    return p.copy()
