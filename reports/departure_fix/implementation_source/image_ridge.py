"""Click-anchored CT ridge paths; SAM supplies an ROI, not a centerline.

For a locally monotone sheet patch, dynamic programming follows image evidence
with bounded slope and penalized changes in slope. The clicked ridge fixes the
initial sheet identity. Later slices use independently measured CT motion and
a one-voxel refinement corridor, never the new mask's moving medial axis.
Folded/non-monotone patches and unsupported motion fail explicitly.
"""
import numpy as np

from src.trace.snap import _smooth


def estimate_motion(previous_image, image, reference, radius=3):
    """Measure local CT translation by NCC, without fitting either curve."""
    p = np.asarray(reference)
    h, w = image.shape
    x0, y0 = np.floor(p.min(0) - 20).astype(int)
    x1, y1 = np.ceil(p.max(0) + 21).astype(int)
    x0, y0 = max(x0, radius), max(y0, radius)
    x1, y1 = min(x1, w - radius), min(y1, h - radius)
    if x1 - x0 < 10 or y1 - y0 < 10 or previous_image.shape != image.shape:
        return {"verified": False, "reason": "insufficient_image_overlap"}
    a = previous_image[y0:y1, x0:x1].astype(float)
    a -= a.mean()
    aa = (a*a).sum()
    if aa < 1e-9:
        return {"verified": False, "reason": "textureless_reference"}
    scores = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            b = image[y0+dy:y1+dy, x0+dx:x1+dx].astype(float)
            b -= b.mean()
            score = float((a*b).sum() / max(np.sqrt(aa*(b*b).sum()), 1e-9))
            scores.append((score, -dx*dx-dy*dy, dx, dy))
    score, _, dx, dy = max(scores)
    verified = score >= .8 and abs(dx) < radius and abs(dy) < radius
    return {"method": "ct_ncc", "verified": verified, "score": score,
            "dx": dx, "dy": dy, "search_radius": radius,
            "reason": "verified" if verified else "weak_or_boundary_motion"}


def _path(cost, max_slope=2):
    """Second-order DP over transverse coordinate and integer slope."""
    n, width = cost.shape
    velocities = np.arange(-max_slope, max_slope + 1)
    count = len(velocities)
    prior = np.repeat(cost[0, :, None], count, axis=1)
    parents = np.zeros((n, width, count), dtype=np.int8)
    for row in range(1, n):
        current = np.full_like(prior, np.inf)
        for vi, velocity in enumerate(velocities):
            target = np.arange(max(0, velocity), min(width, width + velocity))
            source = target - velocity
            options = prior[source] + .5*(velocity-velocities)**2 + .1*velocity**2
            best = options.argmin(1)
            parents[row, target, vi] = best
            current[target, vi] = options[np.arange(len(target)), best] + cost[row, target]
        prior = current
    if not np.isfinite(prior).any():
        return None
    x, velocity = np.unravel_index(np.argmin(prior), prior.shape)
    path = np.empty(n)
    for row in range(n - 1, -1, -1):
        path[row] = x
        pv = int(parents[row, x, velocity])
        x -= velocities[velocity]
        velocity = pv
    return path


def trace_image_ridge(image, mask, transform, guide, seed, reference=None,
                      reference_image=None):
    """Return (curve, diagnostics). Empty curve is an explicit failure.

    First frame: seed selects a CT peak within 12 voxels, then anchors a full
    ridge path inside a padded SAM ROI. Later frames: the old CT image predicts
    motion, followed by <=1 voxel of image-supported transverse refinement.
    Points are sampled every source voxel along the dominant axis. Slope <=2
    and five-sample smoothing keep along-curve continuity. No tuning to guard
    outcomes, no curve-to-curve registration, no fallback to copying a curve.
    """
    def fail(reason, **extra):
        return np.empty((0, 2)), {"success": False, "reason": reason, **extra}
    guide = np.asarray(guide, dtype=float)
    if len(guide) < 3:
        return fail("missing_sam_guide")
    xs, ys, ox, oy = transform
    full_guide = guide * [xs, ys] + [ox, oy]
    source = full_guide if reference is None else np.asarray(reference)
    axis = int(np.argmax(np.ptp(source, axis=0)))
    other = 1 - axis
    motion = {"method": "seed", "dx": 0, "dy": 0, "verified": False}
    if reference is not None:
        if reference_image is None:
            return fail("missing_reference_image")
        motion = estimate_motion(reference_image, image, source)
        if not motion["verified"]:
            return fail("unverified_ct_motion", motion=motion)
        source = source + [motion["dx"], motion["dy"]]
    direction = 1 if source[-1, axis] >= source[0, axis] else -1
    if np.maximum(-direction*np.diff(source[:, axis]), 0).sum() > .1*np.ptp(source[:, axis]):
        return fail("non_monotone_patch")
    longitudinal = np.arange(int(np.ceil(source[:, axis].min())), int(np.floor(source[:, axis].max())) + 1)
    if len(longitudinal) < 3:
        return fail("short_patch")
    h, w = image.shape
    dimensions = [w, h]
    if longitudinal[0] < 0 or longitudinal[-1] >= dimensions[axis]:
        return fail("motion_out_of_frame")
    lower = max(0, int(np.floor(source[:, other].min() - (32 if reference is None else 2))))
    upper = min(dimensions[other]-1, int(np.ceil(source[:, other].max() + (32 if reference is None else 2))))
    transverse = np.arange(lower, upper + 1)
    coords = np.empty((len(longitudinal), len(transverse), 2), dtype=int)
    coords[:, :, axis] = longitudinal[:, None]
    coords[:, :, other] = transverse[None, :]
    values = image[coords[..., 1], coords[..., 0]].astype(float)
    # Smooth only image evidence; do not derive the path from mask thickness.
    pad = np.pad(values, ((1,1),(1,1)), mode="edge")
    strength = .8 * sum(pad[i:i+len(longitudinal), j:j+len(transverse)] for i in range(3) for j in range(3))/9 + .2*values
    lo, hi = np.percentile(strength, [5, 95])
    if hi-lo < 1e-9:
        hi = strength.max()
    if hi-lo < 1e-9:
        return fail("no_ct_ridge_evidence")
    cost = 1 - np.clip((strength-lo)/(hi-lo), 0, 1)
    # The SAM ROI may include the sheet and inter-layer space. Expand only in
    # the transverse direction by 12 source voxels to accommodate coarse masks.
    support = np.zeros(cost.shape, dtype=bool)
    for offset in range(-12,13):
        q = coords.copy()
        q[:, :, other] += offset
        mx = np.floor((q[...,0]-ox)/xs).astype(int)
        my = np.floor((q[...,1]-oy)/ys).astype(int)
        valid = (mx>=0)&(my>=0)&(mx<mask.shape[1])&(my<mask.shape[0])
        support |= valid & mask[np.clip(my,0,mask.shape[0]-1),np.clip(mx,0,mask.shape[1]-1)]
    anchor = None
    if reference is None:
        cost[~support] = np.inf
        seed = np.asarray(seed)*[xs,ys]+[ox,oy]
        row = int(np.argmin(abs(longitudinal-seed[axis])))
        available = support[row] & (abs(transverse-seed[other]) <= 12)
        if not available.any():
            return fail("seed_outside_sam_support")
        anchor_cost = cost[row] + .003*(transverse-seed[other])**2
        anchor_cost[~available] = np.inf
        chosen = int(np.argmin(anchor_cost))
        anchor = coords[row,chosen].tolist()
        cost[row, np.arange(len(transverse)) != chosen] = np.inf
    else:
        ordered = source if direction > 0 else source[::-1]
        centers = np.interp(longitudinal, ordered[:,axis], ordered[:,other])
        corridor = abs(transverse[None,:] - centers[:,None]) <= 1.0 + 1e-9
        support_fraction = float((support & corridor).any(1).mean())
        if support_fraction < .5:
            return fail("insufficient_sam_overlap", motion=motion,
                        sam_overlap_fraction=support_fraction)
        # Coarse SAM ROI endpoints jitter between frames. Once CT motion is
        # verified, a few missing ROI rows must not destroy the entire image
        # path. Keep the one-voxel corridor hard; make only the ROI a soft cue.
        cost += .25*(~support)
        cost[~corridor] = np.inf
        # The hard corridor already bounds identity changes. A second strong
        # proximity penalty pins the curve to yesterday's sub-pixel position
        # despite contrary CT evidence. Use proximity only to break image ties.
        cost = 4*cost + 1e-4*(transverse[None,:] - centers[:,None])**2
    path = _path(cost)
    if path is None:
        return fail("no_supported_continuous_path", motion=motion)
    curve = np.empty((len(path),2))
    curve[:,axis] = longitudinal
    curve[:,other] = path + lower
    curve = _smooth(curve)
    # Pin the seed with a smooth local correction. Replacing a single sample
    # after smoothing creates an artificial kink even on a straight sheet.
    if anchor is not None:
        row = int(np.argmin(abs(curve[:,axis]-anchor[axis])))
        correction = anchor[other] - curve[row,other]
        curve[:,other] += correction * np.exp(-.5*((curve[:,axis]-anchor[axis])/4.)**2)
    if direction < 0:
        curve = curve[::-1].copy()
    return curve, {"success": True, "reason": "image_ridge", "motion": motion,
                   "axis": axis, "seed_anchor": anchor, "refinement_radius": 1.0,
                   "sam_overlap_fraction": 1.0 if reference is None else support_fraction}
