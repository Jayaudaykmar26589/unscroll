"""Cross-slice layer continuity, spacing and local smoothness. Numpy + yaml.

Tangent rotation is not a layer identity signal. Layer continuity compares
source-space polylines; it needs a reference and is a geometric warning, not
proof of anatomical identity. Curves are never fitted to each other: that
could align a parallel wrong layer and erase the jump. LayerGuard can apply
independently verified CT-image motion for its matching reference frame.
"""

from pathlib import Path

import numpy as np
import yaml

DEFAULTS = {"layer_distance_voxels": 2.0, "layer_min_run_voxels": 3.0,
            "layer_min_overlap": 0.5, "spacing_frac": 0.30,
            "spacing_window": 10, "kink_deg": 60.0}


def load_cfg(path=None):
    cfg = dict(DEFAULTS)
    if path and Path(path).exists():
        cfg.update(yaml.safe_load(Path(path).read_text()))
    return cfg


def check_tangent_rotation(cum_angle_deg, tol_deg=5.0):
    """Legacy diagnostic for historical comparisons; NOT a layer-jump guard."""
    a = np.asarray(cum_angle_deg, dtype=float)
    d = np.diff(a)
    bad = np.where(d < -tol_deg)[0] + 1
    return {"violations": [int(i) for i in bad],
            "pass": bool(len(bad) == 0)}


def _polyline(points):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("expected curve coordinates (N,2), not tangent angles")
    if not np.isfinite(p).all():
        raise ValueError("curve coordinates must be finite")
    if len(p) > 1:
        keep = np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-9]
        p = p[keep]
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    return p, s


def _project(points, reference):
    """Closest points on segments, chunked in both dimensions."""
    a, v = reference[:-1], np.diff(reference, axis=0)
    length = np.linalg.norm(v, axis=1)
    arc = np.r_[0., np.cumsum(length)]
    distances, locations = np.full(len(points), np.inf), np.zeros(len(points))
    for i in range(0, len(points), 128):
        p = points[i:i + 128]
        best, where = np.full(len(p), np.inf), np.zeros(len(p))
        for j in range(0, len(a), 1024):
            av, vv = a[j:j + 1024], v[j:j + 1024]
            ll = length[j:j + 1024]
            t = np.clip(((p[:, None] - av) * vv).sum(2) / ll**2, 0, 1)
            foot = av + t[:, :, None] * vv
            d = np.linalg.norm(p[:, None] - foot, axis=2)
            k = d.argmin(1)
            current = d[np.arange(len(p)), k]
            update = current < best
            best[update] = current[update]
            where[update] = (arc[j + k] + t[np.arange(len(p)), k] * ll[k])[update]
        distances[i:i + len(p)] = best
        locations[i:i + len(p)] = where
    return distances, locations


def check_winding(points_xy, reference_xy=None, *, distance_voxels=2.0,
                  min_run_voxels=3.0, min_overlap=0.5):
    """Historical guard name, now a spatial layer-correspondence check.

    Resample at <=1-voxel spacing, then project onto reference segments. Away
    from endpoint caps this is the transverse distance to the reference path.
    Flag runs outside the 2-voxel G1 tolerance lasting >=3 voxels of arc length.
    Endpoint extensions are unassessed, not wrong-layer evidence. If less than
    half the current arc overlaps, return pass=None, assessed=False.

    This is invariant to traversal reversal, sampling density, angle wrapping
    and a common rigid coordinate transform. It accepts any normal rotation
    already represented by the reference, including S-curves and clockwise
    arcs. A uniformly shifted parallel layer is not registered away. Genuine
    deformation exceeding tolerance is also flagged; labels/image motion are
    needed to distinguish it from a jump. First slices cannot certify identity.
    """
    if distance_voxels <= 0 or min_run_voxels <= 0 or not 0 < min_overlap <= 1:
        raise ValueError("positive distance/run and overlap in (0,1] required")
    p, arc = _polyline(points_xy)
    result = {"pass": None, "assessed": False, "reason": "missing_reference",
              "violations": [], "runs": [], "overlap_fraction": 0.0,
              "distance_voxels": float(distance_voxels),
              "min_run_voxels": float(min_run_voxels)}
    if len(p) < 2:
        result["reason"] = "degenerate_curve"
        return result
    if reference_xy is None:
        return result
    ref, ref_arc = _polyline(reference_xy)
    if len(ref) < 2:
        result["reason"] = "degenerate_reference"
        return result
    # Equal physical spacing, independent of how the tracker sampled pixels.
    s = np.linspace(0., arc[-1], max(2, int(np.ceil(arc[-1])) + 1))
    q = np.column_stack([np.interp(s, arc, p[:, d]) for d in range(2)])
    distances, locations = _project(q, ref)
    overlap = (locations > 1e-6) & (locations < ref_arc[-1] - 1e-6)
    edge_overlap = overlap[:-1] & overlap[1:]
    fraction = float(np.diff(s)[edge_overlap].sum() / arc[-1])
    result["overlap_fraction"] = fraction
    if fraction < min_overlap:
        result["reason"] = "insufficient_overlap"
        return result
    bad = overlap & (distances > distance_voxels + 1e-9)
    changes = np.diff(np.r_[False, bad, False].astype(int))
    starts, stops = np.where(changes == 1)[0], np.where(changes == -1)[0]
    for start, stop in zip(starts, stops):
        run_length = s[stop - 1] - s[start]
        if run_length + 1e-9 >= min_run_voxels:
            result["runs"].append({"start_arc": float(s[start]), "end_arc": float(s[stop - 1]),
                                   "length_voxels": float(run_length),
                                   "max_distance": float(distances[start:stop].max())})
            result["violations"].extend(range(int(start), int(stop)))
    result.update({"assessed": True, "pass": not bool(result["runs"]),
                   "reason": "transverse_discontinuity" if result["runs"] else "consistent_overlap",
                   "max_distance": float(distances[overlap].max()),
                   "p95_distance": float(np.percentile(distances[overlap], 95))})
    return result


class LayerGuard:
    """State shared by live trackers and replay; rejects cannot move the anchor.

    The first geometrically valid curve seeds a hypothesis but stays unassessed.
    Retain that reference through missing/rejected frames; do not enlarge the
    spatial tolerance to hide gaps. Recovery to the accepted layer is possible.
    """
    def __init__(self, cfg=None, *, curvature_stride=1):
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.curvature_stride = curvature_stride
        self.reference, self.reference_z = None, None

    def evaluate(self, points_xy, z, *, image_motion=None):
        p = np.asarray(points_xy, dtype=float).reshape(-1, 2)
        if len(p) < 3:
            return {"status": "failed", "winding": {"pass": None, "assessed": False,
                    "reason": "missing_trace", "violations": [], "runs": []},
                    "kink": {"pass": False, "violations": []}, "reference_z": self.reference_z}
        reference = self.reference
        motion_used = False
        if reference is not None and image_motion is not None:
            m = image_motion
            if (m.get("method") == "ct_ncc" and m.get("verified") is True
                    and m.get("reference_z") == self.reference_z
                    and m.get("score", -1) >= .8
                    and np.isfinite([m.get("dx", np.nan), m.get("dy", np.nan)]).all()
                    and max(abs(m["dx"]), abs(m["dy"])) < m.get("search_radius", 0)):
                reference = reference + [m["dx"], m["dy"]]
                motion_used = True
        w = check_winding(p, reference,
                          distance_voxels=self.cfg["layer_distance_voxels"],
                          min_run_voxels=self.cfg["layer_min_run_voxels"],
                          min_overlap=self.cfg["layer_min_overlap"])
        k = check_curvature(p[::self.curvature_stride], self.cfg["kink_deg"])
        old_z = self.reference_z
        # A departure from an old anchor is not evidence of a *new* transition
        # at every later slice. Quarantine after a gap/rejection, keep the
        # measured anchor departure for review, and require recovery/reseeding.
        if w["assessed"] and w["pass"] is False and z != old_z + 1:
            w = {**w, "pass": None, "assessed": False,
                 "reason": "stale_reference_departure", "anchor_departure": True}
        if k["pass"] and (w["pass"] is True or self.reference is None):
            self.reference, self.reference_z = p.copy(), z
        return {"status": "ok" if w["pass"] is True and k["pass"] else "suspect",
                "winding": w, "kink": k, "reference_z": old_z,
                "image_motion_used": motion_used}


def check_spacing(gaps, window=10, frac=0.30):
    """Fail idx where gap deviates >frac from trailing-window median."""
    g = np.asarray(gaps, dtype=float)
    bad = []
    for i in range(len(g)):
        med = np.median(g[max(0, i - window):i] if i > 0 else g[i:i + 1])
        if med > 0 and abs(g[i] - med) / med > frac:
            bad.append(int(i))
    return {"violations": bad, "pass": bool(not bad)}


def check_curvature(points_xy, kink_deg=60.0):
    """Fail idx where adjacent-segment turn angle exceeds kink."""
    p = np.asarray(points_xy, dtype=float)
    v = np.diff(p, axis=0)
    n = np.linalg.norm(v, axis=1)
    n[n == 0] = 1.0
    u = v / n[:, None]
    dots = np.clip((u[:-1] * u[1:]).sum(1), -1.0, 1.0)
    turns = np.degrees(np.arccos(dots))
    bad = [int(i + 1) for i, t in enumerate(turns) if t > kink_deg]
    return {"violations": bad, "pass": bool(not bad)}
