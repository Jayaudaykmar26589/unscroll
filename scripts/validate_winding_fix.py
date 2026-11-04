"""Reassess the same 50 frozen curves: legacy tangent test vs live LayerGuard.

This is a guard-only replay. It never moves points or reruns SAM/snapping.
Known geometric counterfactuals are separate from unlabeled real CT results.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.trace.guards import LayerGuard, check_winding, load_cfg
from scripts.validate_curve_fix import geometry, summarize, load_module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", default="reports/ridge_fix/curves_corrected")
    ap.add_argument("--out", default="reports/winding_fix")
    a = ap.parse_args()
    curves, out = Path(a.curves), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(curves.glob("[0-9]*.jsonl"))
    if len(paths) != 50:
        raise ValueError(f"expected 50 curves, got {len(paths)}")
    legacy = load_module(out / "baseline_source/guards.py", "legacy_winding_guards")
    cfg = load_cfg(ROOT / "src/trace/guards.yaml")
    guard = LayerGuard(cfg)
    rows, hashes, geoms = [], [], []
    controls = {k: {"pass": 0, "flag": 0, "unassessed": 0} for k in
                ("same_curve_reversed_and_rotated", "normal_drift_1_5_voxels", "normal_shift_20_voxels")}
    rotation = np.array([[np.cos(.7), -np.sin(.7)], [np.sin(.7), np.cos(.7)]])
    for path in paths:
        rec = json.loads(path.read_text())
        p = np.asarray(rec["points"], dtype=float)
        z = rec.get("source_z", rec["z"])
        angles = np.degrees(np.unwrap(np.arctan2(np.diff(p[:, 1]), np.diff(p[:, 0]))))
        old = legacy.check_winding(angles.tolist(), 5.)
        new = guard.evaluate(p, z)
        rows.append({"z": rec["z"], "source_z": z, "source_file": rec.get("source_file"),
                     "legacy": old, "corrected": new})
        hashes.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        geoms.append(geometry(p))
        tangent = np.gradient(p, axis=0)
        normals = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
        pairs = {"same_curve_reversed_and_rotated": (p[::-1] @ rotation + 123, p @ rotation + 123),
                 "normal_drift_1_5_voxels": (p + 1.5 * normals, p),
                 "normal_shift_20_voxels": (p + 20 * normals, p)}
        for name, (current, reference) in pairs.items():
            result = check_winding(current, reference,
                                   distance_voxels=cfg["layer_distance_voxels"],
                                   min_run_voxels=cfg["layer_min_run_voxels"],
                                   min_overlap=cfg["layer_min_overlap"])
            controls[name]["unassessed" if not result["assessed"] else
                           "pass" if result["pass"] else "flag"] += 1
        print(f"z={z} {new['status']} assessed={new['winding']['assessed']} "
              f"runs={len(new['winding']['runs'])} ref={new['reference_z']} "
              f"reason={new['winding']['reason']}", flush=True)
    legacy_summary = {"failed_slices": sum(not r["legacy"]["pass"] for r in rows),
                      "point_violations": sum(len(r["legacy"]["violations"]) for r in rows),
                      "tangent_threshold_deg": 5.0}
    corrected_summary = {
        "assessed_slices": sum(r["corrected"]["winding"]["assessed"] for r in rows),
        "unassessed_slices": sum(not r["corrected"]["winding"]["assessed"] for r in rows),
        "flagged_slices": sum(r["corrected"]["winding"]["pass"] is False for r in rows),
        "passed_slices": sum(r["corrected"]["winding"]["pass"] is True for r in rows),
        "flagged_arc_runs": sum(len(r["corrected"]["winding"]["runs"]) for r in rows
                                if r["corrected"]["winding"]["pass"] is False),
        "anchor_departure_arc_runs_including_unassessed": sum(len(r["corrected"]["winding"]["runs"]) for r in rows),
        "stale_reference_slices": sum(r["corrected"]["winding"]["reason"] == "stale_reference_departure" for r in rows),
        "kink_failed_slices": sum(not r["corrected"]["kink"]["pass"] for r in rows),
        "ok_slices": sum(r["corrected"]["status"] == "ok" for r in rows)}
    result = {"method": "Guard-only replay of unchanged ridge-fix curves; same live LayerGuard implementation.",
              "limitations": "No real layer-ID annotations: flagged transverse departures are candidate jumps, "
              "deformation or segmentation errors, not confirmed layer switches. First curve seeds an "
              "unverified hypothesis; no registration is fitted and rejected curves cannot reanchor it. "
              "Large legitimate motion can also exceed the G1 2-voxel tolerance.",
              "legacy": legacy_summary, "corrected": corrected_summary,
              "config": cfg, "geometry_unchanged": summarize(geoms),
              "controlled_geometry_probes": controls, "source_hashes": hashes, "per_slice": rows}
    (out / "validation.json").write_text(json.dumps(result, indent=2))
    with open(out / "assessments.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("source_hashes", "per_slice")}, indent=2))


if __name__ == "__main__":
    main()
