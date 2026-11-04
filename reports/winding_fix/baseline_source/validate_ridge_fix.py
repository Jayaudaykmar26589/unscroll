"""Replay all saved masks through live snapping; compare the prior 50 slices.

No new model inference: identical masks, frames, transforms and seed policy.
Guard formulas/thresholds stay unchanged. Metrics are measured on serialized
coordinates for both arms. Ridge coincidence is an image proxy, not layer GT.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import tifffile
from PIL import Image

from scripts.flatten_sheet import flatten
from scripts.ridge_eval import dilate2, ridge_mask
from scripts.validate_curve_fix import geometry, summarize, sheet_metrics
from src.trace.guards import check_winding, check_curvature, load_cfg
from src.trace.sam3_tracker import postprocess_mask
from src.trace.slices import resolve_slice


def guards(points, cfg):
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(p) < 3:
        return {"status": "failed", "winding_violations": 0, "kink_violations": 0,
                "winding_pass": False, "kink_pass": False}
    angles = np.degrees(np.unwrap(np.arctan2(np.diff(p[:, 1]), np.diff(p[:, 0]))))
    winding = check_winding(angles.tolist(), cfg["winding_reversal_deg"])
    kink = check_curvature(p, cfg["kink_deg"])
    return {"status": "ok" if winding["pass"] and kink["pass"] else "suspect",
            "winding_violations": len(winding["violations"]),
            "kink_violations": len(kink["violations"]),
            "winding_pass": winding["pass"], "kink_pass": kink["pass"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="reports/curve_fix/curves_corrected")
    ap.add_argument("--slices", default="data/cache/scroll1_6100_6200")
    ap.add_argument("--out", default="reports/ridge_fix")
    ap.add_argument("--click", default="4820,3403,6100")
    a = ap.parse_args()
    baseline, out = Path(a.baseline), Path(a.out)
    corrected = out / "curves_corrected"
    corrected.mkdir(parents=True, exist_ok=True)
    paths = sorted(baseline.glob("[0-9]*.jsonl"))
    if len(paths) != 50:
        raise ValueError(f"expected exactly 50 baseline slices, got {len(paths)}")
    tifs = sorted(Path(a.slices).glob("*.tif*"))
    cfg = load_cfg(ROOT / "src/trace/guards.yaml")
    previous, rows, source_hashes = None, [], []
    cx, cy, _ = map(int, a.click.split(","))
    for frame, path in enumerate(paths):
        rec = json.loads(path.read_text())
        tif = resolve_slice(rec, tifs)
        mask_path = baseline / f"mask_{rec['z']:05d}.npz"
        frame_path = baseline / "frames3" / f"{frame:05d}.jpg"
        with np.load(mask_path) as d:
            mask, transform = d["mask"], d["transform"]
        xs, ys, ox, oy = transform
        gray = np.asarray(Image.open(frame_path))[:, :, 0]
        seed = ((cx - ox) / xs, (cy - oy) / ys) if previous is None else previous.mean(axis=0)
        curve, raw = postprocess_mask(mask, gray, transform, tif, seed=seed, previous=previous)
        if len(raw):
            previous = raw
        new = {**rec, "points": np.round(curve, 1).tolist(),
               "origin": np.round(curve[0], 1).tolist() if len(curve) else None}
        (corrected / path.name).write_text(json.dumps(new) + "\n")
        image = tifffile.imread(tif).astype(float)
        ridges = dilate2(ridge_mask(image))
        pair = {"z": rec["z"], "source_file": tif.name}
        for label, data in (("baseline", rec), ("corrected", new)):
            p = np.asarray(data["points"], dtype=float).reshape(-1, 2)
            stats = {**geometry(p), **guards(p, cfg)}
            x = np.clip(p[:, 0].astype(int), 0, image.shape[1] - 1)
            y = np.clip(p[:, 1].astype(int), 0, image.shape[0] - 1)
            stats["ridge_frac2"] = float(ridges[y, x].mean()) if len(p) else 0.
            pair[label] = stats
        pair["max_displacement_from_raw_fullres"] = float(np.max(np.linalg.norm(
            curve - (raw * [xs, ys] + [ox, oy]), axis=1))) if len(raw) else 0.
        source_hashes.append({"z": rec["z"], "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
                              "frame_sha256": hashlib.sha256(frame_path.read_bytes()).hexdigest()})
        rows.append(pair)
        print(f"validated z={rec.get('source_z', rec['z'])} "
              f"step={pair['corrected']['max_step']:.2f} "
              f"kinks={pair['corrected']['kink_violations']} "
              f"winding={pair['corrected']['winding_violations']}", flush=True)
    flatten(baseline, a.slices, out / "sheet_baseline")
    flatten(corrected, a.slices, out / "sheet_corrected")
    result = {"method": "Identical cached masks/JPEG frames; live postprocess_mask function. No GPU rerun.",
              "limitations": "Unchanged guard formulas, including direction-sensitive winding. "
              "Ridge coincidence and geometric consistency do not establish layer identity.",
              "guard_config": cfg, "baseline": str(baseline), "corrected": str(corrected),
              "summary": {}, "flattening": {}, "source_hashes": source_hashes, "per_slice": rows}
    for label in ("baseline", "corrected"):
        metrics = [r[label] for r in rows]
        stats = summarize(metrics)
        for key in ("winding_violations", "kink_violations"):
            stats[key] = sum(r[key] for r in metrics)
        stats["suspect_slices"] = sum(r["status"] == "suspect" for r in metrics)
        stats["ok_slices"] = sum(r["status"] == "ok" for r in metrics)
        stats["failed_slices"] = sum(r["status"] == "failed" for r in metrics)
        stats["winding_failed_slices"] = sum(not r["winding_pass"] for r in metrics)
        stats["kink_failed_slices"] = sum(not r["kink_pass"] for r in metrics)
        stats["mean_ridge_frac2"] = float(np.mean([r["ridge_frac2"] for r in metrics]))
        stats["ridge_slices_ge_0_5"] = sum(r["ridge_frac2"] >= .5 for r in metrics)
        result["summary"][label] = stats
        result["flattening"][label] = sheet_metrics(np.load(out / f"sheet_{label}" / "sheet.npy"))
    (out / "validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("source_hashes", "per_slice")}, indent=2))


if __name__ == "__main__":
    main()
