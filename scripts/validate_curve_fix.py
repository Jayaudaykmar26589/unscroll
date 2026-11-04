"""Compare saved legacy and corrected tracing, including same-mask ordering.

Run after sam3_tracker --save-masks. Uses the frozen baseline source to
compare ordering on IDENTICAL masks; final curves compare two tracking runs.
All measures are geometry/image proxies, not independent layer accuracy.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import tifffile

from scripts.flatten_sheet import flatten
from scripts.ridge_eval import dilate2, ridge_mask
from src.flatten.anchor import resample
from src.trace.slices import resolve_slice
from src.trace.thin import mask_to_curve


def geometry(points):
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    d = np.diff(p, axis=0)
    lengths = np.linalg.norm(d, axis=1)
    unit = d / np.maximum(lengths[:, None], 1e-12)
    valid = (lengths[:-1] > 1e-9) & (lengths[1:] > 1e-9)
    turns = np.degrees(np.arccos(np.clip((unit[:-1] * unit[1:]).sum(1), -1, 1)))
    return {"points": len(p), "edges": len(lengths),
            "max_step": float(lengths.max()) if len(lengths) else 0.0,
            "steps_gt20": int((lengths > 20).sum()),
            "steps_gt_sqrt2": int((lengths > np.sqrt(2) + 1e-9).sum()),
            "arc_length": float(lengths.sum()),
            "turns_gt60": int(((turns > 60) & valid).sum())}


def summarize(rows):
    return {"slices": len(rows),
            **{key: sum(r[key] for r in rows) for key in
               ("points", "edges", "steps_gt20", "steps_gt_sqrt2", "turns_gt60")},
            "max_step": max((r["max_step"] for r in rows), default=0),
            "mean_arc_length": float(np.mean([r["arc_length"] for r in rows]))}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sheet_metrics(sheet):
    steps = np.linalg.norm(np.diff(sheet, axis=1), axis=-1)
    drift = np.linalg.norm(np.diff(sheet, axis=0), axis=-1)
    return {"shape": list(sheet.shape),
            "along_step_p95": float(np.percentile(steps, 95)),
            "along_step_max": float(steps.max()),
            "cross_slice_distance_median": float(np.median(drift)),
            "cross_slice_distance_p95": float(np.percentile(drift, 95)),
            "duplicate_edge_fraction": float((steps < 1e-9).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="curves_gono")
    ap.add_argument("--corrected", default="reports/curve_fix/curves_corrected")
    ap.add_argument("--slices", default="data/cache/scroll1_6100_6200")
    ap.add_argument("--out", default="reports/curve_fix")
    a = ap.parse_args()
    out, corrected = Path(a.out), Path(a.corrected)
    old_thin = load_module(out / "baseline_source/thin.py", "legacy_thin")
    old_anchor = load_module(out / "baseline_source/anchor.py", "legacy_anchor")
    tifs = sorted(Path(a.slices).glob("*.tif*"))
    rows, raw_old, raw_new = [], [], []
    old_curves = []
    for path in sorted(corrected.glob("[0-9]*.jsonl")):
        new = json.loads(path.read_text())
        old = json.loads((Path(a.baseline) / path.name).read_text())
        source = resolve_slice(new, tifs)
        image = tifffile.imread(source).astype(float)
        ridges = dilate2(ridge_mask(image))
        pair = {"z": new["z"], "source_file": source.name}
        for label, rec in (("baseline", old), ("corrected", new)):
            p = np.asarray(rec["points"], dtype=float).reshape(-1, 2)
            stats = geometry(p)
            x = np.clip(p[:, 0].astype(int), 0, image.shape[1] - 1)
            y = np.clip(p[:, 1].astype(int), 0, image.shape[0] - 1)
            stats["ridge_frac2"] = float(ridges[y, x].mean()) if len(p) else 0.0
            pair[label] = stats
        old_curves.append(resample(old["points"]))
        mask_file = corrected / f"mask_{new['z']:05d}.npz"
        with np.load(mask_file) as data:
            # No seed in either arm: isolates ordering/component handling only.
            mask = data["mask"]
            raw_old.append(geometry(old_thin.mask_to_curve(mask)))
            raw_new.append(geometry(mask_to_curve(mask)))
        rows.append(pair)
        print(f"validated {source.name}", flush=True)
    if not rows:
        raise ValueError("no corrected curves")
    old_sheet = np.stack(old_anchor.align_origins(old_curves)[0])
    np.save(out / "sheet_baseline.npy", old_sheet)
    flatten(a.corrected, a.slices, out / "sheet_corrected")
    new_sheet = np.load(out / "sheet_corrected/sheet.npy")
    result = {"limitations": "No aligned ground truth: geometry and ridge proxies only. "
              "Raw ordering compares identical masks without ridge snapping or seeds; "
              "final curves compare the saved historical run and a fresh GPU run. "
              "Shorter paths can lower distortion by discarding branches; inspect coverage.",
              "same_mask_raw_ordering": {"baseline": summarize(raw_old),
                                         "corrected": summarize(raw_new)},
              "final_curves": {},
              "flattening": {"baseline": sheet_metrics(old_sheet),
                             "corrected": sheet_metrics(new_sheet)}, "per_slice": rows}
    for label in ("baseline", "corrected"):
        stats = summarize([r[label] for r in rows])
        stats["mean_ridge_frac2"] = float(np.mean([r[label]["ridge_frac2"] for r in rows]))
        stats["ridge_slices_ge_0_5"] = sum(r[label]["ridge_frac2"] >= .5 for r in rows)
        result["final_curves"][label] = stats
    (out / "validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "per_slice"}, indent=2))


if __name__ == "__main__":
    main()
