"""Reference-seeded evaluation on real Scroll 1 CT: click ON a released mesh.

  python scripts/seeded_eval.py seeds   # pick one seed per covering mesh
  python scripts/seeded_eval.py trace   # run src.trace.sam3_tracker per seed
  python scripts/seeded_eval.py score   # exact distances to the seed's own mesh

The earlier evaluation clicked by hand and later found the nearest released
mesh 12 voxels away, so target identity was unconfirmed. Here each click is a
point on a released segment's intersection with the first slice, so the
intended layer is known. Meshes only choose the click and score the result;
the tracker never sees them. OBJ XYZ = TIFF (column, row, physical slice).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.eval.meshref import (SurfaceIndex, curve_xyz, load_obj,  # noqa: E402
                              sample_curve, slice_segments)

SEGS = ROOT / "data" / "cache" / "scroll1_segments"
SLICES = ROOT / "data" / "cache" / "scroll1_6100_6200"
OUT = ROOT / "reports" / "seeded_eval"
Z0, CROP = 6100, 2048
# seeds must cover the full cached 50-slice block; traces use the first WINDOW
# slices (SAM 3.1 runs about a minute per real slice on an 8 GB laptop GPU)
COVER, WINDOW = 50, 20
Z1 = Z0 + COVER - 1
PAD = 4  # z margin when keeping triangles for scoring


def _volume():
    return json.loads((SEGS / "volume.meta.json").read_text())


def _window_mesh(sid):
    """Triangles of mesh sid overlapping [Z0-PAD, Z1+PAD], cached as .npz."""
    cache = OUT / "mesh_cache" / f"{sid}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["v"], d["f"]
    v, f = load_obj(SEGS / f"{sid}.obj")
    zt = v[f][:, :, 2]
    keep = (zt.max(1) >= Z0 - PAD) & (zt.min(1) <= Z1 + PAD)
    f = f[keep]
    used = np.unique(f)
    remap = np.full(len(v), -1)
    remap[used] = np.arange(len(used))
    v, f = v[used], remap[f]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, v=v, f=f)
    return v, f


def compatible_ids():
    vol = _volume()
    ids = []
    for meta in sorted(SEGS.glob("*.meta.json")):
        sid = meta.name.split(".")[0]
        if sid == "volume" or not (SEGS / f"{sid}.obj").exists():
            continue
        if json.loads(meta.read_text()).get("volume") == vol["uuid"]:
            ids.append(sid)
    return ids


def pick_seed(segments, width, height):
    """Midpoint of the intersection segment nearest the median midpoint,
    restricted to points whose CROP window fits inside the slice."""
    mids = segments.mean(1)
    half = CROP // 2
    ok = ((mids[:, 0] >= half) & (mids[:, 0] <= width - half)
          & (mids[:, 1] >= half) & (mids[:, 1] <= height - half))
    if not ok.any():
        return None
    mids = mids[ok]
    med = np.median(mids, axis=0)
    return mids[np.argmin(np.linalg.norm(mids - med, axis=1))]


def cmd_seeds(_a):
    vol = _volume()
    rows = []
    for sid in compatible_ids():
        t0 = time.time()
        v, f = _window_mesh(sid)
        if not len(f):
            rows.append({"id": sid, "covers": False, "reason": "no triangles in z window"})
            print(sid, "no triangles in window", flush=True)
            continue
        s0, s1 = slice_segments(v, f, Z0), slice_segments(v, f, Z1)
        row = {"id": sid, "covers": bool(len(s0) and len(s1)),
               "segments_z0": int(len(s0)), "segments_z1": int(len(s1)),
               "length_z0": round(float(np.linalg.norm(s0[:, 1] - s0[:, 0], axis=1).sum()), 1) if len(s0) else 0.0}
        if row["covers"]:
            seed = pick_seed(s0, vol["width"], vol["height"])
            if seed is None:
                row.update(covers=False, reason="no point with a full crop inside the slice")
            else:
                row["seed"] = [int(round(seed[0])), int(round(seed[1])), Z0]
        print(sid, row.get("seed"), row["segments_z0"], row["segments_z1"],
              f"{time.time() - t0:.1f}s", flush=True)
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "seeds.json").write_text(json.dumps(rows, indent=2))
    print(f"{sum(r.get('covers', False) for r in rows)} seeds of {len(rows)} meshes")


SEED_SETS = {"main": "seeds.json", "heldout": "seeds_heldout.json", "large": "seeds_large.json"}


def _seeds(a=None):
    name = SEED_SETS[getattr(a, "seed_set", "main") or "main"]
    return [r for r in json.loads((OUT / name).read_text()) if r.get("covers")]


def _suffix(a, mode=None):
    """Separate outputs per first-slice mode and seed set; dp/main keep the original paths."""
    mode = mode or a.first_frame[0]
    s = getattr(a, "seed_set", "main") or "main"
    held = "" if s == "main" else f"_{s}"
    return ("" if mode == "dp" else f"_{mode}") + held


def cmd_heldout(_a, min_vx=500.0, target_vx=700.0):
    """A second click per mesh on the same released layer, >= min_vx from the first."""
    vol = _volume()
    rows = []
    for r in _seeds():
        v, f = _window_mesh(r["id"])
        mids = slice_segments(v, f, Z0).mean(1)
        half = CROP // 2
        ok = ((mids[:, 0] >= half) & (mids[:, 0] <= vol["width"] - half)
              & (mids[:, 1] >= half) & (mids[:, 1] <= vol["height"] - half))
        d = np.linalg.norm(mids - np.asarray(r["seed"][:2], float), axis=1)
        ok &= d >= min_vx
        row = {"id": r["id"], "covers": bool(ok.any()), "main_seed": r["seed"]}
        if ok.any():
            i = np.flatnonzero(ok)[np.argmin(np.abs(d[ok] - target_vx))]
            row["seed"] = [int(round(mids[i, 0])), int(round(mids[i, 1])), Z0]
            row["distance_from_main_vx"] = round(float(d[i]), 1)
        else:
            row["reason"] = f"no intersection point >= {min_vx} vx from the main seed"
        print(r["id"], row.get("seed"), row.get("distance_from_main_vx"), flush=True)
        rows.append(row)
    (OUT / SEED_SETS["heldout"]).write_text(json.dumps(rows, indent=2))
    print(f"{sum(r['covers'] for r in rows)} held-out seeds")


def cmd_large(a, per_mesh=4, min_vx=300.0):
    """Fresh clicks for a larger held-out test: up to per_mesh per mesh, each >= min_vx
    from the development and held-out clicks and from each other. Never used in development."""
    vol = _volume()
    used = {}
    for name in ("main", "heldout"):
        for r in _seeds(argparse.Namespace(seed_set=name)):
            used.setdefault(r["id"], []).append(np.asarray(r["seed"][:2], float))
    rng = np.random.default_rng(20261101)
    rows = []
    for r in _seeds():
        v, f = _window_mesh(r["id"])
        mids = slice_segments(v, f, Z0).mean(1)
        half = CROP // 2
        ok = ((mids[:, 0] >= half) & (mids[:, 0] <= vol["width"] - half)
              & (mids[:, 1] >= half) & (mids[:, 1] <= vol["height"] - half))
        taken = list(used.get(r["id"], []))
        for i in rng.permutation(np.flatnonzero(ok)):
            if sum(1 for row in rows if row["id"] == r["id"]) >= per_mesh:
                break
            if all(np.linalg.norm(mids[i] - t) >= min_vx for t in taken):
                taken.append(mids[i])
                k = sum(1 for row in rows if row["id"] == r["id"])
                rows.append({"id": r["id"], "key": f"{r['id']}_{k}", "covers": True,
                             "seed": [int(round(mids[i, 0])), int(round(mids[i, 1])), Z0]})
        print(r["id"], sum(1 for row in rows if row["id"] == r["id"]), "clicks", flush=True)
    (OUT / SEED_SETS["large"]).write_text(json.dumps(rows, indent=2))
    print(f"{len(rows)} large-set seeds")


def cmd_trace(a):
    """Trace every seed in every mode, seed by seed so a timeout leaves paired results;
    with --gpus N (0 = all visible), N traces run at once, one per GPU."""
    from concurrent.futures import ThreadPoolExecutor
    import queue
    if a.gpus == 0:
        import torch
        a.gpus = max(1, torch.cuda.device_count())
    free = queue.Queue()
    for g in range(a.gpus):
        free.put(g)

    def one(job):
        g = free.get()
        try:
            _trace_one(a, *job, g if a.gpus > 1 else None)
        finally:
            free.put(g)
    with ThreadPoolExecutor(a.gpus) as ex:
        list(ex.map(one, [(mode, r) for r in _seeds(a) for mode in a.first_frame]))


def _trace_one(a, mode, r, gpu):
    key = r.get("key", r["id"])
    out = OUT / f"curves{_suffix(a, mode)}" / key
    if out.exists() and len(list(out.glob("[0-9]*.jsonl"))) >= WINDOW:
        print(key, "already traced", flush=True)
        return
    if out.exists():  # interrupted run: the tracker refuses a non-empty --out
        shutil.rmtree(out)
    click = ",".join(str(c) for c in r["seed"])
    cmd = [sys.executable, "-m", "src.trace.sam3_tracker", "--slices", str(SLICES),
           "--click", click, "--window", str(WINDOW), "--crop", str(CROP),
           "--out", str(out), "--ckpt", str(ROOT / "checkpoints" / "sam3.1_multiplex.pt"),
           "--trim-memory", "--first-frame", mode]
    log = OUT / f"logs{_suffix(a, mode)}" / f"{key}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, **({"CUDA_VISIBLE_DEVICES": str(gpu)} if gpu is not None else {}))
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        code = subprocess.call(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env)
    print(mode, key, "exit", code, f"{time.time() - t0:.0f}s", flush=True)


def cmd_score(a):
    seeds = _seeds(a)
    ids = [r["id"] for r in seeds]
    indexes = {sid: SurfaceIndex(*_window_mesh(sid)) for sid in ids}
    results = []
    for r in seeds:
        sid, seed, key = r["id"], np.array(r["seed"], float), r.get("key", r["id"])
        target = indexes[sid]
        # other meshes passing within 2 voxels of the seed annotate the same
        # sheet; only meshes farther away count as a different layer
        same = {o for o in ids if o != sid and indexes[o].distances(seed[None])[0] <= 2.0}
        others = [o for o in ids if o != sid and o not in same]
        curves = sorted((OUT / f"curves{_suffix(a)}" / key).glob("[0-9]*.jsonl"))
        pts, wts, traced, statuses = [], [], 0, []
        for c in curves:
            rec = json.loads(c.read_text().splitlines()[0])
            xyz = curve_xyz(rec)
            q, w = sample_curve(xyz, 1.0)
            if len(q):
                traced += 1
                pts.append(q)
                wts.append(w)
        row = {"id": sid, "key": key, "seed": r["seed"], "same_sheet_meshes": sorted(same),
               "slices": len(curves), "slices_with_trace": traced}
        if pts:
            p, w = np.concatenate(pts), np.concatenate(wts)
            d_t = target.distances(p)
            d_o = np.min([indexes[o].distances(p) for o in others], axis=0) if others else np.full(len(p), np.inf)
            wn = w / w.sum()
            row.update({
                "arc_voxels": round(float(w.sum()), 1),
                "within2": round(float(wn[d_t <= 2].sum()), 4),
                "within5": round(float(wn[d_t <= 5].sum()), 4),
                "mean_dist": round(float((wn * d_t).sum()), 3),
                "median_dist": round(float(np.median(d_t)), 3),
                "closer_to_other_layer": round(float(wn[(d_o <= 2) & (d_t > 2)].sum()), 4),
            })
        results.append(row)
        print(json.dumps(row), flush=True)
    scored = [r for r in results if "within2" in r]
    summary = {"seeds": len(results), "seeds_with_trace": len(scored),
               "slices_traced": sum(r["slices_with_trace"] for r in results),
               "slices_total": sum(r["slices"] for r in results)}
    if scored:
        for k in ("within2", "within5", "mean_dist", "closer_to_other_layer"):
            vals = [r[k] for r in scored]
            summary[f"{k}_median"] = round(float(np.median(vals)), 4)
            summary[f"{k}_mean"] = round(float(np.mean(vals)), 4)
    (OUT / f"score{_suffix(a)}.json").write_text(json.dumps({"summary": summary, "seeds": results}, indent=2))
    print(json.dumps(summary, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["seeds", "heldout", "large", "trace", "score"])
    ap.add_argument("--first-frame", nargs="+", choices=["dp", "walk", "flow", "flowridge", "smooth1", "smooth2", "smooth4"], default=["dp"],
                    help="tracker first-slice mode(s); walk writes curves_walk/, score_walk.json")
    ap.add_argument("--seed-set", choices=sorted(SEED_SETS), default="main",
                    help="heldout: second click per mesh (seeds_heldout.json), outputs *_heldout")
    ap.add_argument("--gpus", type=int, default=1, help="trace: parallel traces, one per GPU; 0 = every visible GPU")
    a = ap.parse_args(argv)
    {"seeds": cmd_seeds, "heldout": cmd_heldout, "large": cmd_large, "trace": cmd_trace, "score": cmd_score}[a.step](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
