"""Bundle the real-data evaluation for the offline workbench (viewer/evaluate.html).

  python scripts/build_evaluation.py

Reads reports/seeded_eval/score.json + score_walk.json + per_slice_median.json,
reports/before_after/*.gif|png and reports/spark_review/summary*.json, copies
the media into viewer/eval/ and writes viewer/eval-data.js. No GPU or API.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "reports" / "seeded_eval"
BA = ROOT / "reports" / "before_after"
SPARK = ROOT / "reports" / "spark_review"
OUT = ROOT / "viewer"
ON_LAYER_VX = 6.0


def _scores(name):
    data = json.loads((EVAL / name).read_text())
    return data["summary"], {r["id"]: r for r in data["seeds"]}


def main():
    dp_sum, dp = _scores("score.json")
    wk_sum, wk = _scores("score_walk.json")
    seeds = []
    for sid in dp:
        a, b = dp[sid], wk.get(sid, {})
        seeds.append({
            "id": sid, "seed": a["seed"], "slices": a["slices"],
            "dp": {k: a.get(k) for k in ("within2", "within5", "median_dist", "slices_with_trace")},
            "walk": {k: b.get(k) for k in ("within2", "within5", "median_dist", "slices_with_trace")},
        })
    tp = ROOT / "reports" / "triage" / "triage.json"
    if tp.exists():
        # Reference-free review order for the corrected (walk) traces: 1 = check first.
        risk = {r["id"]: r["risk"] for r in json.loads(tp.read_text())["rows"] if r["set"] == "dev" and r["mode"] == "walk"}
        order = sorted(risk, key=lambda k: -risk[k])
        for sd in seeds:
            sd["check_order"] = order.index(sd["id"]) + 1 if sd["id"] in risk else None
    media = OUT / "eval"
    media.mkdir(parents=True, exist_ok=True)
    animations = []
    for gif in sorted(BA.glob("*.gif")):
        shutil.copyfile(gif, media / gif.name)
        still = next(BA.glob(f"{gif.stem}_z*.png"), None)
        if still:
            shutil.copyfile(still, media / still.name)
        animations.append({"id": gif.stem, "gif": f"eval/{gif.name}",
                           "still": f"eval/{still.name}" if still else None})
    spark = {}
    for variant, name in (("v1", "summary.json"), ("v2", "summary_tiles.json")):
        if (SPARK / name).exists():
            spark[variant] = json.loads((SPARK / name).read_text())
    sheets = []
    for folder in sorted((ROOT / "reports" / "flatten").glob("sheet_*")):
        if not (folder / "meta.json").exists():
            continue
        meta = json.loads((folder / "meta.json").read_text())
        files = {}
        for name in ("flat_mean.png", "flat_layers.png"):
            if (folder / name).exists():
                shutil.copyfile(folder / name, media / f"{folder.name}_{name}")
                files[name.split(".")[0]] = f"eval/{folder.name}_{name}"
        score = ROOT / "reports" / "flatten" / (folder.name.replace("sheet_", "score_") + ".json")
        extra = {}
        if score.exists():
            sc = json.loads(score.read_text())
            extra = {"score": {k: sc[k] for k in ("within5", "within2", "median_vx", "scored_slices")}}
        sheets.append({"name": folder.name, **files, "meta": meta, **extra})
    qa = {}
    for f in sorted((ROOT / "reports" / "qa").glob("*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        if rec.get("numbers_ok"):                     # only notes whose numbers passed the check
            qa[rec["seed"]] = {"text": rec["text"], "model": rec["model"]}
    heldout = {}
    if (EVAL / "score_heldout.json").exists() and (EVAL / "score_walk_heldout.json").exists():
        hd_sum, hd = _scores("score_heldout.json")
        hw_sum, hw = _scores("score_walk_heldout.json")
        on = lambda d: sum(1 for r in d.values() if r.get("median_dist") is not None and r["median_dist"] <= ON_LAYER_VX)
        heldout = {"dp": {**hd_sum, "on_layer": on(hd)}, "walk": {**hw_sum, "on_layer": on(hw)}}
    letters = {}
    L = ROOT / "reports" / "letters"
    for key, folder in (("unassisted", L), ("assisted", L / "assisted")):
        if not (folder / "summary.json").exists():
            continue
        sm = json.loads((folder / "summary.json").read_text())
        imgs = {}
        for name in ("official_ink", "control_ink", "ours_ink", "ours_ct", "control3d_ink", "ours3d_ink", "ours3d_ct"):
            if (folder / f"{name}.png").exists():
                shutil.copyfile(folder / f"{name}.png", media / f"letters_{key}_{name}.png")
                imgs[name] = f"eval/letters_{key}_{name}.png"
        letters[key] = {"images": imgs, "traced_slices": sm.get("traced_slices"),
                        "window_z": sm["window"]["z_range"], "distance": sm.get("distance_to_official_surface"),
                        "control_corr": sm["control"]["corr_with_official_ink"],
                        "calibration_r": sm["calibration"]["corr_zarr_vs_official"],
                        "clicks": len(sm.get("assisted", {}).get("clicks_crop", [])) or 1,
                        "control3d": sm.get("control_own_normals"), "ours3d": sm.get("ours3d")}
    triage = None
    tp = ROOT / "reports" / "triage" / "triage.json"
    if tp.exists():
        t = json.loads(tp.read_text())
        triage = {k: t[k] for k in ("dev", "heldout", "rule", "note")}
        sp = json.loads((ROOT / "reports/spark_review/summary.json").read_text())["summary"]
        triage["spark"] = {"review_load": sp["review_load"], "catch_rate": sp["wrong_layer_catch_rate"], "traces": sp["traces"]}
    compare = json.loads((EVAL / "compare.json").read_text()) if (EVAL / "compare.json").exists() else None
    shift = None
    sf = ROOT / "reports" / "letters" / "shift" / "summary.json"
    if sf.exists():                                   # scripts/letters_kaggle.py --shift-curve
        sc = json.loads(sf.read_text())["shift_curve"]
        imgs = {}
        for name in ("official_ink", "shift_0vx_ink", "shift_4vx_ink", "shift_7vx_ink"):
            f = sf.parent / f"{name}.png"
            if f.exists():
                shutil.copyfile(f, media / f"letters_shift_{name}.png")
                imgs[name] = f"eval/letters_shift_{name}.png"
        shift = {"constant": sc["constant"], "wobble": sc["wobble"], "images": imgs}
    smooth = None
    sm = ROOT / "reports" / "letters" / "assisted_smooth" / "summary.json"
    if sm.exists():                                   # letters_kaggle.py --assist-every 40 --sigmas 2 4 8
        d = json.loads(sm.read_text())
        rows, imgs = [], {}
        # --tag s1216 re-ran the same (deterministic) assisted trace at σ 12 and 16
        s16 = sm.parent.parent / "assisted_smooth_s1216" / "summary.json"
        parts = [(d, sm.parent)] + ([(json.loads(s16.read_text()), s16.parent)] if s16.exists() else [])
        for src, folder in parts:
            for key in sorted((k for k in src if k.startswith("ours3d") and isinstance(src[k], dict)),
                              key=lambda k: src[k].get("sigma_rows", 2.0)):
                rows.append({k: src[key][k] for k in ("sigma_rows", "ink_corr_with_official_all_matched",
                                                       "ink_corr_with_official_where_within_5vx", "snap_shift_vx")})
                f = folder / f"{key}_ink.png"
                if f.exists():
                    shutil.copyfile(f, media / f"smooth_{key}_ink.png")
                    imgs[f"{src[key]['sigma_rows']:g}"] = f"eval/smooth_{key}_ink.png"
        rows.sort(key=lambda r: r["sigma_rows"])
        shutil.copyfile(sm.parent / "official_ink.png", media / "smooth_official_ink.png")
        smooth = {"rows": rows, "images": imgs, "official": "eval/smooth_official_ink.png",
                  "control": d["control"]["corr_with_official_ink"], "distance": d.get("distance_to_official_surface")}
    probe = None
    pf = ROOT / "reports" / "probe24" / "summary.json"
    if pf.exists():                                   # scripts/probe24_kaggle.py on Kaggle
        d = json.loads(pf.read_text())
        imgs = {}
        for L in (0, 1):
            f = pf.parent / f"overlay_L{L}.png"
            if f.exists():
                shutil.copyfile(f, media / f"probe24_overlay_L{L}.png")
                imgs[str(L)] = f"eval/probe24_overlay_L{L}.png"
        probe = {k: d[k] for k in ("volume", "layer", "z0", "window", "crop", "minutes", "median_by_level_and_mode")}
        probe.update(clicks=len(d["clicks"]), runs=len(d["runs"]), completed=sum(r["exit"] == 0 for r in d["runs"]), images=imgs)
    # cost of reading: human clicks vs ink recovered on Unscroll's own trace of the known-text sheet
    reading = []
    LT = ROOT / "reports" / "letters"
    runs = {"one click (no help)": ["unassisted_smooth_one"],
            "a click every 80 slices": ["assisted80_smooth_a80"],
            "a click every 40 slices": ["assisted", "assisted_smooth", "assisted_smooth_s1216"]}
    for label, folders in runs.items():
        pts, clicks, dist, slices = {}, None, None, None
        for name in folders:
            f = LT / name / "summary.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            clicks = len(d.get("assisted", {}).get("clicks_crop", [])) or 1
            dist = d.get("distance_to_official_surface") or dist
            slices = d.get("ours", {}).get("image", [None])[0] or slices   # unrolled rows = slices traced
            for k, v in d.items():
                if k.startswith("ours3d") and isinstance(v, dict) and "ink_corr_with_official_all_matched" in v:
                    pts[float(v.get("sigma_rows", 2.0))] = round(v["ink_corr_with_official_all_matched"], 4)
        if pts:
            reading.append({"run": label, "clicks": clicks, "by_sigma": {f"{k:g}": v for k, v in sorted(pts.items())},
                            "median_um": round(dist["median_vx"] * 7.91, 1) if dist else None,
                            "within40um": round(dist["within5"], 3) if dist else None, "slices": slices})
    bundle = {"reading_cost": reading, "smooth": smooth, "probe24": probe, "compare": compare, "shift": shift, "triage": triage, "on_layer_vx": ON_LAYER_VX, "heldout": heldout, "letters": letters, "summary": {"dp": dp_sum, "walk": wk_sum},
              "seeds": seeds, "animations": animations, "spark": spark, "sheets": sheets, "qa": qa}
    (OUT / "eval-data.js").write_text(
        "window.UNSCROLL_EVAL=" + json.dumps(bundle, allow_nan=False) + ";\n", encoding="utf-8")
    pe = ROOT / "reports" / "predict" / "eval.json"
    if pe.exists():
        ev = json.loads(pe.read_text(encoding="utf-8"))
        sp = ev.get("spark")
        pred = {"corpus": ev["corpus"], "corpus_sentences": ev["corpus_sentences"],
                "lookup": {k: v for k, v in ev["lookup"].items() if k != "failures"},
                "spark": {k: v for k, v in sp.items() if k != "rows"} if sp else None}
        (OUT / "predict-data.js").write_text(
            "window.UNSCROLL_PREDICT=" + json.dumps(pred, ensure_ascii=False, allow_nan=False) + ";\n", encoding="utf-8")
    print(f"Built {OUT / 'evaluate.html'} data: {len(seeds)} seeds, {len(animations)} animations, "
          f"Spark variants {sorted(spark)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
