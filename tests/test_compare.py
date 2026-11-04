import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "reports" / "seeded_eval"


def test_baselines_start_from_the_fixed_first_slice():
    walk = EVAL / "curves_walk"
    for mode in ("flow", "flowridge"):
        for d in sorted((EVAL / f"curves_{mode}").iterdir())[:5]:
            a = json.loads((d / "00000.jsonl").read_text())["points"]
            b = json.loads((walk / d.name / "00000.jsonl").read_text())["points"]
            assert np.allclose(a, b, atol=0.01)
            assert len(list(d.glob("*.jsonl"))) == 20


def test_ridge_snap_moves_points_onto_the_bright_line():
    from scripts.baseline_flow import ridge_snap
    img = np.zeros((40, 80), np.uint8)
    img[20, :] = 255
    p = np.column_stack([np.arange(10, 70, dtype=float), np.full(60, 18.0)])
    q = ridge_snap(p, img)
    assert np.abs(q[5:-5, 1] - 20).max() < 0.5


def test_comparison_is_paired_and_counts_missing_traces_as_zero():
    c = json.loads((EVAL / "compare.json").read_text())
    for g in c["groups"].values():
        assert set(g["methods"]) == {"before", "after", "flow", "flowridge"}
        for m, v in g["vs_after"].items():
            assert v["after_better"] + v["after_worse"] + v["ties"] == g["clicks"]
            assert v["mean_gain_ci95"][0] <= v["mean_gain"] <= v["mean_gain_ci95"][1]
    assert "no trace = 0" in c["measure"]


def test_ink_drift_curve_is_on_the_known_text_sheet():
    s = json.loads((ROOT / "reports/letters/shift/summary.json").read_text())
    assert s["segment"] == "20230827161847"
    const = {c["offset_vx"]: c["corr_with_official_ink"] for c in s["shift_curve"]["constant"]}
    wob = [w["corr_with_official_ink"] for w in s["shift_curve"]["wobble"]]
    assert abs(const[0] - s["control"]["corr_with_official_ink"]) < 1e-6   # zero offset = the control
    assert all(a >= b for a, b in zip(wob, wob[1:]))                        # more wobble, fewer letters
    assert wob[0] < const[0]


def test_evaluation_page_bundle_has_comparison_and_drift():
    js = (ROOT / "viewer" / "eval-data.js").read_text(encoding="utf-8")
    e = json.loads(js.split("window.UNSCROLL_EVAL=", 1)[1].rsplit(";", 1)[0])
    assert e["compare"]["groups"]["new_clicks"]["clicks"] >= 15
    assert e["shift"]["images"]["official_ink"].startswith("eval/")


def test_tifxyz_export_follows_the_vc3d_format(tmp_path):
    import tifffile
    from scripts.export_tifxyz import export
    shape, n = export(ROOT / "reports/flatten/curves_20231210121321_kaggle", [3493, 2426], tmp_path / "s", sigma=2)
    x, y, z = (tifffile.imread(tmp_path / "s" / f"{a}.tif") for a in "xyz")
    meta = json.loads((tmp_path / "s" / "meta.json").read_text())
    assert x.dtype == np.float32 and x.shape == y.shape == z.shape == shape
    assert meta["format"] == "tifxyz" and meta["scale"] == [1.0, 1.0]
    valid = z > 0
    assert valid.sum() == n and np.all((x[~valid] == -1) & (y[~valid] == -1) & (z[~valid] == -1))
    assert np.allclose(z[valid], np.round(z[valid]))               # each row stays on its CT slice


def test_smoothing_removes_wobble_without_moving_slices():
    from scripts.smooth_traces import wobble
    from scripts.unroll3d import smooth_grid
    rng = np.random.default_rng(0)
    G = np.zeros((30, 50, 3))
    G[..., 0] = np.arange(50)[None, :]
    G[..., 1] = 100 + rng.normal(0, 1, (30, 1))                     # each slice jumps sideways
    G[..., 2] = np.arange(30)[:, None]
    assert wobble(smooth_grid(G, sigma_rows=2)) < wobble(G) / 3


def test_new_click_results_are_on_record():
    c = json.loads((EVAL / "compare.json").read_text())["groups"]["new_clicks"]
    assert c["clicks"] == 74
    a = json.loads((EVAL / "agreement.json").read_text())
    assert a["groups"]["new_clicks"]["traces"] > 50
    r = json.loads((EVAL / "reference_check.json").read_text())
    assert r["segment"] == "20231210121321" and r["old_reference_vs_2026"]["points"] > 1000


def test_smoothing_and_probe_results_are_on_record():
    s = json.loads((ROOT / "reports/letters/assisted_smooth/summary.json").read_text())
    r = [s[k]["ink_corr_with_official_all_matched"] for k in ("ours3d", "ours3d_sigma4", "ours3d_sigma8")]
    assert r == sorted(r) and r[-1] < s["control"]["corr_with_official_ink"]
    p = json.loads((ROOT / "reports/probe24/summary.json").read_text())
    assert p["volume"].startswith("20260411134726-2.400um") and len(p["runs"]) == 32
    assert all(x["exit"] == 0 for x in p["runs"])


def test_stronger_smoothing_reran_the_same_trace_and_stays_below_the_official_surface():
    a = json.loads((ROOT / "reports/letters/assisted_smooth/summary.json").read_text())
    b = json.loads((ROOT / "reports/letters/assisted_smooth_s1216/summary.json").read_text())
    assert a["distance_to_official_surface"] == b["distance_to_official_surface"]  # deterministic trace
    hi = [b[k]["ink_corr_with_official_all_matched"] for k in ("ours3d_sigma12", "ours3d_sigma16")]
    assert min(hi) > a["ours3d_sigma8"]["ink_corr_with_official_all_matched"]
    assert max(hi) < b["control"]["corr_with_official_ink"]


def test_one_click_run_is_on_record_and_not_oversold():
    # One click with no help keeps less of the trace on the sheet than the 10-click run,
    # whatever its smoothed ink score says: the page must carry both numbers.
    one = json.loads((ROOT / "reports/letters/unassisted_smooth_one/summary.json").read_text())
    ten = json.loads((ROOT / "reports/letters/assisted_smooth/summary.json").read_text())
    assert "assisted" not in one and one["reseeds"] >= 1
    assert one["distance_to_official_surface"]["within5"] < ten["distance_to_official_surface"]["within5"]
    assert set(k for k in one if k.startswith("ours3d")) == {"ours3d", "ours3d_sigma4", "ours3d_sigma8", "ours3d_sigma16"}
    page = (ROOT / "viewer/evaluate.html").read_text(encoding="utf-8")
    assert "within40um" in page and "Trace on the sheet" in page
    build = (ROOT / "scripts/build_evaluation.py").read_text(encoding="utf-8")
    assert '"within40um"' in build and "unassisted_smooth_one" in build


def test_click_every_80_run_is_shorter_so_its_score_is_not_a_ranking():
    a80 = json.loads((ROOT / "reports/letters/assisted80_smooth_a80/summary.json").read_text())
    a40 = json.loads((ROOT / "reports/letters/assisted_smooth/summary.json").read_text())
    assert len(a80["assisted"]["clicks_crop"]) == 5 and len(a40["assisted"]["clicks_crop"]) == 10
    assert a80["ours"]["image"][0] < a40["ours"]["image"][0]          # fewer slices covered
    page = (ROOT / "viewer/evaluate.html").read_text(encoding="utf-8")
    assert "Slices covered" in page and "not a ranking" in page


def test_story_page_uses_real_data_and_is_linked():
    story = (ROOT / "viewer/story.html").read_text(encoding="utf-8")
    for needle in ("data.js", "eval-data.js", "UNSCROLL_DATA", "UNSCROLL_EVAL", "per_slice", "E.shift", "new_clicks",
                   "No.</strong>"):  # the first flip card answers "read any new text?" with No
        assert needle in story, needle
    for page in ("home.html", "evaluate.html", "index.html", "read.html", "predict.html"):
        assert 'href="story.html"' in (ROOT / "viewer" / page).read_text(encoding="utf-8"), page
