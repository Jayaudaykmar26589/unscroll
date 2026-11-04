import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.synth_make import main as make  # noqa: E402
from scripts.synth_score import score  # noqa: E402
from src.synth.scroll import Scroll, SynthConfig  # noqa: E402


def _small(tmp_path, crush=0, view="open"):
    out = tmp_path / f"c{crush}_{view}"
    make(["--out", str(out), "--crush", str(crush), "--view", view, "--slices", "3"])
    return out, json.loads((out / "meta.json").read_text())


def _write_curves(curves, truth, theta, half_width=0.08):
    """A trace along the sheet at spiral angle theta (+- half_width rad)."""
    curves.mkdir()
    for z in range(3):
        x, y, t = truth[f"x_{z:05d}"], truth[f"y_{z:05d}"], truth[f"t_{z:05d}"]
        sel = np.flatnonzero(np.abs(t - theta) < half_width)[::10]
        pts = [[float(x[i]), float(y[i])] for i in sel]
        rec = {"z": z, "source_z": z, "source_file": f"{z:05d}.tif", "points": pts}
        (curves / f"{z:05d}.jsonl").write_text(json.dumps(rec) + "\n")


def test_click_is_window_centre_on_sheet(tmp_path):
    out, meta = _small(tmp_path)
    assert len(list(out.glob("*.tif"))) == 3
    truth = np.load(out / "truth.npz")
    x, y, t = truth["x_00000"], truth["y_00000"], truth["t_00000"]
    cx, cy, _ = meta["click"]
    assert abs(cx - 512) <= 2 and abs(cy - 512) <= 2
    d = np.hypot(x - cx, y - cy)
    assert d.min() < 1.0
    assert abs(t[np.argmin(d)] - meta["click_theta"]) < 0.01


def test_window_shows_large_radius_layers(tmp_path):
    out, meta = _small(tmp_path)
    truth = np.load(out / "truth.npz")
    t = truth["t_00000"]
    # a 1024 px window near radius 900 px spans many windings, each a short arc
    windings = np.unique(np.floor(t / (2 * np.pi)))
    assert len(windings) > 30


def test_perfect_trace_scores_on_target(tmp_path):
    out, meta = _small(tmp_path)
    curves = tmp_path / "perfect"
    _write_curves(curves, np.load(out / "truth.npz"), meta["click_theta"])
    s = score(out, curves)
    assert s["on_target_mean"] > 0.99 and s["jumped_mean"] == 0.0


def test_neighbour_winding_counts_as_jump(tmp_path):
    out, meta = _small(tmp_path)
    shifted = tmp_path / "next"
    _write_curves(shifted, np.load(out / "truth.npz"), meta["click_theta"] + 2 * np.pi)
    s = score(out, shifted)
    assert s["on_target_mean"] < 0.05 and s["jumped_mean"] > 0.95


def test_crush_closes_gaps_on_flattened_side():
    def gap(crush):
        s = Scroll(SynthConfig(slices=1, crush=crush, view_angle=np.pi / 2))
        line = s.centerline(0)
        c = s.click_theta
        a = np.stack([line["x"], line["y"]], 1)[np.abs(line["t"] - c) < 0.02]
        b = np.stack([line["x"], line["y"]], 1)[np.abs(line["t"] - c - 2 * np.pi) < 0.02]
        return np.sqrt(((a[:, None] - b[None]) ** 2).sum(-1)).min()
    open_gap, crushed_gap = gap(0), gap(4)
    assert crushed_gap < open_gap
    assert crushed_gap <= SynthConfig().thickness + 2.0
