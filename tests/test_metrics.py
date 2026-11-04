"""Metric formula guards on toy arrays (design Verification Plan)."""

import numpy as np

from src.eval.metrics import g1, g2, g3


def test_g1_toy_pass():
    xs, ys = np.meshgrid(np.arange(10), np.arange(10))
    ref = np.stack([xs.ravel(), ys.ravel(),
                    np.zeros(100)]).T.astype(float)
    rng = np.random.default_rng(0)
    traced = ref + rng.normal(0, 0.5, ref.shape)  # all well within 2
    r = g1(traced, ref)
    assert r["pass"] and r["frac_within"] >= 0.90


def test_g1_p95_skewed():
    ref = np.zeros((100, 3))
    d = np.zeros(100)
    d[:80] = 0.5
    d[80:] = 4.0
    traced = np.zeros((100, 3))
    traced[:, 0] = d  # distance from origin row = d
    r = g1(traced, ref)
    assert r["p95"] == 4.0
    assert not r["pass"]  # frac 0.80 < 0.90


def test_g2_crafted_flips():
    ids = np.zeros(100, dtype=int)
    ids[10:14] = 1   # 4-slice excursion -> jump
    ids[50:53] = 2   # 3-slice excursion -> jump
    ids[80:82] = 3   # 2-slice blip -> not a jump
    r = g2(ids)
    assert r["jumps"] == 2 and r["per_100"] == 2.0 and r["pass"]


def test_g3_exact_and_cer():
    pred = list("abgd")
    true = list("abgd")
    r = g3(pred, true)
    assert r["accuracy"] == 1.0 and r["cer"] == 0.0 and r["pass"]
    r2 = g3(list("abxd"), true)
    assert r2["accuracy"] == 0.75 and r2["cer"] == 0.25 and r2["pass"]
    r3 = g3(list("xxxx"), true, excluded=[True, False, False, False])
    assert r3["n"] == 3 and r3["n_excluded"] == 1
