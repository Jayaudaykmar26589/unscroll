import numpy as np

from scripts import triage as T


def rows(pairs):
    return [{"risk": r, "on_layer": ok} for r, ok in pairs]


def test_perfect_ranking():
    rs = rows([(0.9, False), (0.8, False), (0.2, True), (0.1, True)])
    assert T.auc(rs) == 1.0
    pts = T.curve(rs)
    assert T.review_needed(pts, 1.0) == 0.5          # both bad traces in the first half
    assert T.caught_at(pts, 0.25) == 0.5


def test_inverted_ranking():
    rs = rows([(0.1, False), (0.9, True)])
    assert T.auc(rs) == 0.0
    assert T.review_needed(T.curve(rs), 1.0) == 1.0


def test_rank_risk_directions():
    batch = [
        {"click_offset": 0.0, "ridge_contrast": 0.4, "mode_disagree": 0.0},   # safe on every signal
        {"click_offset": 9.0, "ridge_contrast": 0.0, "mode_disagree": 9.0},   # risky on every signal
        {"click_offset": 3.0, "ridge_contrast": float("nan"), "mode_disagree": 3.0},
    ]
    r = T.rank_risk(batch, T.DEV_SELECTED)
    assert r[1] > r[2] > r[0]


def test_ridge_contrast_prefers_bright_core():
    img = np.zeros((40, 40), np.uint16)
    img[20, :] = 1000                                  # a bright horizontal sheet
    T._slices[-1] = img
    on = np.stack([np.arange(5, 35), np.full(30, 20.0)], 1)
    off = on + [0, 3]
    assert T.ridge_contrast(on, -1) > 0.9
    assert T.ridge_contrast(off, -1) < 0.1
