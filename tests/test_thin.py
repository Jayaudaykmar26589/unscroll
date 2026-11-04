"""Regression: Zhang-Suen thinning must thin (neighbor-map orientation)."""

import numpy as np

from src.trace.thin import (_neighbors, mask_to_curve, order_by_angle,
                              skeletonize, skeleton_to_curve)


def test_neighbors_orientation():
    m = np.zeros((5, 5), bool)
    m[1, 2] = True  # north of (2,2)
    m[2, 3] = True  # east of (2,2)
    p2, _, p4, _, p6, _, p8, _ = _neighbors(m)[:8]
    assert p2[2, 2] and p4[2, 2]
    assert not p6[2, 2] and not p8[2, 2]


def test_skeletonize_thins_square():
    m = np.zeros((20, 20), bool)
    m[5:15, 5:15] = True
    sk = skeletonize(m)
    assert 0 < sk.sum() < m.sum()
    # skeleton lies strictly inside the square
    ys, xs = np.nonzero(sk)
    assert ys.min() > 5 and ys.max() < 14
    assert xs.min() > 5 and xs.max() < 14


def test_curve_from_bar():
    m = np.zeros((10, 30), bool)
    m[4:6, 5:25] = True
    pts = mask_to_curve(m, center=(15, 5))
    assert len(pts) > 10
    assert pts[:, 0].min() >= 4 and pts[:, 0].max() <= 26
    assert np.linalg.norm(np.diff(pts, axis=0), axis=1).max() <= np.sqrt(2)
    assert np.all(np.diff(pts[:, 0]) >= 0)


def assert_connected_unique(path):
    assert len(path) == len(set(map(tuple, path)))
    assert np.linalg.norm(np.diff(path, axis=0), axis=1).max() <= np.sqrt(2)


def test_hairpin_preserves_connectivity_and_both_arms():
    sk = np.zeros((30, 40), bool)
    sk[5:25, 5] = True
    sk[24, 5:31] = True
    sk[5:25, 30] = True
    curve = skeleton_to_curve(sk)
    assert_connected_unique(curve)
    assert len(curve) == sk.sum()
    assert {tuple(curve[0]), tuple(curve[-1])} == {(5, 5), (30, 5)}


def test_disconnected_components_use_seed_not_false_connecting_edges():
    sk = np.zeros((30, 50), bool)
    sk[5, 3:43] = True
    sk[20, 10:22] = True
    default = skeleton_to_curve(sk)
    clicked = skeleton_to_curve(sk, seed=(15, 20))
    assert len(default) == 40 and np.all(default[:, 1] == 5)
    assert len(clicked) == 12 and np.all(clicked[:, 1] == 20)
    assert_connected_unique(clicked)


def test_branch_retains_main_path_without_backtracking():
    sk = np.zeros((30, 50), bool)
    sk[20, 5:45] = True
    sk[17:20, 25] = True
    curve = skeleton_to_curve(sk)
    assert_connected_unique(curve)
    assert len(curve) == 40
    assert np.all(curve[:, 1] == 20)


def test_loop_has_deterministic_seam_and_keeps_whole_component():
    sk = np.zeros((30, 30), bool)
    sk[5, 5:25] = sk[24, 5:25] = True
    sk[5:25, 5] = sk[5:25, 24] = True
    curve = skeleton_to_curve(sk)
    assert_connected_unique(curve)
    assert len(curve) == sk.sum()
    np.testing.assert_array_equal(curve, skeleton_to_curve(sk))


def test_empty_and_single_pixel_masks():
    mask = np.zeros((10, 10), bool)
    assert mask_to_curve(mask).shape == (0, 2)
    mask[5, 5] = True
    np.testing.assert_array_equal(mask_to_curve(mask), [[5, 5]])


def test_order_by_angle_monotonic():
    rng = np.random.default_rng(0)
    ang = rng.uniform(-np.pi, np.pi, 200)
    pts = np.stack([10 * np.cos(ang), 10 * np.sin(ang)], 1)
    out = order_by_angle(pts, (0, 0))
    a = np.arctan2(out[:, 1], out[:, 0])
    assert (np.diff(a) >= 0).all()
