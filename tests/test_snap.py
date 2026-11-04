import numpy as np
import pytest
import tifffile

from src.trace.snap import continuous_snap
from src.trace.sam3_tracker import ridge_snap, fullres_ridge_snap, postprocess_mask


def straight(y=25., n=80):
    return np.column_stack((np.arange(10, 10 + n, dtype=float), np.full(n, y)))


def assert_continuous(before, after, span):
    assert after.shape == before.shape
    assert np.isfinite(after).all()
    old = np.diff(before, axis=0)
    new = np.diff(after, axis=0)
    lengths = np.linalg.norm(old, axis=1)
    assert np.all(np.linalg.norm(new, axis=1) <= np.maximum(1.25 * lengths + .25, 1) + 1e-9)
    assert np.all((new * old).sum(1) >= .1 * lengths**2 - 1e-9)
    assert np.max(np.linalg.norm(after - before, axis=1)) <= span + 1e-9


def test_alternating_ridges_cannot_make_independent_point_jumps():
    p = straight()
    image = np.zeros((60, 110))
    for i, x in enumerate(p[:, 0].astype(int)):
        image[25 + (9 if i % 2 else -9), x] = 1
    out = ridge_snap(p, image, span=10)
    assert_continuous(p, out, 10)
    assert np.linalg.norm(np.diff(out, axis=0), axis=1).max() < 2
    # No alternating layer selection, even with adversarial high contrast.
    assert np.max(np.abs(np.diff(out[:, 1]))) < 1


def test_coherent_offset_ridge_is_actually_followed():
    p = straight()
    image = np.zeros((60, 110))
    image[30, :] = 1
    out = ridge_snap(p, image, span=10)
    assert_continuous(p, out, 10)
    np.testing.assert_allclose(out[:, 1], 30, atol=.1)


def test_short_evidence_gap_is_bridged_without_switching_layers():
    p = straight()
    image = np.zeros((60, 110), bool)
    image[29, :] = True
    image[29, 40:45] = False
    image[15, 40:45] = True
    out = continuous_snap(p, image, span=12, ridge=True)
    assert_continuous(p, out, 12)
    assert np.all(out[:, 1] > 26)


def test_fullres_wrapper_has_same_continuity_guarantee(tmp_path):
    p = straight()
    image = np.zeros((60, 110), np.uint16)
    for i, x in enumerate(p[:, 0].astype(int)):
        image[25 + (10 if i % 2 else -10), x] = 1000
    path = tmp_path / "slice.tif"
    tifffile.imwrite(path, image)
    out = fullres_ridge_snap(p, path)
    assert_continuous(p, out, 12)


@pytest.mark.parametrize("ridge", [False, True])
def test_absent_evidence_leaves_curve_unchanged(ridge):
    p = straight()
    np.testing.assert_array_equal(continuous_snap(p, np.zeros((60, 110)), ridge=ridge), p)


def test_curved_path_keeps_order_and_stays_in_search_corridor():
    t = np.linspace(-1.2, 1.2, 100)
    p = np.column_stack((45 + 30 * np.sin(t), 10 + 30 * np.cos(t)))
    image = np.random.default_rng(0).uniform(size=(70, 100))
    out = ridge_snap(p, image)
    assert_continuous(p, out, 10)


def test_border_and_short_degenerate_curves_are_safe():
    p = straight(y=0)
    image = np.random.default_rng(0).uniform(size=(60, 110))
    out = ridge_snap(p, image)
    assert_continuous(p, out, 10)
    assert (out >= 0).all()
    for short in (np.empty((0, 2)), p[:1], p[:2], np.repeat(p[:1], 5, axis=0)):
        np.testing.assert_array_equal(ridge_snap(short, image), short)
    with pytest.raises(ValueError, match="span"):
        ridge_snap(p, image, span=-1)


def test_two_stage_snapping_bounds_compose_in_fullres_units(tmp_path):
    p = straight(n=70)
    image = np.random.default_rng(1).uniform(size=(60, 110))
    small = ridge_snap(p, image)
    full = small * 2 + [20, 20]
    path = tmp_path / "slice.tif"
    tifffile.imwrite(path, np.random.default_rng(2).integers(0, 1000, (160, 260), dtype=np.uint16))
    out = fullres_ridge_snap(full, path)
    assert_continuous(full, out, 12)
    assert np.linalg.norm(np.diff(out, axis=0), axis=1).max() < 5


def test_live_postprocessing_keeps_raw_orientation_and_follows_coherent_ridge(tmp_path):
    mask = np.zeros((60, 110), bool)
    mask[24:27, 10:90] = True
    gray = np.zeros_like(mask, dtype=np.uint8)
    gray[30, :] = 255
    full = np.zeros((160, 260), dtype=np.uint16)
    full[80, :] = 1000
    path = tmp_path / "slice.tif"
    tifffile.imwrite(path, full)
    _, previous = postprocess_mask(mask, gray, [2, 2, 20, 20], path, seed=(40, 25))
    previous = previous[::-1]
    out, raw = postprocess_mask(mask, gray, [2, 2, 20, 20], path,
                                seed=(40, 25), previous=previous)
    np.testing.assert_array_equal(raw, previous)
    assert len(out) >= len(raw)  # image path samples every source voxel
    np.testing.assert_allclose(out[:, 1], 80, atol=.1)
    assert out[0, 0] > out[-1, 0]
    assert np.linalg.norm(np.diff(out, axis=0), axis=1).max() < 5
