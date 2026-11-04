"""Guard thresholds on toy curves (design Verification Plan)."""

import numpy as np

import pytest

from src.trace.guards import (LayerGuard, check_curvature, check_spacing,
                              check_winding, check_tangent_rotation)


def test_legacy_tangent_diagnostic_is_not_the_live_guard():
    assert check_tangent_rotation([0, 1, 2, 3])["pass"]
    assert check_tangent_rotation([0, 1, 4, 1])["pass"]
    r = check_tangent_rotation([0, 10, 0])
    assert not r["pass"] and r["violations"] == [2]
    with pytest.raises(ValueError, match="coordinates"):
        check_winding([0, 10, 0])


def line(y=0, n=101, start=0, stop=100):
    return np.column_stack((np.linspace(start, stop, n), np.full(n, y)))


@pytest.mark.parametrize("direction", [-1, 1])
def test_clockwise_counterclockwise_arcs_and_angle_wrap_are_normal(direction):
    theta = np.linspace(2.7, 4.3, 301) * direction
    ref = np.column_stack((100 * np.cos(theta), 100 * np.sin(theta)))
    theta2 = np.linspace(2.75, 4.25, 155) * direction
    cur = np.column_stack((100 * np.cos(theta2), 100 * np.sin(theta2)))
    result = check_winding(cur[::-1], ref)
    assert result["assessed"] and result["pass"]


def test_s_curve_rotation_reversal_and_uneven_sampling_do_not_change_identity():
    x = np.linspace(0, 100, 1001)
    ref = np.column_stack((x, 15 * np.sin(x / 15)))
    t = np.linspace(0, 1, 177)**1.6
    cur = np.column_stack((100 * t, 15 * np.sin(100 * t / 15)))
    rotation = np.array([[0., -1.], [1., 0.]])
    for a, b in ((cur, ref), (cur[::-1], ref[::-1]),
                  (cur @ rotation + 123, ref @ rotation + 123)):
        assert check_winding(a, b)["pass"]


def test_parallel_layer_jump_is_detected_even_with_perfectly_smooth_tangents():
    result = check_winding(line(20), line())
    assert result["assessed"] and not result["pass"]
    assert len(result["runs"]) == 1
    assert result["runs"][0]["length_voxels"] > 90


def test_smooth_local_switch_to_neighbor_layer_is_detected():
    p = line(n=501)
    x = p[:, 0]
    p[:, 1] = 10 * (np.tanh((x - 30) / 4) - np.tanh((x - 70) / 4))
    assert check_curvature(p)["pass"]  # no sharp kink to rescue detection
    result = check_winding(p, line())
    assert result["assessed"] and not result["pass"]
    assert result["runs"][0]["length_voxels"] > 30


@pytest.mark.parametrize("n", [51, 101, 1001])
def test_distance_boundary_and_sampling_independent_persistence(n):
    assert check_winding(line(2, n=n), line())["pass"]
    assert not check_winding(line(2.1, n=n), line())["pass"]


def test_endpoint_extension_is_not_a_normal_layer_jump():
    assert check_winding(line(start=-10, stop=110), line())["pass"]
    result = check_winding(line(start=200, stop=300), line())
    assert result["pass"] is None and not result["assessed"]
    assert result["reason"] == "insufficient_overlap"


def test_missing_reference_and_degenerate_curves_cannot_silently_pass():
    for cur, ref in ((line(), None), (np.empty((0, 2)), line()),
                     (line(), np.repeat([[1., 1.]], 3, axis=0))):
        result = check_winding(cur, ref)
        assert result["pass"] is None and not result["assessed"]


def test_rejected_and_missing_frames_do_not_reanchor_to_wrong_layer():
    guard = LayerGuard()
    assert guard.evaluate(line(), 0)["winding"]["pass"] is None
    assert guard.evaluate(line(1), 1)["status"] == "ok"
    assert guard.evaluate(line(20), 2)["status"] == "suspect"
    rejected = guard.evaluate(line(20), 3)
    assert rejected["status"] == "suspect" and rejected["reference_z"] == 1
    assert rejected["winding"]["pass"] is None
    assert rejected["winding"]["reason"] == "stale_reference_departure"
    assert guard.evaluate([], 4)["status"] == "failed"
    recovered = guard.evaluate(line(1), 5)
    assert recovered["status"] == "ok" and recovered["reference_z"] == 1


def test_slice_gap_does_not_claim_a_new_adjacent_transition_or_expand_tolerance():
    guard = LayerGuard()
    guard.evaluate(line(), 10)
    result = guard.evaluate(line(20), 13)
    assert result["winding"]["pass"] is None
    assert result["winding"]["anchor_departure"]
    assert result["winding"]["distance_voxels"] == 2
    assert guard.reference_z == 10


def test_short_excursion_is_reported_only_after_physical_persistence():
    # A 2-voxel-long branch is shorter than the 3-voxel persistence criterion.
    p = line(n=1001)
    p[:, 1] = np.where((p[:, 0] > 49) & (p[:, 0] < 51), 2.1, 0.)
    # Sharp connectors are independently rejected by the kink guard; this
    # verifies that winding persistence is measured in arc length, not samples.
    result = check_winding(p, line())
    assert result["pass"]


def test_spacing():
    assert check_spacing([10] * 12)["pass"]
    assert check_spacing([10] * 10 + [12])["pass"]  # 20% <= 30%
    r = check_spacing([10] * 10 + [14])
    assert not r["pass"] and r["violations"] == [10]


def test_curvature():
    straight = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
    assert check_curvature(straight)["pass"]
    gentle = np.array([[0, 0], [1, 0], [2, 1]])  # 45deg turn
    assert check_curvature(gentle)["pass"]
    kink = np.array([[0, 0], [1, 0], [1, 1]])  # 90deg turn
    r = check_curvature(kink)
    assert not r["pass"] and r["violations"] == [1]
