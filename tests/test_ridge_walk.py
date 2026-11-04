import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.trace.image_ridge import _walk, trace_image_ridge  # noqa: E402


def _two_sheets(n=200, width=120, gap=20, dim=0.4, bright=0.9):
    """Horizontal sheets: the clicked one (row 50) is dimmer than its neighbour.

    The neighbour is thick enough to set the 95th-percentile contrast, so the
    clicked sheet keeps a real cost and a global optimum prefers the neighbour.
    """
    img = np.full((width, n), 0.1)
    img[49:52, :] = dim
    img[46 + gap:55 + gap, :] = bright
    return img


def test_walk_keeps_the_clicked_sheet_where_dp_jumps():
    img = _two_sheets()
    guide = np.column_stack([np.arange(10, 190, 2.0), np.full(90, 60.0)])
    seed = (100.0, 50.0)
    mask = np.ones(img.shape, bool)
    dp, _ = trace_image_ridge(img, mask, (1.0, 1.0, 0, 0), guide, seed, first_frame="dp")
    walk, info = trace_image_ridge(img, mask, (1.0, 1.0, 0, 0), guide, seed, first_frame="walk")
    assert info["success"]
    far = np.abs(dp[:, 0] - seed[0]) > 40
    assert np.median(dp[far, 1]) >= 46 + 20                # DP moved onto the brighter neighbour band
    assert np.abs(walk[:, 1] - 50).max() <= 1.5            # walk stayed on the clicked sheet


def test_walk_stops_where_the_sheet_fades():
    cost = np.ones((60, 9))
    cost[:, 4] = 0.1
    cost[40:, 4] = 0.95                                     # sheet disappears after row 40
    path = _walk(cost, 10, 4, lost=0.9, patience=5)
    kept = np.flatnonzero(np.isfinite(path))
    assert kept.min() == 0 and kept.max() < 40
    assert np.all(path[kept] == 4)
