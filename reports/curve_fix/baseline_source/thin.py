"""Zhang-Suen thinning (skeletonize) in pure numpy. No skimage needed."""

import numpy as np


def _neighbors(b):
    # roll(a, s): out[i] = old[i-s]. N needs old[i-1] -> s=+1 (axis 0).
    p2 = np.roll(b, 1, 0)  # N
    p3 = np.roll(np.roll(b, 1, 0), -1, 1)  # NE
    p4 = np.roll(b, -1, 1)  # E
    p5 = np.roll(np.roll(b, -1, 0), -1, 1)  # SE
    p6 = np.roll(b, -1, 0)  # S
    p7 = np.roll(np.roll(b, -1, 0), 1, 1)  # SW
    p8 = np.roll(b, 1, 1)  # W
    p9 = np.roll(np.roll(b, 1, 0), 1, 1)  # NW
    return p2, p3, p4, p5, p6, p7, p8, p9


def _transitions(nb):
    s = 0
    for i in range(8):
        s += (~nb[i]) & nb[(i + 1) % 8]
    return s


def skeletonize(mask):
    """Thin binary mask to 1-voxel skeleton. Returns bool array."""
    b = np.asarray(mask, dtype=bool).copy()
    b[[0, -1], :] = False
    b[:, [0, -1]] = False
    while True:
        prev = b.copy()
        for step in (0, 1):
            p2, p3, p4, p5, p6, p7, p8, p9 = _neighbors(b)
            # NOTE: bool+bool is OR in numpy; cast before counting.
            nzs = (p2.astype(np.uint8) + p3.astype(np.uint8) +
                   p4.astype(np.uint8) + p5.astype(np.uint8) +
                   p6.astype(np.uint8) + p7.astype(np.uint8) +
                   p8.astype(np.uint8) + p9.astype(np.uint8))
            cond_n = (nzs >= 2) & (nzs <= 6)
            cond_s = _transitions((p2, p3, p4, p5, p6, p7, p8, p9)) == 1
            if step == 0:
                cond = cond_n & cond_s & (~(p2 & p4 & p6)) & (~(p4 & p6 & p8))
            else:
                cond = cond_n & cond_s & (~(p2 & p4 & p8)) & (~(p2 & p6 & p8))
            b = b & (~(b & cond))
        if (b == prev).all():
            break
    return b


def prune_spurs(skel, min_len=3):
    """Remove spur branches shorter than min_len (bound <1 voxel kept)."""
    s = np.asarray(skel, dtype=bool).copy()
    for _ in range(min_len):
        p2, p3, p4, p5, p6, p7, p8, p9 = _neighbors(s)
        nzs = sum(v.astype(np.uint8)
                  for v in (p2, p3, p4, p5, p6, p7, p8, p9))
        endpoints = s & (nzs == 1)
        if not endpoints.any():
            break
        s[endpoints] = False
    return s


MAX_SKELETON_PTS = 50000


def order_by_angle(pts, center):
    """Order points by polar angle around center (spiral-layer order)."""
    ang = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(ang)]


def mask_to_curve(mask, center=None):
    """Skeleton -> angle-ordered curve. Empty if no/blob mask."""
    sk = prune_spurs(skeletonize(mask))
    ys, xs = np.nonzero(sk)
    if len(xs) == 0 or len(xs) > MAX_SKELETON_PTS:
        # empty, or a region mask rather than a curve (caller: suspect)
        return np.zeros((0, 2))
    pts = np.stack([xs, ys], 1).astype(float)
    if center is None:
        my, mx = np.nonzero(mask)
        center = (mx.mean(), my.mean())
    return order_by_angle(pts, center)
