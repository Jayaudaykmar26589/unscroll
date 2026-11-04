"""Zhang-Suen thinning (skeletonize) in pure numpy. No skimage needed."""

from collections import deque

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
    """Thin binary mask to 1-voxel skeleton. Returns bool array.

    Thinning is local (8-neighbourhood) and everything outside the mask is
    False, so it runs on the bounding box plus a 1-pixel False border; the
    result is identical to thinning the full frame, only faster.
    """
    b = np.asarray(mask, dtype=bool).copy()
    b[[0, -1], :] = False
    b[:, [0, -1]] = False
    rows, cols = np.flatnonzero(b.any(1)), np.flatnonzero(b.any(0))
    if not len(rows):
        return b
    r0, r1, c0, c1 = rows[0] - 1, rows[-1] + 2, cols[0] - 1, cols[-1] + 2
    out = np.zeros_like(b)
    out[r0:r1, c0:c1] = _thin(b[r0:r1, c0:c1])
    return out


def _thin(b):
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


def _pixel_graph(skel):
    """8-connected graph, without diagonal shortcuts around occupied corners."""
    ys, xs = np.nonzero(skel)
    pixels = set(zip(xs.tolist(), ys.tolist()))
    graph = {}
    for x, y in sorted(pixels):
        adjacent = []
        for dx, dy in ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                       (0, 1), (1, -1), (1, 0), (1, 1)):
            q = (x + dx, y + dy)
            if q not in pixels:
                continue
            if dx and dy and ((x + dx, y) in pixels or (x, y + dy) in pixels):
                continue
            adjacent.append(q)
        graph[x, y] = adjacent
    return graph


def _breadth_first(graph, start):
    parents = {start: None}
    queue = deque([start])
    last = start
    while queue:
        last = queue.popleft()
        for q in graph[last]:
            if q not in parents:
                parents[q] = last
                queue.append(q)
    return last, parents


def skeleton_to_curve(skel, seed=None):
    """Return one connected simple path, never concatenate separate branches.

    Select the component nearest seed (x,y), or the largest component without
    a seed. For trees use a two-sweep diameter; for branched cyclic graphs this
    is a conservative shortest-path approximation. Simple loops retain all
    pixels, with a deterministic seam. Output has no repeated pixels.
    """
    graph = _pixel_graph(skel)
    if not graph or len(graph) > MAX_SKELETON_PTS:
        return np.zeros((0, 2))
    unseen, components = set(graph), []
    while unseen:
        start = min(unseen)
        _, parents = _breadth_first(graph, start)
        component = set(parents)
        unseen.difference_update(component)
        components.append(component)
    if seed is None:
        component = min(components, key=lambda c: (-len(c), min(c)))
    else:
        sx, sy = seed
        component = min(components, key=lambda c: (
            min((x - sx)**2 + (y - sy)**2 for x, y in c), -len(c), min(c)))
    start = min(component)
    if all(len(graph[p]) == 2 for p in component):
        path, seen = [start], {start}
        while True:
            candidates = [p for p in graph[path[-1]] if p not in seen]
            if not candidates:
                break
            path.append(candidates[0])
            seen.add(path[-1])
    else:
        end, _ = _breadth_first(graph, start)
        other, parents = _breadth_first(graph, end)
        path = [other]
        while parents[path[-1]] is not None:
            path.append(parents[path[-1]])
        if path[-1] < path[0]:
            path.reverse()
    return np.asarray(path, dtype=float)


def mask_to_curve(mask, center=None, seed=None):
    """Skeleton -> connected path. center is retained for API compatibility.

    A polar center cannot define adjacency; use seed to select a component.
    Branches are reduced to a main path rather than joined by fictitious edges.
    """
    sk = skeletonize(mask)
    ys, xs = np.nonzero(sk)
    if len(xs) == 0 or len(xs) > MAX_SKELETON_PTS:
        # empty, or a region mask rather than a curve (caller: suspect)
        return np.zeros((0, 2))
    return skeleton_to_curve(sk, seed=seed)
