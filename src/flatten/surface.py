"""Surface volume: sample +-4 voxels along curve normals (streamed)."""
import json
from pathlib import Path

import numpy as np


def normals_2d(points):
    pts = np.asarray(points, dtype=float)
    d = np.gradient(pts, axis=0)
    n = np.stack([d[:, 1], -d[:, 0]], 1)
    l = np.linalg.norm(n, axis=1, keepdims=True)
    l[l == 0] = 1.0
    return n / l


def sample_surface(slice_reader, curves, half_width=4):
    """curves: list of (512,2) full-res. slice_reader(z) -> 2D array.

    Returns (Z, 512, 2*half_width+1) surface volume.
    """
    vols = []
    for z, pts in curves:
        img = slice_reader(z).astype(np.float32)
        h, w = img.shape
        n = normals_2d(pts)
        offs = np.arange(-half_width, half_width + 1)
        coords = pts[:, None, :] + n[:, None, :] * offs[None, :, None]
        xs = np.clip(coords[:, :, 0].astype(int), 0, w - 1)
        ys = np.clip(coords[:, :, 1].astype(int), 0, h - 1)
        vols.append(img[ys, xs])
    return np.stack(vols) if vols else np.zeros((0, 512, 9))
