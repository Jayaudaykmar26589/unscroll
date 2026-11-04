"""Ink patch dataset: 16-deep surface-centered patches + ink labels.

Patch: (16, 128, 128) float32 normalized, label (128, 128) {0,1}.
50% centered on ink pixels, 50% on footprint background. Streams slices
via tifffile (no full-volume RAM).
"""
import json
import random
from pathlib import Path

import functools

import numpy as np
import tifffile
import torch
from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
DEPTH = 16
SIZE = 128


@functools.lru_cache(maxsize=132)  # whole frag1+frag3 stacks fit in RAM
def _read_slice(path_str):
    return tifffile.imread(path_str)


class InkPatches(Dataset):
    def __init__(self, frag, n=2000, seed=0):
        self.frag = Path(frag)
        stats = json.loads((ROOT / "data" / "norm_stats.json").read_text())
        s = stats[self.frag.name]
        self.mean, self.std = s["mean"], s["std"]
        self.tifs = sorted((self.frag / "surface_volume").glob("*.tif*"))
        self.nz = len(self.tifs)
        self.zs = np.load(str(self.frag / "zsurf.npy"))  # stride-4 map
        self.st = 4
        self.ink = (np.array(Image.open(self.frag / "inklabels.png")) > 0)
        ink = self.ink
        mask = np.array(Image.open(self.frag / "mask.png")) > 0
        self.inkys, self.inkxs = np.nonzero(ink)
        bg = mask & ~ink
        self.bgys, self.bgxs = np.nonzero(bg)
        self.H, self.W = ink.shape
        self.n = n
        self.rng = random.Random(seed)

    def __len__(self):
        return self.n

    def _patch(self, yc, xc):
        zc = int(round(float(self.zs[yc // self.st, xc // self.st])))
        z0 = min(max(zc - DEPTH // 2, 0), self.nz - DEPTH)
        y0 = min(max(yc - SIZE // 2, 0), self.H - SIZE)
        x0 = min(max(xc - SIZE // 2, 0), self.W - SIZE)
        vol = np.stack([_read_slice(str(self.tifs[z]))[y0:y0 + SIZE, x0:x0 + SIZE]
                        .astype(np.float32) for z in range(z0, z0 + DEPTH)])
        vol = np.clip((vol - self.mean) / self.std, -3, 3)
        return vol, (y0, x0)

    def __getitem__(self, i):
        if self.rng.random() < 0.5:
            j = self.rng.randrange(len(self.inkys))
            yc, xc = int(self.inkys[j]), int(self.inkxs[j])
        else:
            j = self.rng.randrange(len(self.bgys))
            yc, xc = int(self.bgys[j]), int(self.bgxs[j])
        vol, (y0, x0) = self._patch(yc, xc)
        lab = self.ink[y0:y0 + SIZE, x0:x0 + SIZE].astype(np.float32)
        return torch.from_numpy(vol), torch.from_numpy(lab)
