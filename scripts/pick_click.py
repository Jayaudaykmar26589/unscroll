"""Pick a click pixel principledly: strongest layer peak on middle row.

Loads one slice, takes the middle row, smooths, finds local maxima
above (median + k*std), returns the strongest as (x, y, z). Recorded
procedure, no eyeballing. Stdlib + numpy + tifffile.
"""
import sys
from pathlib import Path

import numpy as np
import tifffile


def pick_click(slice_path, k=1.5, smooth=5):
    img = tifffile.imread(slice_path).astype(np.float64)
    h, w = img.shape
    row = img[h // 2].copy()
    ker = np.ones(smooth) / smooth
    sm = np.convolve(row, ker, mode="same")
    thr = np.median(sm) + k * sm.std()
    peaks = []
    for x in range(1, w - 1):
        if sm[x] > thr and sm[x] >= sm[x - 1] and sm[x] >= sm[x + 1]:
            peaks.append((sm[x], x))
    assert peaks, "no layer peaks found; adjust k"
    peaks.sort(reverse=True)
    return int(peaks[0][1]), h // 2, peaks


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", required=True)
    ap.add_argument("--z", type=int, required=True)
    ap.add_argument("--k", type=float, default=1.5)
    a = ap.parse_args(argv)
    x, y, peaks = pick_click(a.slice, a.k)
    print(f"CLICK {x},{y},{a.z} (top peaks: "
          f"{[(x_) for _, x_ in peaks[:5]]})")


if __name__ == "__main__":
    main(sys.argv[1:] if len(sys.argv) > 1 else [])
