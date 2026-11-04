"""Pick outer-wrap click: first strong peak from image edge inward."""
import sys

import numpy as np
import tifffile


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", required=True)
    ap.add_argument("--z", type=int, required=True)
    ap.add_argument("--row", type=int, default=4000)
    ap.add_argument("--side", default="right", choices=("right", "left"))
    ap.add_argument("--xmax", type=int, default=6800,
                    help="ignore holder beyond this x")
    a = ap.parse_args()
    img = tifffile.imread(a.slice).astype(float)
    row = img[a.row, :a.xmax]
    if a.side == "right":
        row = row[::-1]
        flip = lambda x: len(row) - 1 - x  # noqa: E731
    else:
        flip = lambda x: x  # noqa: E731
    sm = np.convolve(row, np.ones(5) / 5, mode="same")
    thr = np.median(sm) + 2.0 * sm.std()
    for x in range(100, len(sm) - 1):
        if sm[x] > thr and sm[x] >= sm[x - 1] and sm[x] >= sm[x + 1]:
            print(f"CLICK {flip(x)},{a.row},{a.z}")
            return
    raise SystemExit("no outer peak found")


if __name__ == "__main__":
    main()
