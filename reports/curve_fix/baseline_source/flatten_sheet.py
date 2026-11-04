"""Flatten driver: curves -> anchored sheet -> surface volume -> preview."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import tifffile  # noqa: E402
from PIL import Image  # noqa: E402

from src.flatten.anchor import anchor_curves  # noqa: E402
from src.flatten.surface import sample_surface  # noqa: E402


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", required=True)
    ap.add_argument("--slices", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta_path, shape = anchor_curves(a.curves, str(out / "sheet.json"))
    print("sheet:", shape, meta_path, flush=True)
    sheet = np.load(str(out / "sheet.npy"))
    tifs = sorted(Path(a.slices).glob("*.tif*"))
    log = json.loads(Path(meta_path).read_text())["shifts"]
    # sheet row r <-> local slice index = position among jsonls; map via shifts log order
    curves = [(r, sheet[r]) for r in range(sheet.shape[0])]

    def reader(r):
        return tifffile.imread(tifs[r])

    vol = sample_surface(reader, curves)
    print("surface:", vol.shape, vol.dtype, flush=True)
    np.save(str(out / "surface.npy"), vol)
    flat = vol.mean(axis=2)
    lo, hi = np.percentile(flat, (1, 99))
    u8 = np.clip((flat - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(u8).save(out / "flat_preview.png")
    print("saved flat_preview.png", flush=True)


if __name__ == "__main__":
    main()
