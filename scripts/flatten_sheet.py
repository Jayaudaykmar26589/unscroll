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
from src.trace.slices import resolve_slice  # noqa: E402


def flatten(curve_dir, slice_dir, out):
    """Flatten using the retained row-to-source mapping, including skipped z."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta_path, shape = anchor_curves(curve_dir, str(out / "sheet.json"))
    print("sheet:", shape, meta_path, flush=True)
    sheet = np.load(str(out / "sheet.npy"))
    tifs = sorted(Path(slice_dir).glob("*.tif*"))
    log = json.loads(Path(meta_path).read_text())["shifts"]
    sources = [resolve_slice(rec, tifs) for rec in log]
    if not len(sheet):
        raise ValueError("no usable curves to flatten")
    curves = [(r, sheet[r]) for r in range(sheet.shape[0])]

    def reader(row):
        return tifffile.imread(sources[row])

    vol = sample_surface(reader, curves)
    print("surface:", vol.shape, vol.dtype, flush=True)
    np.save(str(out / "surface.npy"), vol)
    flat = vol.mean(axis=2)
    lo, hi = np.percentile(flat, (1, 99))
    u8 = np.clip((flat - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(u8).save(out / "flat_preview.png")
    return vol


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", required=True)
    ap.add_argument("--slices", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    flatten(a.curves, a.slices, a.out)
    print("saved flat_preview.png", flush=True)


if __name__ == "__main__":
    main()
