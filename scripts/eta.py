"""ETA helper: listing count + local bytes/elapsed -> rate + ETA."""
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BASES = {
    "frag1": "https://dl.ash2txt.org/fragments/Frag1/"
             "PHercParis2Fr47.volpkg/working/54keV_exposed_surface/"
             "surface_volume/",
    "frag3": "https://dl.ash2txt.org/fragments/Frag3/"
             "PHercParis1Fr34.volpkg/working/54keV_exposed_surface/"
             "surface_volume/",
}


def main():
    total_files = 0
    for name, url in BASES.items():
        req = urllib.request.Request(url, headers={"User-Agent": "unscroll"})
        html = urllib.request.urlopen(req, timeout=90).read()
        html = html.decode("utf-8", "replace")
        names = sorted(set(re.findall(r'href="([^"/][^"]*)"', html)))
        tifs = [n for n in names if n.endswith((".tif", ".tiff"))]
        print(f"{name}: {len(tifs)} surface files")
        total_files += len(tifs)
    files = [f for f in (ROOT / "data" / "cache").rglob("*") if f.is_file()]
    nbytes = sum(f.stat().st_size for f in files)
    mt = [f.stat().st_mtime for f in files]
    el = max(mt) - min(mt) if len(mt) > 1 else 0
    rate = nbytes / max(el, 1)
    # assume ~99MB per remaining surface tif + frag3 extras ~90MB
    got_tifs = len([f for f in files if f.suffix == ".tif"
                    and "surface_volume" in str(f)])
    remain_gb = ((total_files - got_tifs) * 99e6 + 90e6) / 1e9
    print(f"local: {len(files)} files, {nbytes / 1e9:.2f} GB in {el:.0f}s")
    print(f"rate: {rate / 1e6:.1f} MB/s; remaining ~{remain_gb:.1f} GB -> "
          f"ETA ~{remain_gb * 1e9 / max(rate, 1) / 60:.0f} min")


if __name__ == "__main__":
    main()
