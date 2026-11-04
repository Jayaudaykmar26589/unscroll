"""Fetch Scroll 1 segment .obj meshes (small) for G1 reference + window pick."""
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ("https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/"
        "paths/")
IDS = ["20230518012543", "20230518181521", "20230521113334",
       "20230611145109"]


def fetch(url, dest, timeout=300):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"have {dest.name} ({dest.stat().st_size}B)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "unscroll"})
    with urllib.request.urlopen(req, timeout=timeout) as r, \
            open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    print(f"got {dest.name} ({dest.stat().st_size}B)")


def main():
    d = ROOT / "data" / "cache" / "scroll1_segments"
    for sid in IDS:
        # pattern A: {id}/{id}.obj ; pattern B in script handles _superseded
        url = f"{BASE}{sid}/{sid}.obj"
        try:
            fetch(url, d / f"{sid}.obj")
        except Exception as e:  # noqa: BLE001
            print(f"{sid} FAILED: {str(e)[:120]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
