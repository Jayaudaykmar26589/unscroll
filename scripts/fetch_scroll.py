"""Fetch Scroll 1 window (slices 6100-6200) for Tier A/B tracing eval.

Threaded, .part + size-verify, skips complete files. ~12.8GB.
Physical z overlap alone does not establish a matching reference surface.
Use catalog_meshes.py and validate_mesh_reference.py for XYZ-aware evaluation.
"""
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "unscroll-dl"}
BASE = ("http://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/"
        "volumes/20230205180739/")
Z0, Z1 = 6100, 6200
WORKERS = 6


def head_size(url, timeout=90):
    req = urllib.request.Request(url, method="HEAD", headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        n = r.headers.get("Content-Length")
        return int(n) if n else None


def fetch_one(job):
    url, dest = job
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        expect = head_size(url)
    except Exception:  # noqa: BLE001
        expect = None
    if dest.exists() and expect and dest.stat().st_size == expect:
        return ("have", dest.name, expect)
    if dest.exists():
        dest.unlink()
    tmp = dest.with_suffix(".part")
    req = urllib.request.Request(url, headers=UA)
    total = 0
    with urllib.request.urlopen(req, timeout=1200) as r, \
            open(tmp, "wb") as f:
        expect = expect or int(r.headers.get("Content-Length") or 0) or None
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
    # a server that closes early ends the stream without an error; never keep
    # a short file as if it were complete
    if expect is None or total != expect:
        tmp.unlink()
        raise IOError(f"{dest.name}: got {total} bytes, expected {expect}")
    tmp.replace(dest)
    return ("got", dest.name, total)


def main():
    d = ROOT / "data" / "cache" / "scroll1_6100_6200"
    jobs = [(f"{BASE}{z:05d}.tif", d / f"{z:05d}.tif")
            for z in range(Z0, Z1)]
    print(f"{len(jobs)} slices queued", flush=True)
    total = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, (how, name, nbytes) in enumerate(ex.map(fetch_one, jobs)):
            total += nbytes or 0
            if i % 5 == 0 or how == "got":
                print(f"  [{i + 1}/{len(jobs)}] {how} {name} "
                      f"({total / 1e9:.1f}GB)", flush=True)
    print(f"TOTAL {total / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
