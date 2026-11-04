"""Day-1 fetch: Fragment 1+3 final outputs via plain HTTPS.

Threaded (8 workers), .part files + atomic rename, HEAD size verify.
Safe to re-run: complete files (size match) are skipped.
"""
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "unscroll-dl"}
WORKERS = 8

FRAGS = {
    "frag1": "https://dl.ash2txt.org/fragments/Frag1/"
             "PHercParis2Fr47.volpkg/working/54keV_exposed_surface/",
    "frag3": "https://dl.ash2txt.org/fragments/Frag3/"
             "PHercParis1Fr34.volpkg/working/54keV_exposed_surface/",
}
EXTRA = ["ir.png", "inklabels.png", "result.tif", "mask.png"]


def get(url, timeout=120):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def listing(dir_url):
    html = get(dir_url).decode("utf-8", "replace")
    return sorted(set(re.findall(r'href="([^"/][^"]*)"', html)))


def head_size(url, timeout=60):
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
    if dest.exists() and expect and dest.stat().st_size != expect:
        print(f"  REDO {dest.name} (had {dest.stat().st_size}, want "
              f"{expect})", flush=True)
        dest.unlink()
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers=UA)
    total = 0
    with urllib.request.urlopen(req, timeout=900) as r, \
            open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
    tmp.replace(dest)
    return ("got", dest.name, total)


def main():
    jobs = []
    for name, base in FRAGS.items():
        d = ROOT / "data" / "cache" / name
        for f in EXTRA:
            jobs.append((base + f, d / f))
        try:
            names = [n for n in listing(base + "surface_volume/")
                     if n.endswith((".tif", ".tiff", ".png"))]
        except Exception as e:  # noqa: BLE001
            print(f"{name} surface_volume listing FAILED: {e}")
            names = []
        print(f"{name}: {len(names)} surface files, "
              f"{len(jobs)} jobs queued", flush=True)
        for n in names:
            jobs.append((base + "surface_volume/" + n,
                         d / "surface_volume" / n))
    total = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, (how, name, nbytes) in enumerate(ex.map(fetch_one, jobs)):
            total += nbytes or 0
            if i % 10 == 0 or how == "got":
                print(f"  [{i + 1}/{len(jobs)}] {how} {name}", flush=True)
    print(f"TOTAL {total / 1e9:.2f} GB across {len(jobs)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
