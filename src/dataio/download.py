"""Pinned public-data acquisition. Stdlib + requests(optional fallback).

Usage:
  python -m src.dataio.download --init        # probe candidates -> lock
  python -m src.dataio.download --add ROLE URL # verify HEAD size, append
  python -m src.dataio.download --fetch       # download lock entries
Lock: data/recipe.lock, JSON list of
  {artifact, role, url, version, sha256, bytes, status}
10GB-per-artifact is a hard error (names the artifact).
"""
import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "data" / "recipe.lock"
CACHE = ROOT / "data" / "cache"
MAX_BYTES = 10 * 1024 ** 3

# Candidates from public docs (scrollprize.org tutorials). Verified by
# HEAD at --init time; UNREACHABLE entries stay in the lock as record.
CANDIDATES = [
    {"artifact": "scroll1_1cm_crop", "role": "tierC_eval",
     "url": "https://dl.ash2txt.org/",
     "version": "tutorial3-link", "note": "exact 1cm path Day 1"},
    {"artifact": "ink_labels", "role": "train",
     "url": "https://scrollprize.org/tutorial5",
     "version": "ink_9um-docs", "note": "resolve exact label files Day 1"},
    {"artifact": "open_data_index", "role": "index",
     "url": "https://scrollprize.org/data",
     "version": "index", "note": "browse sample IDs Day 1"},
]


def head(url, timeout=15):
    req = urllib.request.Request(url, method="HEAD",
                                 headers={"User-Agent": "unscroll-dl"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.headers.get("Content-Length")


def cmd_init(_args):
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for c in CANDIDATES:
        e = dict(c)
        try:
            status, nbytes = head(c["url"])
            e["bytes"] = int(nbytes) if nbytes else None
            e["status"] = "REACHABLE"
            e["http"] = status
        except Exception as ex:  # noqa: BLE001
            e["bytes"] = None
            e["status"] = f"UNREACHABLE: {type(ex).__name__}"
        e["sha256"] = None
        entries.append(e)
    LOCK.write_text(json.dumps(entries, indent=1), encoding="utf-8")
    print(f"wrote {LOCK} ({len(entries)} entries)")
    for e in entries:
        print(f"  {e['artifact']}: {e['status']}")
    return 0


def cmd_add(args):
    try:
        status, nbytes = head(args.url)
    except Exception as ex:  # noqa: BLE001
        print(f"UNREACHABLE {args.url}: {ex}")
        return 2
    size = int(nbytes) if nbytes else None
    if size and size > MAX_BYTES:
        print(f"HARD ERROR: {args.url} is {size} bytes > 10GB cap; "
              "use a crop, not the full artifact.")
        return 2
    entries = json.loads(LOCK.read_text()) if LOCK.exists() else []
    entries.append({"artifact": args.name, "role": args.role,
                    "url": args.url, "version": args.version or "cli",
                    "sha256": None, "bytes": size, "status": "REACHABLE",
                    "http": status})
    LOCK.write_text(json.dumps(entries, indent=1), encoding="utf-8")
    print(f"appended {args.name} ({size} bytes)")
    return 0


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_fetch(_args):
    if not LOCK.exists():
        print("no lock; run --init first")
        return 2
    entries = json.loads(LOCK.read_text(encoding="utf-8"))
    CACHE.mkdir(parents=True, exist_ok=True)
    rc = 0
    for e in entries:
        if e.get("status") != "REACHABLE" or not e.get("bytes"):
            print(f"skip {e['artifact']}: {e.get('status')}")
            continue
        if e["bytes"] > MAX_BYTES:
            print(f"HARD ERROR: {e['artifact']} exceeds 10GB; refusing.")
            rc = 2
            continue
        dest = CACHE / Path(e["url"]).name
        print(f"fetch {e['url']} -> {dest} ...")
        try:
            urllib.request.urlretrieve(e["url"], dest)  # noqa: S310
            e["sha256"] = sha256_of(dest)
            e["status"] = "FETCHED"
        except Exception as ex:  # noqa: BLE001
            print(f"FAILED {e['artifact']}: {ex}")
            rc = 2
    LOCK.write_text(json.dumps(entries, indent=1), encoding="utf-8")
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--add", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--name", default="")
    ap.add_argument("--role", default="")
    ap.add_argument("--url", default="")
    ap.add_argument("--version", default="")
    a = ap.parse_args(argv)
    if a.init:
        return cmd_init(a)
    if a.add:
        assert a.name and a.role and a.url, "--add needs --name --role --url"
        return cmd_add(a)
    if a.fetch:
        return cmd_fetch(a)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
