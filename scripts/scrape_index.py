"""Fetch Vesuvius data pages, extract links. Stdlib only."""
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "link_index.json"

PAGES = [
    "https://scrollprize.org/data",
    "https://scrollprize.org/data_datasets",
    "https://scrollprize.org/tutorial3",
    "https://scrollprize.org/tutorial5",
]


def fetch(url, timeout=90):
    req = urllib.request.Request(url, headers={"User-Agent": "unscroll-dl"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def main():
    index = {}
    for url in PAGES:
        try:
            html = fetch(url)
            links = sorted(set(re.findall(r'href="([^"#]+)"', html)))
            # keep interesting ones
            keep = [l for l in links if re.search(
                r"dl\.ash|huggingface|s3:|buckets|\.zarr|fragment|scroll|"
                r"segment|label|crop|download|\.tif|\.7z|github", l,
                re.IGNORECASE)]
            index[url] = {"bytes": len(html), "links": keep}
            print(f"{url}: {len(html)} bytes, {len(keep)} kept links")
            for l in keep[:40]:
                print("   ", l)
        except Exception as e:  # noqa: BLE001
            index[url] = {"error": f"{type(e).__name__}: {e}"}
            print(f"{url}: FAILED {e}")
    OUT.write_text(json.dumps(index, indent=1), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
