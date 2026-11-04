"""Resolve exact data artifact URLs + sizes. Stdlib only."""
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "artifact_sizes.json"

HF_API = "https://huggingface.co/api/datasets/scrollprize/ink_9um"
DL_FILES = [
    "https://dl.ash2txt.org/datasets/grand-prize-banner-region/gp_meshes.7z",
    "https://dl.ash2txt.org/datasets/grand-prize-banner-region/README.txt",
    "https://dl.ash2txt.org/datasets/fiber-skeletons/fiber-skeletons.zip",
]


def get_json(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "unscroll-dl"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def head_size(url, timeout=60):
    req = urllib.request.Request(url, method="HEAD",
                                 headers={"User-Agent": "unscroll-dl"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        n = r.headers.get("Content-Length")
        return r.status, (int(n) if n else None)


def main():
    report = {}
    try:
        meta = get_json(HF_API)
        sibs = meta.get("siblings", [])
        total = sum(s.get("size") or 0 for s in sibs)
        report["ink_9um"] = {"files": len(sibs),
                             "total_gb": round(total / 1e9, 2),
                             "sample": [s.get("rfilename")
                                        for s in sibs[:8]]}
        print(f"ink_9um: {len(sibs)} files, {total / 1e9:.2f} GB")
    except Exception as e:  # noqa: BLE001
        report["ink_9um"] = {"error": f"{type(e).__name__}: {e}"}
        print(f"ink_9um FAILED: {e}")
    for url in DL_FILES:
        try:
            st, n = head_size(url)
            gb = round(n / 1e9, 2) if n else None
            report[url] = {"status": st, "gb": gb}
            print(f"{url}: {st} {gb} GB")
        except Exception as e:  # noqa: BLE001
            report[url] = {"error": f"{type(e).__name__}: {e}"}
            print(f"{url} FAILED: {e}")
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
