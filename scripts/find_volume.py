"""List Scroll1 volpkg volumes + segment meta for volume mapping."""
import re
import urllib.request
from pathlib import Path

UA = {"User-Agent": "unscroll-dl"}
BASE = "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/"


def listing(url):
    req = urllib.request.Request(url, headers=UA)
    html = urllib.request.urlopen(req, timeout=90).read()
    html = html.decode("utf-8", "replace")
    return sorted(set(re.findall(r'href="([^"/][^"]*)"', html)))


def main():
    out = []
    try:
        out.append(f"volumes/: {listing(BASE + 'volumes/')}")
    except Exception as e:  # noqa: BLE001
        out.append(f"volumes FAILED: {e}")
    for f in ["volumes/meta.json", "config.json",
              "paths/20230518181521/meta.json"]:
        try:
            req = urllib.request.Request(BASE + f, headers=UA)
            txt = urllib.request.urlopen(req, timeout=90).read()
            txt = txt.decode("utf-8", "replace")
            out.append(f"--- {f} ({len(txt)} bytes)")
            out.append(txt[:800])
        except Exception as e:  # noqa: BLE001
            out.append(f"{f} FAILED: {e}")
    Path("data/volumes_info.txt").write_text("\n".join(out))
    print("\n".join(out)[:2000])


if __name__ == "__main__":
    main()
