"""Check scroll slice size + fragment volume options. Stdlib only."""
import re
import urllib.request

UA = {"User-Agent": "unscroll-dl"}


def head(url, timeout=90):
    req = urllib.request.Request(url, method="HEAD", headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        n = r.headers.get("Content-Length")
        return r.status, (int(n) if n else None)


def listing(url, timeout=90):
    req = urllib.request.Request(url, headers=UA)
    html = urllib.request.urlopen(req, timeout=timeout).read()
    html = html.decode("utf-8", "replace")
    return sorted(set(re.findall(r'href="([^"/][^"]*)"', html)))


def main():
    out = []
    st, n = head("http://dl.ash2txt.org/full-scrolls/Scroll1/"
                 "PHercParis4.volpkg/volumes/20230205180739/06000.tif")
    out.append(f"scroll slice 06000.tif: {st} "
               f"{n / 1e6 if n else None} MB")
    out.append(f"100-slice window ~ {n * 100 / 1e9 if n else None} GB")
    try:
        names = listing("https://dl.ash2txt.org/fragments/Frag1/"
                        "PHercParis2Fr47.volpkg/volumes/")
        out.append(f"frag1 volumes/: {names[:10]}")
    except Exception as e:  # noqa: BLE001
        out.append(f"frag1 volumes listing FAILED: {e}")
    print("\n".join(out))
    open("data/scroll_check.txt", "w").write("\n".join(out))


if __name__ == "__main__":
    main()
