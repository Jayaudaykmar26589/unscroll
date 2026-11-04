"""List segment dir for mapping files. Stdlib only."""
import re
import urllib.request

URL = ("https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/"
       "paths/20230518181521/")


def main():
    req = urllib.request.Request(URL, headers={"User-Agent": "unscroll"})
    html = urllib.request.urlopen(req, timeout=90).read()
    html = html.decode("utf-8", "replace")
    for n in sorted(set(re.findall(r'href="([^"/][^"]*)"', html))):
        print(" ", n)


if __name__ == "__main__":
    main()
