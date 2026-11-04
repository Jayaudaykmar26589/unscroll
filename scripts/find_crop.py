"""Find the 1cm Scroll 1 crop URL + masked-volume info. Stdlib only."""
import re
import urllib.request
from pathlib import Path

PAGES = [
    "https://scrollprize.org/tutorial3",
    "https://raw.githubusercontent.com/JamesDarby345/VesuviusDataDownload/"
    "main/README.md",
]


def main():
    out_lines = []
    for url in PAGES:
        try:
            req = urllib.request.Request(url,
                                         headers={"User-Agent": "unscroll"})
            text = urllib.request.urlopen(req, timeout=90).read()
            text = text.decode("utf-8", "replace")
            print(f"===== {url} ({len(text)} bytes)")
            for m in re.finditer(r"1\s?cm.{2000}", text,
                                 re.IGNORECASE | re.DOTALL):
                out_lines.append("---8<---")
                clean = re.sub(r"<[^>]+>", " ", m.group(0))
                clean = re.sub(r"\s+", " ", clean)
                out_lines.append(clean[:900])
        except Exception as e:  # noqa: BLE001
            out_lines.append(f"{url} FAILED: {e}")
    Path("data/crop_cmd.txt").write_text("\n".join(out_lines),
                                         encoding="utf-8")
    print("wrote data/crop_cmd.txt")


if __name__ == "__main__":
    main()
