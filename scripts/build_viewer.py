"""Regenerate demo/viewer.html for a frame dir (count + paths)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TPL = (ROOT / "demo" / "viewer.html").read_text()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="frame dir under demo/")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    d = ROOT / "demo" / a.dir
    n = len(sorted(d.glob("[0-9]*.png")))
    html = TPL.replace('const N = 8, DIR = "frames_pilot", PAD = 5;',
                       f'const N = {n}, DIR = "{a.dir}", PAD = 5;')
    html = html.replace('max="7"', f'max="{n - 1}"')
    html = html.replace('src="frames_pilot/flat.png"', f'src="{a.dir}/flat.png"')
    out = ROOT / "demo" / (a.out or "viewer.html")
    out.write_text(html)
    print(f"{out}: {n} frames", flush=True)


if __name__ == "__main__":
    main()
