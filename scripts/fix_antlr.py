"""Manual antlr4 4.9.3 placement (pure-python pkg; pip install breaks
on extended-path script handling). Run once. Idempotent."""
import shutil
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SP = ROOT / ".venv" / "Lib" / "site-packages"
URL = ("https://files.pythonhosted.org/packages/43/83/"
       "f8d35c5928bd3c6c8cbd1e8c2d7b07a3e3d2e0d8b2e8e2b9e3e3e3e3e3e3/placeholder")


def main():
    # download sdist via pip download (no install) to avoid URL guessing
    import subprocess
    dl = ROOT / ".venv" / "sdist_dl"
    dl.mkdir(exist_ok=True)
    subprocess.run(
        [str(ROOT / ".venv" / "Scripts" / "python"), "-m", "pip",
         "download", "--no-cache-dir", "--no-deps", "--dest", str(dl),
         "antlr4-python3-runtime==4.9.3", "--timeout", "120"],
        check=True)
    tgz = next(dl.glob("*.tar.gz"))
    with tarfile.open(tgz) as t:
        t.extractall(dl)
    src = next(dl.glob("antlr4-python3-runtime-*/src/antlr4"))
    dst = SP / "antlr4"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print("placed antlr4 4.9.3 at", dst)
    import sys
    sys.path.insert(0, str(SP))
    import antlr4
    print("antlr4 file:", antlr4.__file__)


if __name__ == "__main__":
    main()
