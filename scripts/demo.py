"""Golden demo replay. --offline installs a socket guard then replays golden/.

  python scripts/demo.py --offline

Any network call raises; SHA256SUMS must verify; translation output is
printed from frozen fixtures (no Spark call).
"""
import hashlib
import json
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _guard():
    def _blocked(*a, **k):
        raise RuntimeError("network blocked by --offline socket guard")
    socket.socket = _blocked


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--golden", default="golden")
    a = ap.parse_args()
    if a.offline:
        import os
        os.environ["UNROLL_OFFLINE"] = "1"
        _guard()
    g = ROOT / a.golden
    run = json.loads((g / "run.json").read_text(encoding="utf-8"))
    ok = True
    for line in (g / "SHA256SUMS").read_text().splitlines():
        if not line.strip():
            continue
        want, name = line.split(None, 1)
        got = hashlib.sha256((g / name.strip()).read_bytes()).hexdigest()
        if got != want:
            print(f"MISMATCH {name}", flush=True)
            ok = False
    print(f"run={run['run_id']} frames={len(run['frames'])} sheet={run['sheet']}")
    for tgt, modes in run["translations"].items():
        for row in modes["fluent"]:
            print(f"[{tgt}] {row['input']} -> {row['translation']} (tconf={row['tconf']})")
    if not ok:
        raise SystemExit("golden manifest mismatch")
    print("GOLDEN OK (API-free)" if a.offline else "GOLDEN OK")


if __name__ == "__main__":
    main()
