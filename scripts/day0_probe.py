"""Day-0 hardware/data probe. Stdlib only. Writes reports/probe_report.md.

Exit 0: hardware asserts pass (or reduced caps recorded).
Exit 2: a hard assert failed (see report).
Network failures are recorded, never fatal (user reruns outside sandbox).
"""
import ctypes
import json
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "probe_report.md"

HOST_PROBES = [
    "https://pypi.org/simple/",
    "https://scrollprize.org/data",
    "https://dl.ash2txt.org/",
    "https://huggingface.co/",
]

IMPORT_PROBES = ["numpy", "yaml", "pytest", "torch", "zarr", "tiffile",
                 "skimage", "requests"]


def gb(n):
    return round(n / 1024 ** 3, 1)


def check_gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            return {"ok": False, "error": out.stderr.strip()[:200]}
        parts = [p.strip() for p in out.stdout.strip().split(",")]
        total_mib, free_mib = int(parts[1]), int(parts[2])
        return {"ok": True, "name": parts[0],
                "total_gb": round(total_mib / 1024, 1),
                "free_gb": round(free_mib / 1024, 1)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def check_ram():
    try:
        class Stat(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        s = Stat()
        s.dwLength = ctypes.sizeof(Stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        return {"total_gb": gb(s.ullTotalPhys),
                "avail_gb": gb(s.ullAvailPhys)}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def check_host(url, timeout=10):
    t0 = time.time()
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "unscroll-probe"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            ms = int((time.time() - t0) * 1000)
            return {"ok": True, "status": r.status, "ms": ms,
                    "bytes": r.headers.get("Content-Length")}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:160]}


def check_import(name):
    try:
        mod = __import__(name)
        ver = getattr(mod, "__version__", "?")
        return {"ok": True, "version": str(ver)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:120]}"}


def main():
    fails = []
    L = ["# Day-0 Probe Report", "", f"python: {sys.version.split()[0]} "
         f"({sys.executable})", f"platform: {platform.platform()}", ""]

    gpu = check_gpu()
    L.append(f"gpu: {json.dumps(gpu)}")
    usable = gpu.get("free_gb", 0) if gpu.get("ok") else 0
    if not gpu.get("ok"):
        fails.append("nvidia-smi failed: " + gpu.get("error", "?"))
    elif usable < 6.5:
        L.append(f"REDUCED CAPS: only {usable}GB free; shrink bank/window "
                 "per VRAM table before Day 2.")
    else:
        L.append(f"VRAM assert PASS: {usable}GB free >= 6.5GB usable.")
    L.append("")

    ram = check_ram()
    L.append(f"ram_gb: {json.dumps(ram)}")
    for drive in ("C:\\", "D:\\"):
        try:
            u = shutil.disk_usage(drive)
            L.append(f"disk {drive}: free={gb(u.free)}GB "
                     f"total={gb(u.total)}GB")
        except Exception as e:  # noqa: BLE001
            L.append(f"disk {drive}: ERROR {e}")
    L.append("")

    L.append("## Imports (system interpreter)")
    for name in IMPORT_PROBES:
        r = check_import(name)
        L.append(f"- {name}: {json.dumps(r)}")
    L.append("")

    L.append("## Network (10s timeout; fail = rerun outside sandbox)")
    for url in HOST_PROBES:
        r = check_host(url)
        L.append(f"- {url}: {json.dumps(r)}")
    L.append("")

    L.append("## Go/no-go window")
    L.append("PENDING: fixed Tier B z-window set at Tier B pin time "
             "(Day 1, recorded here).")
    L.append("")
    if fails:
        L.append("## ASSERT FAILURES")
        L += [f"- {f}" for f in fails]
    else:
        L.append("RESULT: PASS (hardware asserts hold; network/data "
                 "pending Day 1).")

    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 2 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
