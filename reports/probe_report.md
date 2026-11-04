# Day-0 Probe Report

python: 3.12.7 (C:\Users\Jayachandiran U\anaconda3\python.exe)
platform: Windows-11-10.0.26300-SP0

gpu: {"ok": true, "name": "NVIDIA GeForce RTX 4060 Laptop GPU", "total_gb": 8.0, "free_gb": 7.2}
VRAM assert PASS: 7.2GB free >= 6.5GB usable.

ram_gb: {"total_gb": 31.1, "avail_gb": 12.5}
disk C:\: free=46.5GB total=510.3GB
disk D:\: free=630.8GB total=976.6GB

## Imports (system interpreter)
- numpy: {"ok": true, "version": "2.0.2"}
- yaml: {"ok": true, "version": "6.0.1"}
- pytest: {"ok": true, "version": "7.4.4"}
- torch: {"ok": true, "version": "2.2.2+cpu"}
- zarr: {"ok": false, "error": "ModuleNotFoundError: No module named 'zarr'"}
- tiffile: {"ok": false, "error": "ModuleNotFoundError: No module named 'tiffile'"}
- skimage: {"ok": false, "error": "ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject"}
- requests: {"ok": true, "version": "2.32.3"}

## Network (10s timeout; fail = rerun outside sandbox)
- https://pypi.org/simple/: {"ok": false, "error": "URLError: <urlopen error timed out>"}
- https://scrollprize.org/data: {"ok": false, "error": "URLError: <urlopen error timed out>"}
- https://dl.ash2txt.org/: {"ok": false, "error": "URLError: <urlopen error timed out>"}
- https://huggingface.co/: {"ok": true, "status": 200, "ms": 5649, "bytes": "174711"}

## Go/no-go window
PENDING: fixed Tier B z-window set at Tier B pin time (Day 1, recorded here).

RESULT: PASS (hardware asserts hold; network/data pending Day 1).

## Environment decision (Day 0, post-probe)

* `python -m venv` fails on this anaconda python (ensurepip broken).
  Day 0-1 runs on the system interpreter (stdlib + numpy + pytest +
  pyyaml present; no new installs needed). Broken `.venv/` removed.
* torch 2.2.2 imports but is **+cpu (no CUDA)**. Day 2 needs a CUDA
  torch env: `conda create -p ./.conda-env python=3.11 numpy\<2` then
  CUDA torch per pytorch.org + SAM weights per recipe.lock. Requires
  network; run outside this sandbox if pip/conda time out here.
* Data hosts reachable on retry (HTTP 200, slow TLS): exact artifact
  URLs resolve Day 1 via `download.py --add` after browsing the index.
* zarr/tifffile/skimage missing or broken under numpy 2: Day 1 volumes
  use `.npy` + optional-dependency guards until the Day 2 env lands.
