"""Day-2 go/no-go: Tier B 50-slice hold test.

  python scripts/day2_gono.py --slices DIR --click X,Y,Z --ckpt CKPT --z0 Z0

Pass: >=45/50 slices with ridge_frac2 >= 0.50 (interim; chance ~= 0.26).
This is an image-support proxy gate, not G1 surface accuracy. Independent
mesh validation is implemented by scripts/validate_mesh_reference.py for the
frozen 50-slice experiment. Writes reports/gono_verdict.md.
Exit 0 PASS / 2 FAIL.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import tifffile  # noqa: E402

from ridge_eval import dilate2, ridge_mask  # noqa: E402
from src.trace.slices import select_slices  # noqa: E402

NSLICES = 50
RIDGE_PASS = 0.50


def ridge_frac(pts, tif):
    img = tifffile.imread(tif).astype(float)
    rm = dilate2(ridge_mask(img))
    qx = np.clip(pts[:, 0].astype(int), 0, rm.shape[1] - 1)
    qy = np.clip(pts[:, 1].astype(int), 0, rm.shape[0] - 1)
    return float(rm[qy, qx].mean())


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", required=True)
    ap.add_argument("--click", required=True)
    ap.add_argument("--z0", type=int, required=True)
    ap.add_argument("--ckpt", default="checkpoints/sam3.1_multiplex.pt")
    ap.add_argument("--tracker", default="sam3", choices=["sam3", "sam2"])
    ap.add_argument("--model", default="large")
    ap.add_argument("--crop", type=int, default=2048)
    ap.add_argument("--out", default="curves_gono")
    a = ap.parse_args(argv)

    out = ROOT / a.out
    mod = "src.trace.sam3_tracker" if a.tracker == "sam3" else "src.trace.sam_tracker"
    cmd = [sys.executable, "-m", mod,
           "--slices", a.slices, "--click", a.click, "--window",
           str(NSLICES), "--out", str(out), "--z0", str(a.z0),
           "--crop", str(a.crop)]
    if a.tracker == "sam3":
        cmd += ["--ckpt", a.ckpt]
    else:
        cmd += ["--model", a.model]
    print("TRACK:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    refs = select_slices(a.slices, int(a.click.split(",")[2]), NSLICES, a.z0)
    n_ok, fracs = 0, []
    if r.returncode != 0:
        verdict = "FAIL (tracker crashed)"
    else:
        for ref in refs:
            i = ref.index
            p = out / f"{i:05d}.jsonl"
            if not p.exists():
                fracs.append(float("nan"))
                continue
            pts = np.array(json.loads(p.read_text())["points"])
            if len(pts) == 0:
                fracs.append(0.0)
                continue
            f = ridge_frac(pts, ref.path)
            fracs.append(f)
            if f >= RIDGE_PASS:
                n_ok += 1
            print(f"slice {i}: n={len(pts)} ridge_frac2={f:.3f}", flush=True)
        verdict = ("PASS (image-support proxy only)" if n_ok >= 45
                   else "FAIL (build classical fallback Days 3-4)")
    arr = np.array(fracs)
    report = (f"# Go/No-Go Verdict\n\nclick: {a.click}  z0: {a.z0}  "
              f"tracker: {a.tracker}\n\nslices with ridge_frac2 >= "
              f"{RIDGE_PASS}: {n_ok}/{NSLICES} (need >=45)\n\n"
              f"mean ridge_frac2: {np.nanmean(arr):.3f} "
              f"(chance ~= 0.26)\n\n"
              f"Surface accuracy: NOT ASSESSED for this run. "
              f"Use an independent, provenance-matched mesh evaluation; "
              f"ridge proximity does not establish layer correctness.\n\n**{verdict}**\n")
    (ROOT / "reports" / "gono_verdict.md").write_text(report)
    print(report, flush=True)
    return 0 if verdict.startswith("PASS") else 2


if __name__ == "__main__":
    sys.exit(main())
