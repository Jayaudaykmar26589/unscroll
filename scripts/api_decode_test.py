"""Validate mask-record parsing on flower control (known masks)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.trace.api_tracker import decode_masks, track_crop  # noqa: E402


def main():
    status, text, usage = track_crop(
        ROOT / "data" / "cache" / "flower.mp4", "flower")
    print(f"status={status} chars={len(text)} usage={usage}")
    try:
        per_frame = decode_masks(text, (540, 960))
        nf = len(per_frame)
        nm = sum(len(v) for v in per_frame.values())
        print(f"decoded frames={nf} masks={nm}")
        for fi, masks in list(per_frame.items())[:3]:
            for m in masks[:1]:
                print(f"  f{fi}: shape={m.shape} dtype={m.dtype} "
                      f"on={m.sum()}")
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"DECODE FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
