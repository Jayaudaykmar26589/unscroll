"""Debug empty masks: logit stats + 3x3 click grid on frame 0."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external" / "sam2"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.trace.sam_tracker import load_frame  # noqa: E402


def main():
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    tif = sorted((ROOT / "data" / "cache" / "scroll1_6100_6200").glob(
        "*.tif"))[0]
    rgb, meta = load_frame(tif)
    xs, ys = meta[0], meta[1]
    print(f"frame {rgb.shape} scales x={xs:.2f} y={ys:.2f}")
    fdir = ROOT / "curves_pilot" / "chunk_0000_frames"
    predictor = SAM2VideoPredictor.from_pretrained(
        "facebook/sam2-hiera-small")
    cx, cy = 5568 / xs, 3944 / ys
    print(f"click small-frame: ({cx:.1f}, {cy:.1f})")
    with torch.inference_mode(), torch.autocast("cuda",
                                                dtype=torch.bfloat16):
        for dx in (-8, 0, 8):
            for dy in (-8, 0, 8):
                state = predictor.init_state(str(fdir))
                _, _, out = predictor.add_new_points_or_box(
                    state, frame_idx=0, obj_id=0,
                    points=np.array([[cx + dx, cy + dy]]),
                    labels=np.array([1]))
                m = out[0][0].cpu().numpy()
                print(f"dx={dx:+d} dy={dy:+d}: shape={m.shape} "
                      f"min={m.min():.3f} max={m.max():.3f} "
                      f"mean={m.mean():.4f} pos={(m > 0).sum()}")


if __name__ == "__main__":
    main()
