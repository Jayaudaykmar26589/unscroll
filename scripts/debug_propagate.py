"""Inspect propagate_in_video mask shapes/values per frame."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "sam2"))

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main():
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    predictor = SAM2VideoPredictor.from_pretrained(
        "facebook/sam2-hiera-small")
    fdir = ROOT / "curves_pilot" / "chunk_0000_frames"
    with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16):
        state = predictor.init_state(str(fdir))
        _, _, out = predictor.add_new_points_or_box(
            state, frame_idx=0, obj_id=0,
            points=np.array([[5568 / 7.712, 3944 / 7.712]]),
            labels=np.array([1]))
        print("prompt out:", out[0][0].shape,
              "pos:", (out[0][0].cpu().numpy() > 0).sum())
        for fi, oids, masks in predictor.propagate_in_video(state):
            m = masks[0][0].cpu().numpy()
            print(f"frame {fi}: shape={m.shape} dtype={m.dtype} "
                  f"min={m.min():.2f} max={m.max():.2f} "
                  f"pos={(m > 0).sum()}")
            if fi >= 2:
                break


if __name__ == "__main__":
    main()
