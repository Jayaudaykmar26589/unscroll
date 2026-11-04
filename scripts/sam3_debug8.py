"""Debug8: point prompt on synthetic circle video (machinery check)."""
import sys
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "external" / "sam3"))
LOG = open(ROOT / "scripts" / "sam3_debug8.log", "w")

from sam3.model_builder import build_sam3_multiplex_video_predictor  # noqa: E402

fdir = ROOT / "scripts" / "circle_frames"
fdir.mkdir(exist_ok=True)
for i in range(8):
    img = Image.new("RGB", (1024, 1024), (20, 20, 20))
    d = ImageDraw.Draw(img)
    d.ellipse([362, 362, 662, 662], fill=(220, 220, 220))
    img.save(fdir / f"{i:05d}.jpg")
predictor = build_sam3_multiplex_video_predictor(
    checkpoint_path=str(ROOT / "checkpoints" / "sam3.1_multiplex.pt"),
    use_fa3=False, async_loading_frames=False)
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    r = predictor.handle_request(dict(type="start_session", resource_path=str(fdir)))
    sid = r["session_id"]
    rp = predictor.handle_request(dict(
        type="add_prompt", session_id=sid, frame_index=0,
        points=torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        point_labels=torch.tensor([1], dtype=torch.int32), obj_id=0))
    o = rp["outputs"]
    m = np.asarray(o["out_binary_masks"])
    print("seed masks:", m.shape, "fill:",
          (m > 0).sum(axis=(1, 2)) if m.shape[0] else None, file=LOG, flush=True)
    state = predictor._all_inference_states[sid]["state"]
    state["previous_stages_out"][0] = "_THIS_FRAME_HAS_OUTPUTS_"
    n = 0
    for resp in predictor.handle_stream_request(dict(
            type="propagate_in_video", session_id=sid)):
        z = resp["frame_index"]
        o = resp["outputs"]
        m = np.asarray(o["out_binary_masks"])
        print(f"f{z} masks:", m.shape, "fill:",
              (m > 0).sum(axis=(1, 2)) if m.shape[0] else None,
              "probs:", np.asarray(o["out_probs"]), file=LOG, flush=True)
        n += 1
        if n >= 3:
            break
LOG.close()
print("done")
