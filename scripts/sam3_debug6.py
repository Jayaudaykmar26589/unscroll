"""Debug6: save raw seed + frame-0 masks with click overlay."""
import sys
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "external" / "sam3"))
LOG = open(ROOT / "scripts" / "sam3_debug6.log", "w")

from sam3.model_builder import build_sam3_multiplex_video_predictor  # noqa: E402

fdir = ROOT / "curves_sam31_p1" / "frames3"
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
    for resp in predictor.handle_stream_request(dict(
            type="propagate_in_video", session_id=sid)):
        z = resp["frame_index"]
        if z > 0:
            break
        o = resp["outputs"]
        m = np.asarray(o["out_binary_masks"])
        print("f0 masks:", m.shape, "fill:",
              (m > 0).sum(axis=(1, 2)) if m.shape[0] else None,
              "ids:", np.asarray(o["out_obj_ids"]),
              "probs:", np.asarray(o["out_probs"]), file=LOG, flush=True)
        img = Image.open(fdir / "00000.jpg").convert("RGB")
        W, H = img.size
        if m.shape[0]:
            mm = Image.fromarray((m[0] > 0).astype(np.uint8) * 255).resize((W, H))
            red = Image.new("RGB", (W, H), (255, 0, 0))
            img = Image.composite(red, img, mm.point(lambda v: int(v * 0.5)))
            ys, xs = np.nonzero(np.asarray(mm))
            print("f0 bbox x:", xs.min(), xs.max(), "y:", ys.min(), ys.max(),
                  file=LOG, flush=True)
        d = ImageDraw.Draw(img)
        d.ellipse([W / 2 - 8, H / 2 - 8, W / 2 + 8, H / 2 + 8],
                  outline=(0, 255, 0), width=3)
        img.save(ROOT / "curves_sam31_p1" / "f0_rawmask.png")
        print("saved f0_rawmask.png", file=LOG, flush=True)
LOG.close()
print("done")
