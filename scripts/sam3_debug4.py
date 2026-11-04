"""Debug v3: point prompt + manual stage poke, then propagate."""
import sys
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "external" / "sam3"))
LOG = open(ROOT / "scripts" / "sam3_debug4.log", "w")
torch.backends.cuda.sdp_kernel(
    enable_flash=False, enable_mem_efficient=False, enable_math=True)

from sam3.model_builder import build_sam3_multiplex_video_predictor  # noqa: E402

fdir = ROOT / "curves_sam31_p0" / "frames3"
predictor = build_sam3_multiplex_video_predictor(
    checkpoint_path=str(ROOT / "checkpoints" / "sam3.1_multiplex.pt"),
    use_fa3=False, async_loading_frames=False)
r = predictor.handle_request(dict(type="start_session", resource_path=str(fdir)))
sid = r["session_id"]
rp = predictor.handle_request(dict(
    type="add_prompt", session_id=sid, frame_index=0,
    points=torch.tensor([[0.5, 0.5]], dtype=torch.float32),
    point_labels=torch.tensor([1], dtype=torch.int32), obj_id=0))
o = rp["outputs"]
print("seed out_binary_masks:", o["out_binary_masks"].shape,
      "probs:", o["out_probs"], file=LOG, flush=True)
state = predictor._all_inference_states[sid]["state"]
state["previous_stages_out"][0] = "_THIS_FRAME_HAS_OUTPUTS_"
try:
    n = 0
    for resp in predictor.handle_stream_request(dict(
            type="propagate_in_video", session_id=sid)):
        print("prop frame:", resp.get("frame_index"), file=LOG, flush=True)
        o = resp.get("outputs")
        if isinstance(o, dict):
            for k, v in o.items():
                print("  pout", k, type(v),
                      getattr(v, "shape", list(v.keys()) if isinstance(v, dict) else ""),
                      file=LOG, flush=True)
        n += 1
        if n >= 3:
            break
    print("propagate OK", file=LOG, flush=True)
except Exception:
    import traceback
    traceback.print_exc(file=LOG)
    print("propagate FAILED", file=LOG, flush=True)
LOG.close()
print("done")
