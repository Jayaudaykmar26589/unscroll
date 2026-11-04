"""Debug: why does multiplex propagate see no prompts after add_prompt? ."""
import sys
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "external" / "sam3"))
from sam3.model_builder import build_sam3_multiplex_video_predictor  # noqa: E402

import io
LOG = open(ROOT / "scripts" / "sam3_debug.log", "w")
fdir = ROOT / "curves_sam31_p0" / "frames3"
predictor = build_sam3_multiplex_video_predictor(
    checkpoint_path=str(ROOT / "checkpoints" / "sam3.1_multiplex.pt"),
    use_fa3=False, async_loading_frames=False)
r = predictor.handle_request(dict(type="start_session", resource_path=str(fdir)))
sid = r["session_id"]
print("session:", sid, file=LOG, flush=True)
NW, NH = 2048, 2048
pts = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
labs = torch.tensor([1], dtype=torch.int32)
resp = predictor.handle_request(dict(
    type="add_prompt", session_id=sid, frame_index=0,
    points=pts, point_labels=labs, obj_id=0))
print("add_prompt keys:", list(resp.keys()), file=LOG, flush=True)
print("frame_index:", resp.get("frame_index"), file=LOG, flush=True)
outs = resp.get("outputs")
print("outputs type:", type(outs), file=LOG, flush=True)
if isinstance(outs, dict):
    for k, v in outs.items():
        print("  out", k, type(v),
              getattr(v, "shape", list(v.keys()) if isinstance(v, dict) else ""),
              flush=True)
state = predictor._all_inference_states[sid]["state"]
print("state keys:", list(state.keys()) if isinstance(state, dict) else type(state), file=LOG, flush=True)
if isinstance(state, dict):
    for k, v in state.items():
        if "prompt" in k.lower() or "point" in k.lower() or "cond" in k.lower():
            print("  prompt-ish", k, type(v), file=LOG, flush=True)
