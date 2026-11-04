"""Debug v2: text cold-start + point refine, then propagate."""
import sys
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "external" / "sam3"))
LOG = open(ROOT / "scripts" / "sam3_debug2.log", "w")

from sam3.model_builder import build_sam3_multiplex_video_predictor  # noqa: E402

fdir = ROOT / "curves_sam31_p0" / "frames3"
predictor = build_sam3_multiplex_video_predictor(
    checkpoint_path=str(ROOT / "checkpoints" / "sam3.1_multiplex.pt"),
    use_fa3=False, async_loading_frames=False)
r = predictor.handle_request(dict(type="start_session", resource_path=str(fdir)))
sid = r["session_id"]
print("session:", sid, file=LOG, flush=True)
try:
    rt = predictor.handle_request(dict(
        type="add_prompt", session_id=sid, frame_index=0,
        text="thin bright curved layer"))
    print("text add_prompt ok:", rt.get("frame_index"),
          type(rt.get("outputs")), file=LOG, flush=True)
    o = rt.get("outputs")
    if isinstance(o, dict):
        for k, v in o.items():
            print("  tout", k, type(v),
                  getattr(v, "shape", list(v.keys()) if isinstance(v, dict) else ""),
                  file=LOG, flush=True)
except Exception as e:
    print("text add_prompt FAILED:", repr(e)[:300], file=LOG, flush=True)
try:
    rp = predictor.handle_request(dict(
        type="add_prompt", session_id=sid, frame_index=0,
        points=torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        point_labels=torch.tensor([1], dtype=torch.int32), obj_id=0))
    print("point add_prompt ok:", rp.get("frame_index"),
          type(rp.get("outputs")), file=LOG, flush=True)
    o = rp.get("outputs")
    if isinstance(o, dict):
        print("  point outs keys:", list(o.keys()), file=LOG, flush=True)
except Exception as e:
    print("point add_prompt FAILED:", repr(e)[:300], file=LOG, flush=True)
state = predictor._all_inference_states[sid]["state"]
print("prev_stages:", state["previous_stages_out"], file=LOG, flush=True)
try:
    n = 0
    for resp in predictor.handle_stream_request(dict(
            type="propagate_in_video", session_id=sid)):
        print("prop frame:", resp.get("frame_index"),
              type(resp.get("outputs")), file=LOG, flush=True)
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
except Exception as e:
    import traceback
    traceback.print_exc(file=LOG)
    print("propagate FAILED", file=LOG, flush=True)
LOG.close()
print("done")
