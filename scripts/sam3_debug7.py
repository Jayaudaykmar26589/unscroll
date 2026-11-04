"""Debug7: full traceback of text-prompt failure."""
import sys
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "external" / "sam3"))
LOG = open(ROOT / "scripts" / "sam3_debug7.log", "w")

from sam3.model_builder import build_sam3_multiplex_video_predictor  # noqa: E402

fdir = ROOT / "curves_sam31_p1" / "frames3"
predictor = build_sam3_multiplex_video_predictor(
    checkpoint_path=str(ROOT / "checkpoints" / "sam3.1_multiplex.pt"),
    use_fa3=False, async_loading_frames=False)
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    r = predictor.handle_request(dict(type="start_session", resource_path=str(fdir)))
    sid = r["session_id"]
    try:
        predictor.handle_request(dict(
            type="add_prompt", session_id=sid, frame_index=0,
            text="thin bright curved layer"))
        print("text OK", file=LOG, flush=True)
    except Exception:
        import traceback
        traceback.print_exc(file=LOG)
LOG.close()
print("done")
