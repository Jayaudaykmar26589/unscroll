"""Debug9: direct tracker.propagate per frame (bypass multiplex merge)."""
import sys
import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "external" / "sam3"))
LOG = open(ROOT / "scripts" / "sam3_debug9.log", "w")

from sam3.model_builder import build_sam3_multiplex_video_predictor  # noqa: E402


def run_case(predictor, fdir, pt, nframes, tag):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        r = predictor.handle_request(dict(type="start_session", resource_path=str(fdir)))
        sid = r["session_id"]
        predictor.handle_request(dict(
            type="add_prompt", session_id=sid, frame_index=0,
            points=torch.tensor([pt], dtype=torch.float32),
            point_labels=torch.tensor([1], dtype=torch.int32), obj_id=0))
        model = predictor.model
        state = predictor._all_inference_states[sid]["state"]
        sam2_state = state["sam2_inference_states"][0]
        model.tracker.propagate_in_video_preflight(sam2_state, run_mem_encoder=True)
        for z in range(nframes):
            model._prepare_backbone_feats(state, z, False)
            got = None
            for out in model.tracker.propagate_in_video(
                    sam2_state, start_frame_idx=z, max_frame_num_to_track=0,
                    reverse=False, tqdm_disable=True, run_mem_encoder=True):
                got = out
            fz, oids, low, vres, scores = got
            m = (vres[0, 0].detach().cpu().numpy() > 0.0) if len(oids) else None
            fill = int(m.sum()) if m is not None else 0
            sc = float(scores[0].detach().cpu()) if len(oids) else float("nan")
            print(f"{tag} f{fz}: fill={fill} score={sc:.3f}", file=LOG, flush=True)
        predictor.handle_request(dict(type="close_session", session_id=sid,
                                       run_gc_collect=False))


predictor = build_sam3_multiplex_video_predictor(
    checkpoint_path=str(ROOT / "checkpoints" / "sam3.1_multiplex.pt"),
    use_fa3=False, async_loading_frames=False)
run_case(predictor, ROOT / "scripts" / "circle_frames", [0.5, 0.5], 4, "circle")
run_case(predictor, ROOT / "curves_sam31_p1" / "frames3", [0.5, 0.5], 4, "ct")
LOG.close()
print("done")
