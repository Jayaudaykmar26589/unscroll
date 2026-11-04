"""Debug10: direct tracker propagation with 7-point click+ring prompt on CT."""
import sys
import numpy as np
import torch
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "external" / "sam3"))
sys.path.insert(0, str(ROOT))
LOG = open(ROOT / "scripts" / "sam3_debug10.log", "w")

from sam3.model_builder import build_sam3_multiplex_video_predictor  # noqa: E402
from src.trace.sam_tracker import layer_normal  # noqa: E402

fdir = ROOT / "curves_sam31_p1" / "frames3"
first = np.array(Image.open(fdir / "00000.jpg"))[:, :, 0]
NH, NW = first.shape
px, py = NW / 2, NH / 2
nx, ny = layer_normal(first, px, py)
pts = [[px / NW, py / NH]] + [
    [(px + s * nx) / NW, (py + s * ny) / NH]
    for s in (30, 60, 90, -30, -60, -90)]
labs = [1] + [0] * 6
print("normal:", nx, ny, file=LOG, flush=True)

predictor = build_sam3_multiplex_video_predictor(
    checkpoint_path=str(ROOT / "checkpoints" / "sam3.1_multiplex.pt"),
    use_fa3=False, async_loading_frames=False)
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    r = predictor.handle_request(dict(type="start_session", resource_path=str(fdir)))
    sid = r["session_id"]
    predictor.handle_request(dict(
        type="add_prompt", session_id=sid, frame_index=0,
        points=torch.tensor(pts, dtype=torch.float32),
        point_labels=torch.tensor(labs, dtype=torch.int32), obj_id=0))
    model = predictor.model
    state = predictor._all_inference_states[sid]["state"]
    sam2_state = state["sam2_inference_states"][0]
    model.tracker.propagate_in_video_preflight(sam2_state, run_mem_encoder=True)
    for z in range(8):
        model._prepare_backbone_feats(state, z, False)
        got = None
        for out in model.tracker.propagate_in_video(
                sam2_state, start_frame_idx=z, max_frame_num_to_track=0,
                reverse=False, tqdm_disable=True, run_mem_encoder=True):
            got = out
        fz, oids, low, vres, scores = got
        if len(oids):
            m = vres[0, 0].detach().cpu().numpy() > 0.0
            fill = int(m.sum())
            sc = float(scores[0].detach().cpu())
            ys, xs = np.nonzero(m)
            print(f"f{fz}: fill={fill} score={sc:.3f} "
                  f"bbox=({xs.min()},{ys.min()})-({xs.max()},{ys.max()})",
                  file=LOG, flush=True)
            if z in (0, 4, 7):
                Image.fromarray((m * 255).astype(np.uint8)).save(
                    ROOT / "curves_sam31_p1" / f"direct_f{z}.png")
        else:
            print(f"f{fz}: EMPTY", file=LOG, flush=True)
LOG.close()
print("done")
