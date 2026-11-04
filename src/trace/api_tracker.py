"""SAM 3.1 API tracker: slices -> MP4 -> Model API -> masks -> curves.

Crop-to-click: tight crop around the click + generic text prompt,
since the hosted API takes text prompts only (no point prompts).
"""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_key():
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("META_MODEL_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("no META_MODEL_API_KEY in .env")


def frames_to_mp4(frames_rgb, out_path, fps=2):
    import imageio.v2 as imageio
    imageio.mimsave(str(out_path), frames_rgb, fps=fps, codec="libx264")
    return out_path


def crop_around(img, cx, cy, size=256):
    import numpy as np
    h, w = img.shape[:2]
    x0 = max(0, min(w - size, cx - size // 2))
    y0 = max(0, min(h - size, cy - size // 2))
    return img[y0:y0 + size, x0:x0 + size], (x0, y0)


def track_crop(mp4_path, text, model="sam-3.1", timeout=300.0):
    """One API video call. Returns raw output_text (mask records)."""
    from openai import OpenAI
    b64 = base64.b64encode(Path(mp4_path).read_bytes()).decode()
    client = OpenAI(base_url="https://api.meta.ai/v1",
                    api_key=load_key(), timeout=timeout)
    resp = client.responses.create(
        model=model,
        input=[{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": text},
            {"type": "input_video",
             "video_url": f"data:video/mp4;base64,{b64}"},
        ]}],
        metadata={"mask_encoding": "one_bit"},
    )
    d = resp.model_dump()
    texts = [c.get("text", "") for item in d.get("output", [])
             for c in item.get("content", [])]
    return d["status"], "".join(texts), d.get("usage")


def decode_masks(output_text, frame_hw):
    """Parse <Nf>id<|box|><|mask|> records -> {frame: [bool masks]}."""
    import numpy as np
    from meta_sam_parser import (decode_mask_to_raster,
                                 video_segmentation_format)
    h, w = frame_hw
    snaps = video_segmentation_format().create_parser().push(output_text)
    per_frame = {}
    for snap in snaps:
        for rec in snap.records:
            if type(rec).__name__ != "SegmentationMaskRecord":
                continue
            fi = rec.frame.frame_index
            b = rec.bounds
            crop = np.frombuffer(decode_mask_to_raster(rec.mask),
                                 dtype=np.uint8)
            crop = crop.reshape(rec.mask.height, rec.mask.width) > 0
            full = np.zeros((h, w), dtype=bool)
            full[b.top:b.bottom, b.left:b.right] = crop[
                :max(0, b.bottom - b.top), :max(0, b.right - b.left)]
            per_frame.setdefault(int(fi), []).append(full)
    return per_frame
