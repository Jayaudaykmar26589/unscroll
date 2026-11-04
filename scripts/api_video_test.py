"""Test sam-3.1 video path: 8-slice MP4 from real fragment data."""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_key():
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("META_MODEL_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("no key")


def make_mp4(out, n=8, size=256):
    import imageio.v2 as imageio
    import numpy as np
    import tifffile
    tifs = sorted((ROOT / "data" / "cache" / "frag1" /
                   "surface_volume").glob("*.tif"))[:n]
    assert tifs, "no surface tifs yet (fetch still running?)"
    frames = []
    for t in tifs:
        img = tifffile.imread(t).astype(np.float32)
        lo, hi = np.percentile(img, (1, 99))
        u8 = np.clip((img - lo) / max(hi - lo, 1e-6) * 255, 0, 255)
        u8 = u8.astype(np.uint8)
        h, w = u8.shape
        ys = (np.linspace(0, h - 1, size)).astype(int)
        xs = (np.linspace(0, w - 1, size)).astype(int)
        small = u8[ys[:, None], xs]
        frames.append(np.stack([small] * 3, -1))
    imageio.mimsave(out, frames, fps=2, codec="libx264")
    return out


def main():
    from openai import OpenAI
    mp4 = make_mp4(ROOT / "data" / "cache" / "probe8.mp4")
    print(f"mp4: {mp4.stat().st_size} bytes")
    b64 = base64.b64encode(mp4.read_bytes()).decode()
    client = OpenAI(base_url="https://api.meta.ai/v1",
                    api_key=load_key(), timeout=300.0)
    try:
        stream = client.responses.create(
            model="sam-3.1",
            input=[{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "papyrus layer"},
                {"type": "input_video",
                 "video_url": f"data:video/mp4;base64,{b64}"},
            ]}],
            stream=True,
            metadata={"mask_encoding": "one_bit"},
        )
        events = [ev.model_dump() for ev in stream]
        (ROOT / "reports" / "sam_video.json").write_text(
            json.dumps(events, indent=1)[:30000])
        types = {}
        for d in events:
            types[d.get("type", "?")] = types.get(d.get("type", "?"), 0) + 1
        print(f"{len(events)} events: {types}")
        for d in events:
            s = json.dumps(d)
            if "mask" in s.lower() and "mask_encoding" not in s:
                print("MASKED?", s[:800])
                break
        last = events[-1].get("response", events[-1])
        print("status:", last.get("status"))
        print("usage:", json.dumps(last.get("usage")))
        print("output head:", json.dumps(last.get("output"))[:1500])
    except Exception as e:  # noqa: BLE001
        print(f"VIDEO FAILED: {type(e).__name__}: {str(e)[:500]}")


if __name__ == "__main__":
    main()
