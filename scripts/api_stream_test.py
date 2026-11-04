"""Stream a tiny sam-3.1 call, dump event types + payload heads."""
import base64
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_key():
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("META_MODEL_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("no key")


def main():
    from openai import OpenAI
    from PIL import Image
    client = OpenAI(base_url="https://api.meta.ai/v1", api_key=load_key(),
                    timeout=180.0)
    img = Image.open(ROOT / "data" / "cache" / "frag1" / "mask.png")
    img = img.convert("RGB")
    img.thumbnail((256, 256))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    stream = client.responses.create(
        model="sam-3.1",
        input=[{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "papyrus layer"},
            {"type": "input_image",
             "image_url": f"data:image/png;base64,{b64}"},
        ]}],
        stream=True,
        metadata={"mask_encoding": "one_bit"},
    )
    events = []
    for ev in stream:
        d = ev.model_dump()
        events.append(d)
    (ROOT / "reports" / "sam_stream.json").write_text(
        json.dumps(events, indent=1)[:20000])
    print(f"{len(events)} events; types:")
    seen = {}
    for d in events:
        t = d.get("type", "?")
        seen[t] = seen.get(t, 0) + 1
    for t, n in seen.items():
        print(f"  {t} x{n}")
    # print first mask-ish event fully (truncated)
    for d in events:
        s = json.dumps(d)
        if "mask" in s.lower() or "raster" in s.lower() or "base64" in s.lower():
            print("MASK EVENT HEAD:", s[:1200])
            break


if __name__ == "__main__":
    main()
