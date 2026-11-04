"""Test sam-3.1 prompt variants on real fragment photo. Tiny calls."""
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


def img_b64(path, size=256):
    from PIL import Image
    img = Image.open(path).convert("RGB")
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def main():
    from openai import OpenAI
    client = OpenAI(base_url="https://api.meta.ai/v1",
                    api_key=load_key(), timeout=180.0)
    photo = img_b64(ROOT / "data" / "cache" / "frag1" / "ir.png")
    tests = [
        ("photo+fragment", "fragment"),
        ("photo+papyrus", "papyrus"),
        ("photo+white-region", "white region"),
        ("photo+notext", None),
    ]
    for name, text in tests:
        content = []
        if text:
            content.append({"type": "input_text", "text": text})
        content.append({"type": "input_image",
                        "image_url": f"data:image/png;base64,{photo}"})
        try:
            resp = client.responses.create(
                model="sam-3.1",
                input=[{"type": "message", "role": "user",
                        "content": content}],
                metadata={"mask_encoding": "one_bit"},
            )
            d = resp.model_dump()
            outs = []
            for item in d.get("output", []):
                for c in item.get("content", []):
                    outs.append({k: (str(v)[:300]) for k, v in c.items()})
            print(f"{name}: status={d['status']} "
                  f"usage={d.get('usage')} out={json.dumps(outs)[:600]}")
        except Exception as e:  # noqa: BLE001
            print(f"{name} FAILED: {type(e).__name__}: {str(e)[:300]}")


if __name__ == "__main__":
    main()
