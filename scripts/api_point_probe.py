"""Probe undocumented point/box prompt shapes (flower control)."""
import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_key():
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("META_MODEL_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("no key")


def main():
    from openai import OpenAI
    b64 = base64.b64encode(
        (ROOT / "data" / "cache" / "flower.mp4").read_bytes()).decode()
    client = OpenAI(base_url="https://api.meta.ai/v1",
                    api_key=load_key(), timeout=300.0)
    video = {"type": "input_video",
             "video_url": f"data:video/mp4;base64,{b64}"}
    variants = {
        "point_only": [{"type": "input_point", "x": 480, "y": 270}, video],
        "text+point": [{"type": "input_text", "text": "flower"},
                       {"type": "input_point", "x": 480, "y": 270}, video],
        "text_with_coords": [{"type": "input_text",
                              "text": "flower at (480, 270)"}, video],
        "box": [{"type": "input_text", "text": "flower"},
                {"type": "input_box", "x1": 397, "y1": 194,
                 "x2": 561, "y2": 348}, video],
    }
    for name, content in variants.items():
        try:
            resp = client.responses.create(
                model="sam-3.1",
                input=[{"type": "message", "role": "user",
                        "content": content}],
                metadata={"mask_encoding": "one_bit"},
            )
            d = resp.model_dump()
            n = sum(len(c.get("text", "")) for item in d.get("output", [])
                    for c in item.get("content", []))
            print(f"{name}: {d['status']} text_chars={n}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{name} FAILED: {type(e).__name__}: {str(e)[:250]}",
                  flush=True)


if __name__ == "__main__":
    main()
