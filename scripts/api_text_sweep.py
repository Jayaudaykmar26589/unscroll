"""Try text prompts that might match thin CT layers. Tiny calls."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_key():
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("META_MODEL_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("no key")


def main():
    import base64
    from openai import OpenAI
    b64 = base64.b64encode(
        (ROOT / "data" / "cache" / "probe8.mp4").read_bytes()).decode()
    client = OpenAI(base_url="https://api.meta.ai/v1",
                    api_key=load_key(), timeout=300.0)
    for text in ["line", "curve", "white curve", "layer", "thin line",
                 "sheet"]:
        try:
            resp = client.responses.create(
                model="sam-3.1",
                input=[{"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": text},
                    {"type": "input_video",
                     "video_url": f"data:video/mp4;base64,{b64}"},
                ]}],
                metadata={"mask_encoding": "one_bit"},
            )
            d = resp.model_dump()
            outs = []
            for item in d.get("output", []):
                for c in item.get("content", []):
                    t = c.get("text", "")
                    outs.append(f"{c.get('type')}:{len(t)}chars:{t[:120]}")
            print(f"{text!r}: {d['status']} {outs}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{text!r} FAILED: {type(e).__name__}: {str(e)[:200]}",
                  flush=True)


if __name__ == "__main__":
    main()
