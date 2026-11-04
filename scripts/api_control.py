"""Control: natural video + matching text. If masks appear, API path
is fine and CT+text is the mismatch. Tiny calls."""
import base64
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLOWER = ("https://interactive-examples.mdn.mozilla.net/media/"
          "cc0-videos/flower.mp4")


def load_key():
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("META_MODEL_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("no key")


def main():
    from openai import OpenAI
    dest = ROOT / "data" / "cache" / "flower.mp4"
    if not dest.exists():
        req = urllib.request.Request(FLOWER,
                                     headers={"User-Agent": "unscroll"})
        with urllib.request.urlopen(req, timeout=120) as r, \
                open(dest, "wb") as f:
            f.write(r.read())
    print(f"flower: {dest.stat().st_size} bytes")
    b64 = base64.b64encode(dest.read_bytes()).decode()
    client = OpenAI(base_url="https://api.meta.ai/v1",
                    api_key=load_key(), timeout=300.0)
    for text in ["flower", "red flower"]:
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
                    outs.append(f"{c.get('type')}:{len(t)}:{t[:200]}")
            print(f"{text!r}: {d['status']} {outs}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{text!r} FAILED: {type(e).__name__}: {str(e)[:300]}",
                  flush=True)


if __name__ == "__main__":
    main()
