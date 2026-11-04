"""Probe Meta Model API: list models + tiny sam-3.1 call. Stdlib + openai.

Key from .env (META_MODEL_API_KEY). Writes reports/api_probe.json.
Small calls only (cost negligible).
"""
import base64
import io
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_key():
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("META_MODEL_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("no key in .env")


def main():
    from openai import OpenAI
    key = load_key()
    client = OpenAI(base_url="https://api.meta.ai/v1", api_key=key,
                    timeout=120.0)
    report = {}
    try:
        models = client.models.list()
        ids = sorted(m.id for m in models.data)
        report["models"] = ids
        print(f"models ({len(ids)}):")
        for i in ids[:40]:
            print("  ", i)
    except Exception as e:  # noqa: BLE001
        report["models_error"] = f"{type(e).__name__}: {e}"
        print(f"models.list FAILED: {e}")
        ids = []

    # tiny sam-3.1 image call on real fragment mask (downscaled PNG)
    if any("sam" in i for i in ids):
        sam_id = next(i for i in ids if "sam-3.1" in i) if any(
            "sam-3.1" in i for i in ids) else next(
            i for i in ids if "sam" in i)
        try:
            from PIL import Image
            img = Image.open(ROOT / "data" / "cache" / "frag1" / "mask.png")
            img = img.convert("RGB")
            img.thumbnail((256, 256))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            resp = client.responses.create(
                model=sam_id,
                input=[{"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "papyrus layer"},
                    {"type": "input_image",
                     "image_url": f"data:image/png;base64,{b64}"},
                ]}],
                metadata={"mask_encoding": "one_bit"},
            )
            out = resp.model_dump_json()[:3000]
            report["sam_call"] = {"model": sam_id, "ok": True,
                                  "head": out}
            print(f"sam call OK ({sam_id}), head:")
            print(out[:1500])
        except Exception as e:  # noqa: BLE001
            report["sam_call"] = {"model": sam_id, "ok": False,
                                  "error": f"{type(e).__name__}: {e}"}
            print(f"sam call FAILED: {e}")
    (ROOT / "reports" / "api_probe.json").write_text(
        json.dumps(report, indent=1)[:8000])
    print("wrote reports/api_probe.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
