"""Spark agent layer: Greek read + translation + Hieroglyphs.

Read-only curation agent (design: Spark proposes, human/golden disposes).
Offline stub when UNROLL_OFFLINE=1 or no key: deterministic placeholder
so demo/viewer paths never touch the network.
"""
import base64
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPARK_MODEL = "muse-spark-1.3"


def offline():
    if os.environ.get("UNROLL_OFFLINE") == "1":
        return True
    try:
        return not (ROOT / ".env").exists()
    except OSError:
        return True


def load_key():
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("META_MODEL_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("no META_MODEL_API_KEY in .env")


def call_spark(messages, model=SPARK_MODEL, timeout=120.0, max_retries=2):
    """Minimal OpenAI-compat chat call. Returns text (or stub tag)."""
    if offline():
        return "[OFFLINE-STUB] spark unavailable (UNROLL_OFFLINE=1 or no key)"
    from openai import OpenAI
    client = OpenAI(base_url="https://api.meta.ai/v1",
                    api_key=load_key(), timeout=timeout, max_retries=max_retries)
    resp = client.chat.completions.create(model=model, messages=messages)
    return resp.choices[0].message.content


def _img_url(png_path):
    b64 = base64.b64encode(Path(png_path).read_bytes()).decode()
    return f"data:image/png;base64,{b64}"


def read_sheet(png_path, charset="greek"):
    """Letter suggestions + confidences for a flattened sheet image."""
    if charset == "greek":
        sys = ("You read ancient Greek papyrus sheets. List visible Greek "
               "letters/words left-to-right with per-item confidence 0-1. "
               "Reply as JSON list [{text, conf}]. Mark uncertain with '?'.")
    elif charset == "hieroglyphs":
        sys = ("You read Egyptian hieroglyphs (Gardiner signs). List visible "
               "signs left-to-right with Gardiner codes + confidence 0-1. "
               "Reply as JSON list [{text, code, conf}].")
    else:
        raise ValueError(f"charset {charset}")
    if offline():
        return [{"text": "[OFFLINE-STUB]", "conf": 0.0}]
    txt = call_spark([
        {"role": "system", "content": sys},
        {"role": "user", "content": [
            {"type": "text", "text": "Read this sheet."},
            {"type": "image_url", "image_url": {"url": _img_url(png_path)}}]}])
    try:
        return json.loads(txt[txt.index("["):txt.rindex("]") + 1])
    except (ValueError, json.JSONDecodeError):
        return [{"text": txt.strip(), "conf": 0.0}]


def translate_gr(text):
    """Ancient Greek -> English gloss (short)."""
    if offline():
        return "[OFFLINE-STUB] translation unavailable"
    return call_spark([
        {"role": "system", "content": "Translate ancient Greek to concise English."},
        {"role": "user", "content": text}])


if __name__ == "__main__":
    import sys as _s
    if len(_s.argv) > 1 and _s.argv[1] == "smoke":
        os.environ["UNROLL_OFFLINE"] = "1"
        print(read_sheet("x.png"))
        print(translate_gr("χαῖρε"))
