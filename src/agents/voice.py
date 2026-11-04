"""Muse Voice Transcribe (push-to-talk) for hands-free review in the workbench.

The browser records 16 kHz mono 16-bit WAV and posts it to the local server,
which calls this module; the API key never leaves the server. Keyword biasing
favours the workbench's command words.
"""
import json
import uuid
import urllib.request

from src.agents.spark import load_key

MODEL = "muse-voice-transcribe-1.0"
URL = "https://api.meta.ai/v1/asr/transcribe"
KEYWORDS = ["next", "back", "previous", "slice", "worst", "play", "pause", "stop",
            "mesh", "trace", "previous trace", "mark", "show", "hide"]


def transcribe(wav_bytes, timeout=60):
    """Return the transcript text of a WAV clip (raises on provider errors)."""
    boundary = uuid.uuid4().hex
    request = {"mode": "PUSH_TO_TALK", "model": MODEL, "audioEncoding": "WAV",
               "keywords": KEYWORDS, "languageBias": ["English", "Tamil"]}
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"request\"\r\n"
        f"Content-Type: application/json\r\n\r\n{json.dumps(request)}\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"clip.wav\"\r\n"
        f"Content-Type: audio/wav\r\n\r\n".encode() + wav_bytes + b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ]
    req = urllib.request.Request(URL, data=b"".join(parts), method="POST", headers={
        "Authorization": f"Bearer {load_key()}",
        "Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()).get("transcript", "").strip()
