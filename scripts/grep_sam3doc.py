"""Grep saved article HTML for prompt-format details. Stdlib only."""
import re
from pathlib import Path

html = Path("data/sam3_article.html").read_text(encoding="utf-8")
text = re.sub(r"<[^>]+>", " ", html)
text = re.sub(r"\s+", " ", text)
for pat in ["input_image", "point", "base64", "mask_encoding",
            "one_bit", "15,000", "16 tracked", "limits"]:
    idxs = [m.start() for m in re.finditer(re.escape(pat), text)][:4]
    for i in idxs:
        print(f"--- {pat} ---")
        print(text[max(0, i - 200):i + 300].strip()[:500])
        print()
