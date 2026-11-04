"""Fetch LearnOpenCV SAM3 article, extract API code blocks. Stdlib only."""
import re
import urllib.request
from pathlib import Path

URL = "https://learnopencv.com/sam-3-whats-new/"


def main():
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=120).read()
    html = html.decode("utf-8", "replace")
    Path("data/sam3_article.html").write_bytes(html.encode("utf-8"))
    print(f"saved {len(html)} bytes")
    # code blocks mentioning sam-3.1 / api.meta.ai / handle_request
    blocks = re.findall(r"<pre[^>]*>(.*?)</pre>", html, re.DOTALL)
    for b in blocks:
        clean = re.sub(r"<[^>]+>", "", b)
        clean = clean.replace("&quot;", '"').replace("&#39;", "'")
        clean = clean.replace("&amp;", "&")
        if re.search(r"sam-3|api\.meta|handle_request|video_predictor",
                     clean, re.IGNORECASE):
            print("=" * 60)
            print(clean[:2500])


if __name__ == "__main__":
    main()
