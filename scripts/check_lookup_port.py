"""Does the browser port of the attested lookup (viewer/predict-lookup.js) agree with src/agents/complete.py?

  "%USERPROFILE%\\anaconda3\\python.exe" scripts/check_lookup_port.py [n]

Cuts n fragments (default 120) from random corpus sentences the way scripts/eval_complete.py does
(start/middle/end, clean or noisy), runs both implementations and compares status and top match.
"""
import json
import random
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.agents.complete import complete, words  # noqa: E402
from integration_check import CHROME, Tab  # noqa: E402
import shutil, subprocess, tempfile, time  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def fragments(n, seed=7):
    rng = random.Random(seed)
    sents = json.loads((ROOT / "data/predict/corpus.json").read_text(encoding="utf-8"))["sentences"]
    out = []
    while len(out) < n:
        s = rng.choice(sents)["text"]
        ws = words(s)
        if len(ws) < 6:
            continue
        k = max(3, int(len(ws) * 0.4))
        where = rng.choice(["start", "middle", "end"])
        a = 0 if where == "start" else len(ws) - k if where == "end" else rng.randint(1, len(ws) - k - 1)
        frag = s[ws[a][1]:ws[a + k - 1][2]]
        noise = rng.choice(["clean", "noaccent", "drop", "cut"])
        if noise == "noaccent":
            frag = "".join(c for c in unicodedata.normalize("NFD", frag) if not unicodedata.combining(c))
        elif noise == "drop" and len(frag) > 8:
            i = rng.randint(2, len(frag) - 3)
            frag = frag[:i] + frag[i + 1:]
        elif noise == "cut":
            frag = frag[2:-2]
        out.append((frag, s))
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    fr = fragments(n)
    py = [complete(f, use_model=False) for f, _ in fr]
    prof = tempfile.mkdtemp(prefix="vxport-")
    chrome = subprocess.Popen([CHROME, "--headless=new", "--remote-debugging-port=9336", f"--user-data-dir={prof}", "--allow-file-access-from-files", "--no-first-run", "about:blank"])
    try:
        tab = Tab(9336)
        tab.send("Page.enable"); tab.send("Runtime.enable")
        tab.send("Page.navigate", url=(ROOT / "viewer/predict.html").as_uri())
        time.sleep(5)
        js = tab.js("(" + json.dumps([f for f, _ in fr]) + ").map(f=>{try{const r=window.UNSCROLL_LOOKUP.complete(f);return {status:r.status,top:r.matches&&r.matches[0]?r.matches[0].id:null,score:r.matches&&r.matches[0]?r.matches[0].score:null}}catch(e){return {error:String(e)}}})")
    finally:
        chrome.terminate(); time.sleep(1); shutil.rmtree(prof, ignore_errors=True)
    if not isinstance(js, list):
        print("browser lookup failed:", js); return 1
    same = right_py = 0
    for (frag, src), a, b in zip(fr, py, js):
        pa = (a["status"], a["matches"][0]["id"] if a["matches"] else None)
        pb = (b.get("status"), b.get("top"))
        ta = a["matches"][0]["text"] == src if a["matches"] else False
        right_py += ta
        tie = pa[0] == pb[0] and a["matches"] and b.get("score") is not None and abs(a["matches"][0]["score"] - b["score"]) < 1e-9
        if pa == pb or tie:                      # equal scores: the two sorts may order ties differently
            same += 1
        else:
            print(f"DIFF {frag[:40]!r}: python={pa} js={pb}")
    print(f"{same}/{n} fragments give the same status and top sentence; python recovers the source on {right_py}/{n}")
    return 0 if same == n else 1


if __name__ == "__main__":
    sys.exit(main())
