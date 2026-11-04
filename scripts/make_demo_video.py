"""Build the 3-minute demo video (docs/video-script.md) with narration and captions.

  "%USERPROFILE%\\anaconda3\\python.exe" scripts/make_demo_video.py [--voice Mark] [--rate 8] [--only tts|record|mix]

Needs: Windows speech (OneCore voices, built in), Google Chrome, ffmpeg on PATH, and the
`websocket-client` package (Anaconda ships it). Records the hosted pages
(GitHub Pages) in headless Chrome, so nothing takes over the screen.
Writes demo/video/unscroll-demo.mp4 and unscroll-demo.srt; work files go to demo/video/build/.
"""
import argparse
import base64
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import requests
import websocket

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo" / "video"
BUILD = OUT / "build"
SITE = "https://jayaudaykmar26589.github.io/unscroll/viewer/"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
W, H, DSF = 1536, 864, 1.25          # CSS viewport; frames come out 1920x1080
LEAD, TAIL = 0.35, 0.75               # silence before / after each shot's narration
AGENT_LOG = ROOT / "reports/agent/20231210121321_20261031T180912.jsonl"
SEAM_PNG = ROOT / "reports/agent/seam_20231210121321_reseed_6159.png"

NOT_OURS = """<div id="vx-credit" style="position:fixed;right:28px;bottom:120px;width:430px;z-index:99;background:rgba(10,18,19,.94);
border:1px solid #2c4446;border-radius:10px;padding:18px 20px;font:14px/1.5 'Segoe UI',sans-serif;color:#d6e2e2;box-shadow:0 8px 30px rgba(0,0,0,.5)">
<div style="color:#5fd4c4;font-size:12px;letter-spacing:.08em;margin-bottom:6px">BUILT DURING THE HACKATHON (OCT 27 – NOV 4)</div>
Tracing pipeline and first-slice fix, evaluation, Muse Wonder agent, smoothing, these pages.
<div style="color:#e0a35f;font-size:12px;letter-spacing:.08em;margin:12px 0 6px">NOT OURS, CREDITED</div>
SAM 3.1 (Meta) · Muse Spark, Muse Wonder (Meta Model API) · 2023 Grand Prize ink model (MIT) ·
Scroll 1 CT and hand-traced layers (EduceLab-Scrolls / Vesuvius Challenge, CC BY-NC 4.0)
<div style="margin-top:12px;padding-top:10px;border-top:1px solid #2c4446;color:#fff"><b>Unscroll has not read any new text.</b></div></div>"""

END_CARD = """<div style="position:fixed;inset:0;z-index:100;background:#0b1213;display:flex;flex-direction:column;align-items:center;
justify-content:center;font-family:'Segoe UI',sans-serif;color:#e8f0f0;text-align:center;animation:vxf .8s ease both">
<style>@keyframes vxf{from{opacity:0}to{opacity:1}}</style>
<div style="font-size:64px;font-weight:600;letter-spacing:-.01em">Unscroll</div>
<div style="font-size:22px;color:#9fb3b3;margin-top:10px">Click a sheet. See whether the AI stayed on it, and how far it is from reading.</div>
<div style="font-size:17px;margin-top:40px;line-height:1.9;color:#5fd4c4">github.com/Jayaudaykmar26589/unscroll<br>jayaudaykmar26589.github.io/unscroll/viewer/story.html</div>
<div style="font-size:13px;color:#6f8585;margin-top:36px">Meta Global AI Developer Hackathon · Indie track · Push the Frontier</div></div>"""

# Each shot: narration (spoken and captioned), on-screen label, page, and actions keyed to the
# phrase in the narration they should line up with ("@0" = shot start, "@end-4" = 4 s before the end).
SHOTS = [
    dict(id="01_stakes", page="story.html", label="Story page · chapter 1 · a scroll is one long rolled sheet",
         say="In 79 AD, Vesuvius carbonized a library. The scrolls can't be opened, only scanned. "
             "To read one, you follow a single papyrus sheet through thousands of CT slices, "
             "where every sheet is crushed against its neighbours.",
         acts=[("@0", "window.scrollTo(0,0)"), ("The scrolls", "vxScroll('c1')"),
               ("To read one", "vxSlide('crush',0,100,5000)")]),
    dict(id="02_rules", page="story.html", label="Real Scroll 1 CT slice · what is ours and what is not",
         say="Everything in this video was built during the hackathon, except SAM 3.1, the Muse models, "
             "the 2023 Grand Prize ink model and the Vesuvius data, which I'll name when they appear. "
             "And to be clear: Unscroll has not read any new text. "
             "It measures how close an AI trace gets to being readable.",
         acts=[("@0", "vxScroll('c1',0);vxSet('crush',100);vxClick('realbtn')"), ("Everything", "vxOverlay(NOT_OURS)")]),
    dict(id="03_trace", page="story.html", label="Story page · chapter 2 · 50 real slices · pink = hand-traced, orange and teal = two early traces (28 Oct)",
         say="I click one sheet, and SAM 3.1 follows it through the slices like an object in a video. "
             "Pink is the sheet a person traced by hand. These are two early traces from day two: "
             "orange slides further and further off, and teal is closer but still off. "
             "The meter turns teal only within 40 microns, about half a hair.",
         acts=[("@0", "vxScroll('c2',0);vxCheck('k_old')"), ("follows it", "vxClick('play')"), ("The meter", "vxHi('dist')")]),
    dict(id="03b_fix", page="evaluate.html", label="Evaluation page · the first-slice fix · green = hand-traced, red = Unscroll, yellow = click",
         wait=9,
         say="The cause was the first slice: it took the brightest path through SAM's mask, which can jump to the "
             "neighbouring sheet. Unscroll's fix walks outward from the click instead. "
             "On this click: 23 voxels off before, under 2 after.",
         acts=[("@0", "vxZoom('img[src*=\"20231210121321\"]')")]),
    dict(id="04_test", page="story.html", label="Story page · chapter 4 · 74 clicks never used while building the fix",
         say="We tested it on 74 clicks we never used while building the fix. The share of the trace "
             "within 40 microns of the hand-traced sheet doubles, from 25 to 50 percent: better on 50 clicks, "
             "worse on 19. Against a classic optical-flow tracer, the gain is smaller, four and a half points, "
             "and on our 15 development clicks it's a tie. We report both.",
         acts=[("@0", "vxScroll('c4',0)"), ("The share", "vxClick('runtest')"), ("Against a classic", "vxClick('runflow')")]),
    dict(id="05_agent", page="agent", label="Muse Wonder agent log, 31 Oct 2026 · reports/agent/…T180912.jsonl",
         say="When a trace stops, a Muse Wonder agent takes over. It calls tools to re-seed from the last good slice, "
             "launch GPU runs on Kaggle and score the result. Then it checks its own seams: continuity numbers, "
             "plus Muse Spark looking at the image. Here it caught its first re-seed jumping to another sheet, "
             "threw out that run and the two built on it, and wrote down why.",
         acts=[("@0", "window.scrollTo(0,0)"), ("Then it checks", "vxScrollTo(document.getElementById('seams').offsetTop-90,2500)"),
               ("Here it caught", "vxHi('s0')"), ("threw out", "vxScrollTo(document.getElementById('excl').offsetTop-90,2000)")]),
    dict(id="06_drift", page="story.html", label="Story page · chapter 3 · ink map from the 2023 Grand Prize model (MIT)",
         say="Why does staying on the sheet matter? Ink here is invisible to the eye; the Grand Prize model finds it "
             "from the surface. If the surface is off by a steady five voxels, the ink survives: agreement 0.79. "
             "If it wobbles by five voxels from slice to slice, it's gone: 0.01. "
             "So the enemy is wobble, not distance.",
         acts=[("@0", "vxScroll('c3',0)"), ("If the surface", "vxSlide('off',0,5,2500)"), ("If it wobbles", "vxClick('wobbly')")]),
    dict(id="07_smooth", page="evaluate.html", label="Evaluation page · our own trace on the known-text sheet · ink maps by the Grand Prize model",
         say="So we smooth our own trace across slices. On a sheet with known text, agreement with the official ink map "
             "rises from 0.21 to 0.73, close to the 0.83 the official surface gets. But the maps show broad strokes, "
             "not letters, and fewer scholar clicks cover less of the sheet. That's the measured gap that's left.",
         acts=[("@0", "vxScroll('smoothsection',0)"), ("But the maps", "vxScrollBy(260,1500)"),
               ("fewer scholar", "vxScroll('costsection');document.querySelector('#costsection details').open=true")]),
    dict(id="08_close", page="story.html", label="Story page · chapter 5 · what Unscroll is, and what it isn't",
         say="Unscroll: click a sheet, see whether the AI stayed on it, and see how far it is from reading. "
             "The code, the data and every number are public.",
         acts=[("@0", "vxScroll('c5',0)"), ("@0.6", "vxFlip(0)"), ("see whether", "vxFlip(1)"), ("see how far", "vxFlip(2)"),
               ("The code", "vxOverlay(END_CARD)")]),
]

HELPERS = """
window.vxEase=t=>t<.5?2*t*t:1-Math.pow(-2*t+2,2)/2;
window.vxScrollTo=(y,ms=1600)=>{const y0=scrollY,t0=performance.now();(function f(n){const t=Math.min(1,(n-t0)/ms);scrollTo(0,y0+(y-y0)*vxEase(t));if(t<1)requestAnimationFrame(f)})(t0)};
window.vxScroll=(id,ms=1600)=>{const e=document.getElementById(id);const y=e.getBoundingClientRect().top+scrollY-24;ms?vxScrollTo(y,ms):scrollTo(0,y)};
window.vxScrollBy=(d,ms=1500)=>vxScrollTo(scrollY+d,ms);
window.vxSet=(id,v)=>{const e=document.getElementById(id);e.value=v;e.dispatchEvent(new Event('input',{bubbles:true}))};
window.vxSlide=(id,a,b,ms)=>{const t0=performance.now();(function f(n){const t=Math.min(1,(n-t0)/ms);vxSet(id,Math.round(a+(b-a)*vxEase(t)));if(t<1)requestAnimationFrame(f)})(t0)};
window.vxClick=id=>document.getElementById(id).click();
window.vxCheck=id=>{const e=document.getElementById(id);if(!e.checked){e.checked=true;e.dispatchEvent(new Event('change',{bubbles:true}))}};
window.vxZoom=sel=>{const e=document.querySelector(sel);const fig=e.closest('figure')||e.parentElement;const cap=(fig.querySelector('figcaption')||{}).textContent||'';
 const d=document.createElement('div');d.style.cssText='position:fixed;inset:0;z-index:98;background:#0b1213;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:16px';
 const im=new Image();im.src=e.currentSrc||e.src;im.style.cssText='width:1240px;max-height:700px;object-fit:contain;border-radius:6px;border:1px solid #233534';
 const c=document.createElement('div');c.textContent=cap;c.style.cssText="font:16px 'Segoe UI',sans-serif;color:#9fb3b3";d.append(im,c);document.body.appendChild(d)};
window.vxShow=sel=>{const e=document.querySelector(sel);const r=e.getBoundingClientRect();scrollTo(0,r.top+scrollY-(innerHeight-r.height)/2);e.style.outline='3px solid #e0a35f';e.style.outlineOffset='3px'};
window.vxFlip=i=>document.querySelectorAll('.flipcard button')[i].click();
window.vxOverlay=h=>{const d=document.createElement('div');d.innerHTML=h;document.body.appendChild(d)};
window.vxHi=id=>{const e=document.getElementById(id);e.style.outline='3px solid #e0a35f';e.style.outlineOffset='4px'};
"""


NUDGE = ("(()=>{let e=document.getElementById('vxn');if(!e){e=document.createElement('div');e.id='vxn';"
         "document.documentElement.appendChild(e)}e.style.cssText='position:fixed;left:0;top:0;width:1px;height:1px;"
         "z-index:999;background:rgba(0,0,0,'+(0.01+Math.random()*0.01)+')'})()")


def sh(*a, **k):
    return subprocess.run(a, check=True, capture_output=True, text=True, **k)


def duration(f):
    return float(sh("ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(f)).stdout)


def spoken(text):
    t = text.replace("SAM 3.1", "Sam three point one").replace("79 AD", "seventy-nine A D")
    t = re.sub(r"\b0\.(\d)(\d)\b", lambda m: f"zero point {m[1]} {m[2]}", t)
    t = t.replace("CT", "C T").replace("Kaggle", "Kaggle")
    return html.escape(t)


# ---------- narration ----------
def make_tts(voice, rate):
    BUILD.mkdir(parents=True, exist_ok=True)
    for s in SHOTS:
        ssml = (f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
                f'<voice name="Microsoft {voice}"><prosody rate="+{rate}%">{spoken(s["say"])}</prosody></voice></speak>')
        x = BUILD / f"{s['id']}.xml"
        x.write_text(ssml, encoding="utf-8")
        sh("powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "scripts/tts_windows.ps1"),
           "-Ssml", str(x), "-Out", str(BUILD / f"{s['id']}.wav"))
    durs = {s["id"]: duration(BUILD / f"{s['id']}.wav") for s in SHOTS}
    (BUILD / "durations.json").write_text(json.dumps(durs, indent=1))
    total = sum(d + LEAD + TAIL for d in durs.values())
    print("narration", {k: round(v, 1) for k, v in durs.items()}, f"total video {total:.1f}s")
    return durs


def phrase_time(s, key, dur):
    """Seconds from shot start at which the narration reaches this phrase (by character share)."""
    if key.startswith("@"):
        v = key[1:]
        return LEAD + dur + TAIL - float(v[4:]) if v.startswith("end-") else float(v)
    i = s["say"].find(key)
    assert i >= 0, (s["id"], key)
    return LEAD + dur * i / len(s["say"])


# ---------- the agent log page ----------
def agent_page():
    rows = [json.loads(l) for l in AGENT_LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
    out, seam_i = [], 0
    for r in rows:
        if r["kind"] == "model":
            if r.get("content"):
                out.append(f'<div class="m"><span class="who">Muse Wonder</span>{html.escape(r["content"])}</div>')
            calls = [t["name"] for t in r.get("tool_calls") or [] if t["name"] != "finish"]
            if calls:
                out.append('<div class="calls">' + "".join(f"<code>{html.escape(c)}()</code>" for c in calls) + "</div>")
        elif r.get("name") == "review_seam":
            res, n = r["result"], r["result"]["numbers"]
            sp = res["spark"]
            if seam_i == 0:
                out.append('<div id="seams"></div>')
            out.append(f'<div class="t seam" id="s{seam_i}"><b>review_seam · {res["run"]}</b>'
                       f'<div class="nums">slices {n["slice_before"]}→{n["slice_after"]} · median distance {n["median_distance_vx"]} vx · '
                       f'length {n["length_before_vx"]}→{n["length_after_vx"]} vx (×{n["length_ratio"]})</div>'
                       f'<div class="spark">Muse Spark: same sheet = <b class="no">false</b> (confidence {sp["confidence"]})<br>'
                       f'<i>“{html.escape(sp["reason"])}”</i></div></div>')
            seam_i += 1
    ex = json.loads((ROOT / "reports/flatten/excluded_20231210121321.json").read_text())
    out.append('<div class="t excl" id="excl"><b>exclude_run × 3</b>' + "".join(
        f'<div class="nums"><span class="no">{k}</span> {html.escape(v[:150])}…</div>' for k, v in ex.items()) + "</div>")
    out.append('<div class="m"><span class="who">Result</span>The patch keeps only the verified 58 slices '
               '(6101–6159); the three re-seeded runs are excluded with their reasons.</div>')
    seam = "data:image/png;base64," + base64.b64encode(SEAM_PNG.read_bytes()).decode()
    page = f"""<!doctype html><meta charset="utf-8"><title>Agent log</title><style>
body{{margin:0;background:#0b1213;color:#d6e2e2;font:15px/1.5 'Segoe UI',sans-serif}}
.wrap{{display:grid;grid-template-columns:minmax(0,1fr) 640px;gap:36px;padding:36px 44px}}
h1{{font-size:24px;font-weight:600;margin:0 0 4px}} .sub{{color:#7f9696;font-size:13px;margin-bottom:22px}}
.m{{background:#13201f;border:1px solid #233534;border-radius:10px;padding:12px 16px;margin:12px 0}}
.who{{display:block;color:#5fd4c4;font-size:12px;letter-spacing:.06em;text-transform:uppercase;margin-bottom:3px}}
.calls code{{display:inline-block;background:#1b2a3a;color:#9cc7ff;border-radius:5px;padding:3px 9px;margin:2px 6px 2px 0;font:13px Consolas,monospace}}
.t{{border-left:3px solid #3e5e60;padding:10px 16px;margin:14px 0;background:#0f1919;border-radius:0 8px 8px 0}}
.t b{{font:600 14px Consolas,monospace;color:#e8f0f0}} .nums{{color:#a9bcbc;font-size:13.5px;margin-top:4px}}
.spark{{margin-top:6px;font-size:14px}} .spark i{{color:#c5d3d3}} .no{{color:#ff8f8f}}
.excl{{border-left-color:#e06c6c}} .side{{position:sticky;top:36px;align-self:start}}
.side img{{width:100%;border-radius:8px;border:1px solid #233534}} .cap{{font-size:13px;color:#9fb3b3;margin-top:10px}}
</style><div class="wrap"><div><h1>Muse Wonder checks its own re-seeds</h1>
<div class="sub">Agent log of 31 Oct 2026 · tool calls and results as recorded in {AGENT_LOG.relative_to(ROOT).as_posix()}</div>
{''.join(out)}<div style="height:600px"></div></div>
<div class="side"><img src="{seam}"><div class="cap">Seam of the first re-seed. Left: last good slice 6159, the sheet in green.
Right: the re-seeded trace at 6161 in red runs along a different band, 4.4× longer and a median 181 voxels away.
Muse Spark judged it a different sheet (confidence 0.82).</div></div></div>"""
    f = BUILD / "agent.html"
    f.write_text(page, encoding="utf-8")
    return f.as_uri()


# ---------- headless Chrome over the DevTools protocol ----------
class Tab:
    def __init__(self, port):
        for _ in range(50):
            try:
                ws = [t for t in requests.get(f"http://127.0.0.1:{port}/json").json() if t["type"] == "page"][0]["webSocketDebuggerUrl"]
                break
            except Exception:
                time.sleep(0.2)
        self.ws = websocket.create_connection(ws, max_size=None)
        self.n, self.wait, self.frames, self.lock = 0, {}, None, threading.Lock()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        while True:
            try:
                m = json.loads(self.ws.recv())
            except Exception:
                return
            if m.get("method") == "Page.screencastFrame":
                p = m["params"]
                if self.frames is not None:
                    self.frames.append((time.perf_counter(), p["data"]))
                self.send("Page.screencastFrameAck", sessionId=p["sessionId"], wait=False)
            elif "id" in m and m["id"] in self.wait:
                self.wait[m["id"]].append(m)

    def send(self, method, wait=True, **params):
        with self.lock:
            self.n += 1
            i = self.n
            box = self.wait[i] = []
            self.ws.send(json.dumps({"id": i, "method": method, "params": params}))
        if not wait:
            return None
        t0 = time.time()
        while not box and time.time() - t0 < 30:
            time.sleep(0.005)
        r = self.wait.pop(i)[0] if box else {}
        if "error" in r:
            raise RuntimeError(f"{method}: {r['error']}")
        return r.get("result")

    def js(self, code):
        r = self.send("Runtime.evaluate", expression=code, awaitPromise=True)
        if r and r.get("exceptionDetails"):
            raise RuntimeError(r["exceptionDetails"])
        return (r or {}).get("result", {}).get("value")


def record(durs):
    prof = tempfile.mkdtemp(prefix="vxchrome-")
    port = 9333
    chrome = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={prof}",
                               "--hide-scrollbars", "--mute-audio", "--no-first-run", "--disable-extensions",
                               f"--window-size={W},{H}", "about:blank"])
    try:
        tab = Tab(port)
        tab.send("Page.enable")
        tab.send("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=DSF, mobile=False)
        tab.send("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": "dark"}])
        agent_url = agent_page()
        consts = f"window.NOT_OURS={json.dumps(NOT_OURS)};window.END_CARD={json.dumps(END_CARD)};"
        for s in SHOTS:
            d = durs[s["id"]]
            total = LEAD + d + TAIL
            url = agent_url if s["page"] == "agent" else SITE + s["page"] + "?v=video"
            tab.send("Page.navigate", url=url)
            time.sleep(s.get("wait", 4.0))                    # page, data scripts and images load
            tab.js(HELPERS + consts)
            acts = sorted((phrase_time(s, k, d), code) for k, code in s["acts"])
            # the @0 action is setup: run it before recording starts so the shot opens on the right view
            while acts and acts[0][0] == 0:
                tab.js(acts.pop(0)[1])
            time.sleep(0.8)
            tab.frames = []
            tab.send("Page.startScreencast", format="jpeg", quality=88, maxWidth=1920, maxHeight=1080, everyNthFrame=1)
            time.sleep(0.5)
            tab.frames = []                                   # drop stale frames from before the setup
            t0 = time.perf_counter()
            tab.js(NUDGE)                                     # one tiny repaint so a frame is sent at t0
            for at, code in acts:
                while time.perf_counter() - t0 < at:
                    time.sleep(0.005)
                tab.js(code)
            while time.perf_counter() - t0 < total:
                time.sleep(0.01)
            tab.send("Page.stopScreencast")
            frames, tab.frames = tab.frames, None
            write_frames(s["id"], frames, t0, total)
            print(f"recorded {s['id']}: {len(frames)} frames over {total:.1f}s")
    finally:
        chrome.terminate()
        time.sleep(1)
        shutil.rmtree(prof, ignore_errors=True)


def write_frames(sid, frames, t0, total):
    d = BUILD / sid
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    assert frames, f"no frames for {sid}"
    lines = []
    for k, (t, data) in enumerate(frames):
        (d / f"{k:05d}.jpg").write_bytes(base64.b64decode(data))
        start = 0.0 if k == 0 else t - t0
        end = frames[k + 1][0] - t0 if k + 1 < len(frames) else total
        lines.append(f"file '{k:05d}.jpg'\nduration {max(end - start, 0.001):.4f}")
    lines.append(f"file '{len(frames) - 1:05d}.jpg'")
    (d / "frames.txt").write_text("\n".join(lines) + "\n")


# ---------- captions and mix ----------
def ass_time(t):
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def srt_time(t):
    return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{int(t % 60):02d},{int(t * 1000 % 1000):03d}"


def chunks(text, limit=88):
    parts = re.split(r"(?<=[.:;?])\s+|(?<=,)\s+(?=\S{4,})", text)
    out, cur = [], ""
    for p in parts:
        if cur and len(cur) + 1 + len(p) > limit:
            out.append(cur)
            cur = p
        else:
            cur = (cur + " " + p).strip()
    return out + [cur] if cur else out


def mix(durs):
    ass = ["[Script Info]", "ScriptType: v4.00+", "PlayResX: 1920", "PlayResY: 1080", "",
           "[V4+ Styles]",
           "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, "
           "StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
           "Style: Cap,Segoe UI,40,&H00FFFFFF,&H00FFFFFF,&H00000000,&H50000000,0,0,0,0,100,100,0,0,3,14,0,2,160,160,46,1",
           "Style: Label,Segoe UI Semibold,25,&H00C4D45F,&H00FFFFFF,&H00000000,&H38000000,0,0,0,0,100,100,0,0,3,10,0,7,40,40,34,1",
           "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    srt, t, n = [], 0.0, 0
    parts = []
    for s in SHOTS:
        d = durs[s["id"]]
        total = LEAD + d + TAIL
        clip = BUILD / f"{s['id']}.mp4"
        sh("ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(BUILD / s["id"] / "frames.txt"),
           "-i", str(BUILD / f"{s['id']}.wav"),
           "-filter_complex", f"[0:v]fps=30,scale=1920:1080,format=yuv420p,trim=duration={total:.3f}[v];"
                              f"[1:a]adelay={int(LEAD * 1000)}|{int(LEAD * 1000)},apad,atrim=duration={total:.3f},"
                              f"aformat=sample_rates=48000:channel_layouts=stereo[a]",
           "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-c:a", "aac", "-b:a", "160k",
           str(clip))
        parts.append(clip)
        if s["id"] != "08_close":
            ass.append(f"Dialogue: 0,{ass_time(t + 0.1)},{ass_time(t + total - 0.1)},Label,,0,0,0,,{s['label']}")
        cs = chunks(s["say"])
        a = t + LEAD
        for c in cs:
            dt = d * len(c) / sum(len(x) for x in cs)
            ass.append(f"Dialogue: 1,{ass_time(a)},{ass_time(a + dt - 0.05)},Cap,,0,0,0,,{c}")
            n += 1
            srt.append(f"{n}\n{srt_time(a)} --> {srt_time(a + dt - 0.05)}\n{c}\n")
            a += dt
        t += total
    (BUILD / "captions.ass").write_text("\n".join(ass) + "\n", encoding="utf-8")
    (OUT / "unscroll-demo.srt").write_text("\n".join(srt), encoding="utf-8")
    lst = BUILD / "parts.txt"
    lst.write_text("".join(f"file '{p.name}'\n" for p in parts))
    sh("ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(BUILD / "joined.mp4"))
    # libass needs a path without a drive colon: run from BUILD and pass the bare name
    sh("ffmpeg", "-y", "-v", "error", "-i", "joined.mp4", "-vf", "ass=captions.ass", "-c:v", "libx264", "-preset", "medium",
       "-crf", "20", "-c:a", "copy", "-movflags", "+faststart", str(OUT / "unscroll-demo.mp4"), cwd=BUILD)
    print(f"wrote {OUT / 'unscroll-demo.mp4'} ({t:.1f}s) and unscroll-demo.srt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="Mark")
    ap.add_argument("--rate", type=int, default=8)
    ap.add_argument("--only", choices=["tts", "record", "mix"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.only in (None, "tts"):
        durs = make_tts(a.voice, a.rate)
    else:
        durs = json.loads((BUILD / "durations.json").read_text())
    if a.only in (None, "record"):
        record(durs)
    if a.only in (None, "mix"):
        mix(durs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
