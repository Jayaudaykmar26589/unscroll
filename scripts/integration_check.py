"""Click-through integration check of every page in viewer/, in headless Chrome.

  "%USERPROFILE%\\anaconda3\\python.exe" scripts/integration_check.py                 # local files (file://)
  "%USERPROFILE%\\anaconda3\\python.exe" scripts/integration_check.py https://jayaudaykmar26589.github.io/unscroll/viewer/
  "%USERPROFILE%\\anaconda3\\python.exe" scripts/integration_check.py http://127.0.0.1:8765/

For each page: load it, record uncaught exceptions, console errors and failed or 4xx/5xx
requests, click every enabled button and example chip, drive the Predict page with a real
fragment, follow every in-site link and hash anchor, and report. Exit code 1 on any failure.
Needs Google Chrome and the `websocket-client` package (Anaconda has it).
"""
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
import websocket

ROOT = Path(__file__).resolve().parents[1]
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PAGES = ["home.html", "story.html", "how.html", "explore.html", "agent.html", "evaluate.html", "index.html", "read.html", "predict.html"]
# buttons that call a model service or the microphone: not clicked (they are disabled on static hosting anyway)
SKIP = {"recognizeLive", "recognizeLocal", "translateLive", "voiceBtn"}
IGNORE_FAIL = ("favicon.ico",)


class Tab:
    def __init__(self, port):
        for _ in range(60):
            try:
                ws = [t for t in requests.get(f"http://127.0.0.1:{port}/json").json() if t["type"] == "page"][0]["webSocketDebuggerUrl"]
                break
            except Exception:
                time.sleep(0.2)
        self.ws = websocket.create_connection(ws, max_size=None)
        self.n, self.wait, self.events, self.lock = 0, {}, [], threading.Lock()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        while True:
            try:
                m = json.loads(self.ws.recv())
            except Exception:
                return
            if "id" in m and m["id"] in self.wait:
                self.wait[m["id"]].append(m)
            elif "method" in m:
                self.events.append(m)

    def send(self, method, **params):
        with self.lock:
            self.n += 1
            i = self.n
            box = self.wait[i] = []
            self.ws.send(json.dumps({"id": i, "method": method, "params": params}))
        t0 = time.time()
        while not box and time.time() - t0 < 30:
            time.sleep(0.005)
        r = self.wait.pop(i)[0] if box else {}
        return r.get("result", r.get("error"))

    def js(self, code):
        r = self.send("Runtime.evaluate", expression=code, awaitPromise=True, returnByValue=True)
        if r and r.get("exceptionDetails"):
            return {"__error__": r["exceptionDetails"].get("text") or str(r["exceptionDetails"])}
        return (r or {}).get("result", {}).get("value")


def problems(tab, since):
    out = []
    for e in tab.events[since:]:
        m, p = e["method"], e["params"]
        if m == "Runtime.exceptionThrown":
            d = p["exceptionDetails"]
            out.append("exception: " + (d.get("exception", {}).get("description") or d.get("text", ""))[:200])
        elif m == "Runtime.consoleAPICalled" and p["type"] == "error":
            out.append("console.error: " + " ".join(str(a.get("value", a.get("description", "")))[:120] for a in p["args"]))
        elif m == "Network.loadingFailed" and not p.get("canceled"):
            out.append("request failed: " + p.get("errorText", ""))
        elif m == "Network.responseReceived" and p["response"]["status"] >= 400:
            url = p["response"]["url"]
            if not url.endswith(IGNORE_FAIL):
                out.append(f"HTTP {p['response']['status']}: {url}")
    return out


def check_page(tab, base, page):
    url = urljoin(base, page)
    since = len(tab.events)
    tab.send("Page.navigate", url=url)
    time.sleep(4.5)
    report = {"page": page, "problems": [], "clicked": 0, "notes": []}
    title = tab.js("document.title")
    if not title or title == "Error" or tab.js("document.body.innerText.length") < 200:
        report["problems"].append("page did not render (title %r)" % title)
    # click every enabled button (except model-service ones), one at a time, and watch for errors
    if page == "predict.html":                                 # never call Muse Spark from this check
        tab.js("(()=>{const c=document.getElementById('useModel');if(c&&!c.disabled)c.checked=false})()")
    # pages re-render their buttons, so pick the i-th button afresh each time
    count = tab.js("document.querySelectorAll('button').length") or 0
    for i in range(count):
        before = len(tab.events)
        r = tab.js(f"(()=>{{const b=document.querySelectorAll('button')[{i}];if(!b)return 'gone';if(b.disabled)return 'disabled';"
                   f"if({json.dumps(sorted(SKIP))}.includes(b.id))return 'skipped';b.click();return 'ok:'+(b.id||b.textContent.trim().slice(0,30))}})()")
        time.sleep(0.35)
        if not str(r).startswith("ok:"):
            continue
        report["clicked"] += 1
        bad = problems(tab, before)
        if bad:
            report["problems"].append(f"button {r[3:]!r}: {bad}")
    if page == "predict.html":
        def idle():
            for _ in range(60):
                if not tab.js("document.getElementById('go').disabled"):
                    return
                time.sleep(0.5)
        for frag, expect in (("πρὸς ἡμᾶς εἶναι τὸν θάνα", "attested"), ("ta spania pantos kai", "attested"), ("ὁ δὲ ἀνεξέταστος βίος", "not found")):
            idle()
            tab.js(f"document.getElementById('frag').value={json.dumps(frag)};document.getElementById('go').click()")
            time.sleep(0.6)
            idle()                                                # the server's first lookup loads the corpus
            res = tab.js("({badge:(document.querySelector('#result .badge')||{}).textContent||'',sent:!!document.querySelector('#result .sent'),status:document.getElementById('status').textContent})")
            ok = (expect == "attested" and res["sent"] and "Attested" in res["badge"]) or (expect == "not found" and ("Not found" in res["badge"] or "conjecture" in res["badge"].lower()))
            report["notes"].append(f"predict {frag[:18]!r} -> badge={res['badge']!r} sent={res['sent']} status={res['status'][:60]!r}")
            if not ok:
                report["problems"].append(f"predict {frag!r}: expected {expect}, got {res}")
    if page == "story.html":
        tab.js("document.getElementById('zslider').value=30;document.getElementById('zslider').dispatchEvent(new Event('input'));document.getElementById('off').value=5;document.getElementById('off').dispatchEvent(new Event('input'))")
        time.sleep(0.3)
        d = tab.js("document.getElementById('dist').textContent+' | '+document.getElementById('corr').textContent")
        if "µm" not in d:
            report["problems"].append("story slice meter empty: " + d)
    if page == "explore.html":
        time.sleep(1.5)
        r = tab.js("(()=>{const rows=document.querySelectorAll('#rows button');const k=rows[5]&&rows[5].dataset.k;rows[5].click();return k})()")
        time.sleep(2.5)
        st = tab.js("({stats:document.getElementById('stats').textContent,z:document.getElementById('zout').textContent,dl:document.getElementById('dl').getAttribute('href'),"
                    "path:document.querySelectorAll('#chart path').length,cells:document.querySelectorAll('#cells button').length,caught:document.getElementById('caught').textContent})")
        report["notes"].append(f"explore {r}: dl={st['dl']} chart paths={st['path']} cells={st['cells']} caught={st['caught']!r}")
        if "on the sheet" not in st["stats"] or st["path"] < 2 or st["cells"] != 30 or not str(st["dl"]).endswith(".tifxyz.zip"):
            report["problems"].append(f"explore did not load a click: {st}")
        tab.js("document.querySelector('#pset button[data-s=heldout]').click()")
        tab.js("document.getElementById('budget').value=15;document.getElementById('budget').dispatchEvent(new Event('input'))")
        if "9 of 15" not in tab.js("document.getElementById('caught').textContent"):
            report["problems"].append("review planner: 15 of 30 held-out should catch 9 of 15 (reports/triage/triage.json)")
        ok = tab.js(f"fetch({json.dumps(st['dl'])}).then(r=>r.ok?r.arrayBuffer().then(b=>b.byteLength):-r.status).catch(e=>'fetch blocked: '+e)")
        if isinstance(ok, (int, float)) and ok < 1000:
            report["problems"].append(f"tifxyz download {st['dl']}: {ok}")
        report["notes"].append(f"tifxyz bytes: {ok}")
    if page == "agent.html":
        tab.js("document.querySelectorAll('#sess button')[2].click()")
        time.sleep(0.5)
        for _ in range(9):
            tab.js("document.getElementById('next').click()")
        time.sleep(0.6)
        st = tab.js("({note:document.getElementById('statenote').textContent,focus:document.getElementById('focus').textContent,events:document.querySelectorAll('#feed .ev').length})")
        report["notes"].append(f"agent step 10: {st['note'][:80]!r}")
        if "kept 58" not in st["note"] or st["events"] < 10:
            report["problems"].append(f"agent replay state wrong at step 10: {st}")
    if page == "how.html":
        tab.js("window.UNSCROLL_HOW.seek(70)")
        time.sleep(0.5)
        if not tab.js("document.getElementById('dtitle').textContent"):
            report["problems"].append("how.html scene detail did not update")
    # in-site links and hash anchors
    links = tab.js("[...document.querySelectorAll('a[href]')].map(a=>a.getAttribute('href')).filter(h=>h&&!/^(https?:|mailto:|#$)/.test(h))") or []
    for h in sorted(set(links)):
        target, _, frag = h.partition("#")
        target = target.split("?")[0]
        if target and target not in PAGES and not target.startswith(("eval/", "assets/", "explore/")):
            report["notes"].append("link to non-page " + h)
        if frag:
            if target in ("", page):
                ok = tab.js(f"(()=>{{const e=document.getElementById({json.dumps(frag)});return e?(e.hidden?'hidden':'ok'):'missing'}})()")
                if ok != "ok":
                    report["problems"].append(f"anchor #{frag} {ok}")
            else:
                report["notes"].append(f"anchor {h} checked on its page")
    report["problems"] += [p for p in problems(tab, since) if p not in report["problems"] and not any(p in x for x in report["problems"])]
    return report


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    base = sys.argv[1] if len(sys.argv) > 1 else (ROOT / "viewer").as_uri() + "/"
    prof = tempfile.mkdtemp(prefix="vxcheck-")
    chrome = subprocess.Popen([CHROME, "--headless=new", "--remote-debugging-port=9335", f"--user-data-dir={prof}", "--allow-file-access-from-files",
                               "--no-first-run", "--disable-extensions", "--window-size=1400,900", "about:blank"])
    failed = 0
    try:
        tab = Tab(9335)
        for d in ("Page", "Runtime", "Network", "Log"):
            tab.send(f"{d}.enable")
        for page in PAGES:
            r = check_page(tab, base, page)
            status = "FAIL" if r["problems"] else "ok"
            failed += bool(r["problems"])
            print(f"[{status}] {page}: {r['clicked']} buttons clicked")
            for n in r["notes"]:
                print("       ", n)
            for p in r["problems"]:
                print("    !! ", p)
    finally:
        chrome.terminate()
        time.sleep(1)
        shutil.rmtree(prof, ignore_errors=True)
    print(f"\n{len(PAGES) - failed}/{len(PAGES)} pages clean on {base}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
