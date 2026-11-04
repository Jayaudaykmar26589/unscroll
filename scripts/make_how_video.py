"""Record viewer/how.html (the animated explainer) to demo/video/unscroll-how-it-works.mp4.

  "%USERPROFILE%\anaconda3\python.exe" scripts/make_how_video.py

Silent video: the captions are drawn on the canvas. Uses the headless-Chrome screencast
from make_demo_video.py; the page runs in bare mode with autoplay.
"""
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_demo_video import CHROME, DSF, H, W, Tab, sh, write_frames, BUILD  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo" / "video" / "unscroll-how-it-works.mp4"
URL = (ROOT / "viewer" / "how.html").as_uri() + "?bare=1"


def main():
    prof = tempfile.mkdtemp(prefix="vxhow-")
    chrome = subprocess.Popen([CHROME, "--headless=new", "--remote-debugging-port=9334", f"--user-data-dir={prof}",
                               "--hide-scrollbars", "--mute-audio", "--no-first-run", "--allow-file-access-from-files",
                               f"--window-size={W},{H}", "about:blank"])
    try:
        tab = Tab(9334)
        tab.send("Page.enable")
        tab.send("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=DSF, mobile=False)
        tab.send("Page.navigate", url=URL)
        time.sleep(5)
        total = tab.js("window.UNSCROLL_HOW.total") + 0.5
        tab.frames = []
        tab.send("Page.startScreencast", format="jpeg", quality=88, maxWidth=1920, maxHeight=1080, everyNthFrame=1)
        time.sleep(0.5)
        tab.frames = []
        t0 = time.perf_counter()
        tab.js("window.UNSCROLL_HOW.seek(0); window.UNSCROLL_HOW.play(true)")
        while time.perf_counter() - t0 < total:
            time.sleep(0.02)
        tab.send("Page.stopScreencast")
        frames, tab.frames = tab.frames, None
        write_frames("how", frames, t0, total)
        print(f"recorded {len(frames)} frames over {total:.1f}s")
    finally:
        chrome.terminate()
        time.sleep(1)
        shutil.rmtree(prof, ignore_errors=True)
    sh("ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(BUILD / "how" / "frames.txt"),
       "-vf", "fps=30,scale=1920:1080,format=yuv420p", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
       "-movflags", "+faststart", str(OUT))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
