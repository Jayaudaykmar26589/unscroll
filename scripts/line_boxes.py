"""Locate N text-line bands on a published column image from its row darkness profile.

Used to place line rectangles on the published Grand Prize column images; every
result is checked visually (reports/reading/columns/*_boxes.png) before use.
"""
import numpy as np
from PIL import Image, ImageDraw


def gaussian_filter1d(a, sigma):
    r = int(3 * sigma) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    return np.convolve(np.pad(a, r, mode="edge"), k / k.sum(), mode="valid")


def find_peaks(a, distance):
    """Local maxima, strongest first, at least `distance` apart."""
    cand = [i for i in range(1, len(a) - 1) if a[i] >= a[i - 1] and a[i] > a[i + 1]]
    keep = []
    for i in sorted(cand, key=lambda i: -a[i]):
        if all(abs(i - j) >= distance for j in keep):
            keep.append(i)
    return np.array(sorted(keep)), {}


def line_bands(path, n, label_px=40, pad=0.12):
    """n line rectangles [x0, y0, x1, y1]: an evenly spaced grid of n lines fitted to
    the row darkness profile (scribal lines are regular). The bottom label_px rows
    hold the publisher's credit and are ignored."""
    g = np.asarray(Image.open(path).convert("L"), np.float32)
    H, W = g.shape
    paper = g > 20                                    # torn edges are pure black, not ink
    ink = (g < np.percentile(g[:H - label_px][paper[:H - label_px]], 18)) & paper
    dark = ink.mean(axis=1)
    dark[H - label_px:] = 0
    prof = gaussian_filter1d(dark, 4)
    best = None
    for pitch in np.arange(H / (n + 1.5), H / max(n - 0.6, 0.5), 0.5):
        for off in np.arange(0, H - pitch * (n - 1), 1.0):
            ys = (off + pitch * np.arange(n)).astype(int)
            if ys[-1] >= H - label_px:
                break
            sc = prof[ys].sum()
            if best is None or sc > best[0]:
                best = (sc, pitch, off)
    _, pitch, off = best
    half = pitch * (0.5 + pad)
    xs = np.nonzero(gaussian_filter1d(ink[:H - label_px].mean(axis=0), 5) > 0.02)[0]
    x0, x1 = int(max(xs.min() - 10, 0)), int(min(xs.max() + 10, W))
    return [[x0, int(max(off + i * pitch - half, 0)), x1, int(min(off + i * pitch + half, H - label_px))] for i in range(n)]


def draw(path, boxes, out):
    im = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(im)
    for i, b in enumerate(boxes):
        d.rectangle(b, outline=(0, 160, 120) if i % 2 else (200, 30, 90), width=3)
        d.text((b[0] + 4, b[1] + 2), str(i), fill=(255, 0, 0))
    im.save(out)
