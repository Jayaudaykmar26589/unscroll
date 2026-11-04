"""Synthetic crushed scroll: a window into a large rolled sheet, slice by slice.

The sheet is an Archimedean spiral r = r0 + pitch * theta / 2pi. Its centre
line is crushed (squash + smooth warp that drifts slowly with z), then given
thickness along the deformed normal, so crushing closes the gaps between
windings without thinning the sheet: at high crush levels neighbouring layers
touch, as in real scrolls. Each slice image is a size x size window centred on
a click point at a chosen angle, so local layer curvature matches real scans
(radius of thousands of voxels), not a toy spiral that fits in the frame.

Ground truth per slice is the deformed centre line inside the window: x, y,
spiral angle t (continuous along the sheet), winding k and unrolled length u.
NumPy, Pillow and tifffile only (no SciPy or OpenCV).
"""
from dataclasses import asdict, dataclass

import numpy as np

# squash (y scale), warp amplitude (px) per crush level 0..4
CRUSH = {0: (1.00, 0.0), 1: (0.85, 3.0), 2: (0.70, 5.0), 3: (0.58, 7.0), 4: (0.50, 9.0)}

DEFAULT_TEXT = ["ΦΙΛΟΔΗΜΟΥ ΠΕΡΙ ΜΟΥΣΙΚΗΣ", "ΤΑ ΜΕΝ ΟΥΝ ΠΡΩΤΑ ΛΕΓΟΜΕΝΑ"]


@dataclass
class SynthConfig:
    size: int = 1024
    slices: int = 64
    r0: float = 40.0
    r_max: float = 1500.0
    pitch: float = 16.0
    thickness: float = 7.0
    twist: float = 0.004       # rad per slice
    crush: int = 2
    view_angle: float = 0.0    # 0: open side; pi/2: flattened side where layers touch
    view_radius: float = 0.6   # click radius as a fraction of r_max
    ink_contrast: float = 0.06
    noise: float = 0.06
    blur: float = 1.0
    seed: int = 7

    @property
    def squash(self):
        return CRUSH[self.crush][0]

    @property
    def warp(self):
        return CRUSH[self.crush][1]


def theta_max(cfg):
    return 2 * np.pi * (cfg.r_max - cfg.r0) / cfg.pitch


def arc_length(cfg, theta):
    """Unrolled length u(theta) = integral of r dtheta."""
    return cfg.r0 * theta + cfg.pitch * theta ** 2 / (4 * np.pi)


def render_text_tile(cfg, width=1400, lines=None, font_path="C:/Windows/Fonts/arial.ttf"):
    """Text mask (slices x width), lines along u, stacked along z; tiled along u."""
    from PIL import Image, ImageDraw, ImageFont
    lines = lines or DEFAULT_TEXT
    h = cfg.slices
    line_h = h // max(len(lines), 1)
    try:
        font = ImageFont.truetype(font_path, max(int(line_h * 0.7), 1))
    except OSError:
        font = ImageFont.load_default()
    img = Image.new("L", (width, h), 0)
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((20, i * line_h + int(line_h * 0.1)), line, fill=255, font=font)
    return np.asarray(img) > 127


class _Warp:
    """Smooth 2D displacement whose phases drift slowly with z."""

    def __init__(self, cfg, rng, terms=4):
        self.amp = cfg.warp
        self.k = rng.uniform(0.004, 0.012, size=(terms, 2)) * rng.choice([-1, 1], size=(terms, 2))
        self.phase = rng.uniform(0, 2 * np.pi, size=(terms, 2))
        self.drift = rng.uniform(-0.03, 0.03, size=(terms, 2))

    def __call__(self, x, y, z):
        if self.amp == 0:
            return x, y
        dx = np.zeros_like(x)
        dy = np.zeros_like(y)
        n = len(self.k)
        for (kx, ky), (px, py), (wx, wy) in zip(self.k, self.phase, self.drift):
            dx += np.sin(kx * x + ky * y + px + wx * z)
            dy += np.sin(ky * x - kx * y + py + wy * z)
        return x + self.amp * dx / np.sqrt(n), y + self.amp * dy / np.sqrt(n)


def _smooth_noise(rng, n, scale):
    """1D band-limited noise in [-1, 1] (sum of random sinusoids)."""
    t = np.arange(n, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    for _ in range(6):
        f = rng.uniform(0.5, 4.0) / scale
        out += np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    return out / 6


def _blur(img, sigma):
    if sigma <= 0:
        return img
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(img, r, mode="edge")
    rows = sum(w * pad[:, i:i + img.shape[1]] for i, w in enumerate(k))
    return sum(w * rows[i:i + img.shape[0], :] for i, w in enumerate(k))


class Scroll:
    """Renders slice windows and ground truth for one configuration."""

    def __init__(self, cfg):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        self.warp = _Warp(cfg, rng)
        dtheta = 0.4 / cfg.r_max                     # <= 0.4 px along the sheet
        self.theta = np.arange(0.0, theta_max(cfg), dtheta)
        self.r_mid = cfg.r0 + cfg.pitch * self.theta / (2 * np.pi)
        self.u = arc_length(cfg, self.theta)
        self.k = np.floor(self.theta / (2 * np.pi)).astype(np.int16)
        self.text = render_text_tile(cfg)
        self.fiber = 0.06 * _smooth_noise(rng, int(self.u[-1]) + 1, 40.0)
        self.offsets = np.arange(-cfg.thickness / 2, cfg.thickness / 2 + 1e-6, 0.5)
        self.rng = rng
        # click theta: the winding nearest view_radius at view_angle
        m = round((cfg.view_radius * cfg.r_max - cfg.r0) / cfg.pitch - cfg.view_angle / (2 * np.pi))
        self.click_theta = cfg.view_angle + 2 * np.pi * m
        cx, cy = self._place(np.array([cfg.r0 + cfg.pitch * self.click_theta / (2 * np.pi)]),
                             np.array([self.click_theta]), 0, window=False)
        self.origin = (float(cx[0]) - cfg.size / 2, float(cy[0]) - cfg.size / 2)

    def _place(self, radius, theta, z, window=True):
        cfg = self.cfg
        ang = theta + cfg.twist * z
        x = radius * np.cos(ang) * (1 + 0.5 * (1 - cfg.squash))   # crushing spreads sideways
        y = radius * np.sin(ang) * cfg.squash
        x, y = self.warp(x, y, z)
        if window:
            x, y = x - self.origin[0], y - self.origin[1]
        return x, y

    def _visible(self, z, margin=24):
        """Indices of centre-line samples that land inside the window (+margin)."""
        x, y = self._place(self.r_mid, self.theta, z)
        n = self.cfg.size
        sel = np.flatnonzero((x > -margin) & (x < n + margin) & (y > -margin) & (y < n + margin))
        return sel, x[sel], y[sel]

    def centerline(self, z):
        """Ground truth for slice z inside the window: x, y, t, k, u."""
        sel, x, y = self._visible(z, margin=0)
        return {"x": x.astype(np.float32), "y": y.astype(np.float32),
                "t": self.theta[sel].astype(np.float64), "k": self.k[sel],
                "u": self.u[sel].astype(np.float32)}

    def render(self, z):
        """Return (image float32 in ~[0,1], ink mask bool) for slice z."""
        cfg = self.cfg
        n = cfg.size
        sel, xm, ym = self._visible(z)
        acc = np.zeros(n * n, dtype=np.float32)
        cnt = np.zeros(n * n, dtype=np.float32)
        ink_img = np.zeros(n * n, dtype=bool)
        ui = np.clip(self.u[sel].astype(int), 0, len(self.fiber) - 1)
        inked = self.text[min(z, self.text.shape[0] - 1), ui % self.text.shape[1]]
        base = 0.6 + self.fiber[ui]
        # normal of the deformed centre line; consecutive samples along a sheet
        # stay adjacent in `sel` except at window exits, where gradients are local
        tx, ty = np.gradient(xm), np.gradient(ym)
        norm = np.maximum(np.hypot(tx, ty), 1e-9)
        nx, ny = ty / norm, -tx / norm
        gx, gy = self._place(np.zeros(1), np.zeros(1), z)            # scroll centre
        flip = (nx * (xm - gx[0]) + ny * (ym - gy[0])) < 0           # outward = recto
        nx[flip], ny[flip] = -nx[flip], -ny[flip]
        for off in self.offsets:
            x, y = xm + off * nx, ym + off * ny
            xi = np.rint(x).astype(int)
            yi = np.rint(y).astype(int)
            ok = (xi >= 0) & (xi < n) & (yi >= 0) & (yi < n)
            idx = yi[ok] * n + xi[ok]
            # denser core, lighter faces: touching sheets keep a faint seam
            val = base[ok] * (0.75 + 0.25 * np.cos(np.pi * off / cfg.thickness))
            if off >= cfg.thickness / 2 - 2.0:           # recto face carries ink
                hit = inked[ok]
                val = val + cfg.ink_contrast * hit
                ink_img[idx[hit]] = True
            np.add.at(acc, idx, val)
            np.add.at(cnt, idx, 1.0)
        img = np.full(n * n, 0.12, dtype=np.float32)    # air / background
        filled = cnt > 0
        img[filled] = acc[filled] / cnt[filled]
        img = _blur(img.reshape(n, n), cfg.blur)
        img = img + self.rng.normal(0, cfg.noise, img.shape).astype(np.float32)
        return img.astype(np.float32), ink_img.reshape(n, n)

    def click(self):
        """Click at slice 0: the window centre, on the target sheet's centre line."""
        line = self.centerline(0)
        i = int(np.argmin(np.abs(line["t"] - self.click_theta)))
        return int(round(float(line["x"][i]))), int(round(float(line["y"][i]))), float(line["t"][i])

    def meta(self):
        return {"config": asdict(self.cfg), "windings": int(self.k.max()) + 1,
                "unrolled_length": float(self.u[-1]), "window_origin": list(self.origin)}
