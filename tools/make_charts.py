#!/usr/bin/env python3
"""Render the project charts (PNG) from the training log.

Authoring tool only - it is not part of the crate and no Rust build touches it.
It needs numpy and Pillow, which the development machine already has; nothing
is added to the project's dependency list, which stays empty.

Usage:
    python tools/make_charts.py [--log en-train-v2b.log] [--out docs]

Charts written (GitHub renders them in the README):
    docs/loss-curve.png     training loss and the learning-rate schedule
    docs/throughput.png     tokens/s over the run, with the events annotated
    docs/speedup.png        StonefishAI against PyTorch on the same GPU
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- palette (GitHub dark)
BG = (13, 17, 23)
PANEL = (22, 27, 34)
GRID = (40, 46, 54)
AXIS = (88, 96, 105)
TEXT = (230, 237, 243)
MUTED = (139, 148, 158)
BLUE = (88, 166, 255)
ORANGE = (255, 166, 87)
GREEN = (63, 185, 80)
PURPLE = (188, 140, 255)
RED = (248, 81, 73)

W, H = 1600, 780
PAD_L, PAD_R, PAD_T, PAD_B = 110, 150, 96, 92


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Segoe UI if present, otherwise whatever Pillow ships with."""
    names = ["segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]
    for n in names:
        p = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", n)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                pass
    return ImageFont.load_default()


def _font_mono(size: int) -> ImageFont.FreeTypeFont:
    for n in ["consola.ttf", "cour.ttf"]:
        p = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", n)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                pass
    return _font(size)


# ---------------------------------------------------------------- data
LINE = re.compile(
    r"^step\s+(\d+)/(\d+)\s+loss\s+([\d.]+)\s+lr\s+([\d.eE+-]+).*?(\d+)\s+tok/s"
)


def parse_logs(paths):
    """Merge several logs into one series (the run was restarted once).

    A step may appear in more than one file; the first sighting wins, which
    keeps the numbers from the log that produced them.
    """
    by_step = {}
    total = 0
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = LINE.match(line.strip())
                if not m:
                    continue
                step = int(m.group(1))
                total = int(m.group(2))
                if step not in by_step:
                    by_step[step] = (float(m.group(3)), float(m.group(4)), int(m.group(5)))
    if not by_step:
        sys.exit(f"no training lines found in {paths}")
    steps = np.array(sorted(by_step), dtype=np.int64)
    losses = np.array([by_step[s][0] for s in steps], dtype=np.float64)
    lrs = np.array([by_step[s][1] for s in steps], dtype=np.float64)
    toks = np.array([by_step[s][2] for s in steps], dtype=np.float64)
    return steps, losses, lrs, toks, total


def smooth(y: np.ndarray, window: int = 25) -> np.ndarray:
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    padded = np.pad(y, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def nice_ticks(lo: float, hi: float, target: int = 6):
    """Round tick positions (1, 2, 2.5, 5 x 10^k) covering [lo, hi]."""
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / target
    mag = 10 ** np.floor(np.log10(raw))
    for mult in (1, 2, 2.5, 5, 10):
        step = mult * mag
        if span / step <= target + 1:
            break
    start = np.ceil(lo / step) * step
    ticks = []
    v = start
    while v <= hi + 1e-9:
        ticks.append(round(v, 6))
        v += step
    return ticks


# ---------------------------------------------------------------- drawing helpers
class Axes:
    """Minimal linear-scale plot area with grid, ticks and labels."""

    def __init__(self, img: Image.Image, box, xlim, ylim, fmt="{:,.0f}"):
        self.img = img
        self.d = ImageDraw.Draw(img)
        self.x0, self.y0, self.x1, self.y1 = box
        self.xlim = xlim
        self.ylim = ylim
        self.fmt = fmt

    def px(self, x):
        (xa, xb) = self.xlim
        return self.x0 + (x - xa) / (xb - xa) * (self.x1 - self.x0)

    def py(self, y):
        (ya, yb) = self.ylim
        return self.y1 - (y - ya) / (yb - ya) * (self.y1 - self.y0)

    def grid_x(self, ticks):
        for t in ticks:
            x = self.px(t)
            self.d.line([(x, self.y0), (x, self.y1)], fill=GRID, width=1)
            lbl = self.fmt.format(t)
            f = _font(19)
            w = self.d.textlength(lbl, font=f)
            self.d.text((x - w / 2, self.y1 + 12), lbl, fill=MUTED, font=f)

    def grid_y(self, ticks):
        for t in ticks:
            y = self.py(t)
            self.d.line([(self.x0, y), (self.x1, y)], fill=GRID, width=1)
            lbl = self.fmt.format(t)
            f = _font(19)
            w = self.d.textlength(lbl, font=f)
            self.d.text((self.x0 - 14 - w, y - 10), lbl, fill=MUTED, font=f)

    def polyline(self, xs, ys, fill, width=3):
        pts = [(self.px(x), self.py(y)) for x, y in zip(xs, ys)]
        if len(pts) > 1:
            self.d.line(pts, fill=fill, width=width, joint="curve")

    def dots(self, xs, ys, fill, r=2):
        for x, y in zip(xs, ys):
            px, py = self.px(x), self.py(y)
            self.d.ellipse([px - r, py - r, px + r, py + r], fill=fill)

    def frame(self):
        self.d.rectangle([self.x0, self.y0, self.x1, self.y1], outline=AXIS, width=2)

    def band(self, xs, lo, hi, fill):
        """Filled vertical band between two series (a rolling min/max envelope)."""
        upper = [(self.px(x), self.py(y)) for x, y in zip(xs, hi)]
        lower = [(self.px(x), self.py(y)) for x, y in zip(xs, lo)][::-1]
        self.d.polygon(upper + lower, fill=fill)


def canvas(title: str, subtitle: str, watermark: str = "") -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((PAD_L - 40, 30), title, fill=TEXT, font=_font(38, bold=True))
    d.text((PAD_L - 40, 78 - 6), subtitle, fill=MUTED, font=_font(21))
    if watermark:
        f = _font(18)
        w = d.textlength(watermark, font=f)
        d.text((W - PAD_R + 10 - w, 34), watermark, fill=MUTED, font=f)
    return img


def legend(img: Image.Image, entries, x: int, y: int):
    d = ImageDraw.Draw(img)
    f = _font(20)
    for color, label in entries:
        d.rectangle([x, y + 8, x + 26, y + 16], fill=color)
        d.text((x + 36, y), label, fill=TEXT, font=f)
        x += 36 + d.textlength(label, font=f) + 40


def save(img: Image.Image, out_dir: str, name: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    img.save(path, "PNG", optimize=True)
    print(f"  wrote {path} ({os.path.getsize(path) / 1024:.0f} KB)")


# ---------------------------------------------------------------- chart 1: loss
def chart_loss(steps, losses, lrs, total, out_dir):
    img = canvas(
        "Training loss - StonefishAI v2",
        f"222M parameters, {total:,} steps, 2.05B tokens, one RTX 5070",
        "loss = cross-entropy, nat/token",
    )
    d = ImageDraw.Draw(img)

    # two stacked panels sharing the x axis
    xs = (0, total)
    top = (PAD_L, 150, W - PAD_R, 470)
    bot = (PAD_L, 540, W - PAD_R, 664)

    ax = Axes(img, top, xs, (2.8, 10.2))
    ax.grid_y(nice_ticks(3.0, 10.0, 5))
    ax.frame()
    d.text((PAD_L - 78, top[1] - 26), "loss", fill=MUTED, font=_font(18))

    base = np.log(16384)
    y = ax.py(base)
    for x in range(ax.x0, ax.x1, 18):
        d.line([(x, y), (x + 9, y)], fill=(120, 130, 140), width=2)
    d.text((ax.x1 - 470, y + 12), f"random baseline  ln(16384) = {base:.2f}", fill=MUTED, font=_font(19))

    step = max(1, len(steps) // 1400)
    ax.dots(steps[::step], losses[::step], (32, 66, 104), r=2)
    ax.polyline(steps, smooth(losses), BLUE, width=4)
    ax.dots(steps[::step], losses[::step], (32, 66, 104), r=2)
    ax.polyline(steps, smooth(losses), BLUE, width=4)

    # where the run was restarted
    resume = 5500
    rx = ax.px(resume)
    for yy in range(ax.y0, ax.y1, 14):
        d.line([(rx, yy), (rx, yy + 7)], fill=(90, 100, 110), width=1)
    d.text((rx + 10, ax.y1 - 30), "run resumed here", fill=MUTED, font=_font(17))

    axr = Axes(img, bot, xs, (0, max(lrs) * 1.08), fmt="{:g}e-4")
    axr.ylim = (0, max(lrs) * 1.08)
    for t in nice_ticks(0, max(lrs), 3):
        y = axr.py(t)
        axr.d.line([(axr.x0, y), (axr.x1, y)], fill=GRID, width=1)
        lbl = f"{t * 1e4:.1f}e-4" if t > 0 else "0"
        f2 = _font(19)
        w = axr.d.textlength(lbl, font=f2)
        axr.d.text((axr.x0 - 14 - w, y - 10), lbl, fill=MUTED, font=f2)
    axr.frame()
    d.text((PAD_L - 78, bot[1] - 26), "learning rate", fill=ORANGE, font=_font(18))
    axr.polyline(steps, lrs, ORANGE, width=3)

    # x labels under the lower panel only
    f = _font(19)
    for t in nice_ticks(0, total, 8):
        x = axr.px(t)
        d.line([(x, axr.y1), (x, axr.y1 + 6)], fill=AXIS, width=1)
        lbl = f"{t:,.0f}"
        d.text((x - d.textlength(lbl, font=f) / 2, axr.y1 + 12), lbl, fill=MUTED, font=f)

    legend(img, [(BLUE, "loss (smoothed)"), ((32, 66, 104), "loss (raw)"), (ORANGE, "learning rate")],
           PAD_L, H - 46)
    save(img, out_dir, "loss-curve.png")


# ---------------------------------------------------------------- chart 2: throughput
def chart_throughput(steps, toks, out_dir):
    img = canvas(
        "Training throughput - one RTX 5070 at 250 W",
        "tokens per second, sampled every 20 steps",
        "higher is better",
    )
    d = ImageDraw.Draw(img)
    lo = float(np.percentile(toks, 0.3))
    hi = float(np.percentile(toks, 99.7))
    pad = (hi - lo) * 0.10
    ax = Axes(img, (PAD_L, 150, W - PAD_R, H - PAD_B), (0, steps[-1]), (lo - pad, hi + pad))

    ax.grid_y(nice_ticks(lo - pad, hi + pad, 6))
    ax.grid_x(nice_ticks(0, steps[-1], 8))
    ax.frame()

    # rolling 5th/95th percentile envelope instead of a raw sawtooth line
    win = max(20, len(toks) // 120)
    env_lo = np.empty_like(toks)
    env_hi = np.empty_like(toks)
    for i in range(len(toks)):
        a = max(0, i - win // 2)
        b = min(len(toks), i + win // 2 + 1)
        chunk = toks[a:b]
        env_lo[i] = np.percentile(chunk, 5)
        env_hi[i] = np.percentile(chunk, 95)
    # keep the envelope inside the frame (deep dips would paint over the border)
    env_lo = np.maximum(env_lo, ax.ylim[0])
    env_hi = np.minimum(env_hi, ax.ylim[1])
    ax.band(steps, env_lo, env_hi, (26, 50, 78))
    ax.polyline(steps, smooth(toks, 60), BLUE, width=4)

    mean = float(np.mean(toks[-400:]))
    y = ax.py(mean)
    for x in range(ax.x0, ax.x1, 16):
        d.line([(x, y), (x + 8, y)], fill=GREEN, width=2)
    d.text((ax.x0 + 16, y - 34), f"final average  {mean:,.0f} tok/s", fill=GREEN, font=_font(21))

    f = _font(19)
    d.text((ax.x1 - 40, ax.y1 + 14), "step", fill=MUTED, font=f)

    legend(img, [(BLUE, "tokens/s"), ((26, 50, 78), "5th-95th percentile"), (GREEN, "final average")],
           PAD_L, H - 46)
    save(img, out_dir, "throughput.png")


# ---------------------------------------------------------------- chart 3: speed comparison
def chart_speedup(out_dir):
    # measured with tools/torch_bench.py: same architecture, same GPU, same data
    rows = [
        ("StonefishAI", 20487, BLUE, "from-scratch Rust + own CUDA FFI"),
        ("PyTorch  (SDPA)", 10810, ORANGE, "scaled_dot_product_attention"),
        ("PyTorch  (naive)", 11456, RED, "explicit attention math"),
    ]
    best = max(r[1] for r in rows)

    img = canvas(
        "StonefishAI vs PyTorch - same GPU, same model",
        "tokens/second, measured on one RTX 5070",
        "higher is better",
    )
    d = ImageDraw.Draw(img)
    x0, x1 = PAD_L, W - PAD_R - 60
    y = PAD_T + 70
    bh, gap = 108, 54
    for name, val, color, note in rows:
        d.text((PAD_L - 40, y - 6), name, fill=TEXT, font=_font(26, bold=True))
        d.text((PAD_L - 40, y + 30), note, fill=MUTED, font=_font(18))
        w = (val / best) * (x1 - x0 - 210)
        d.rounded_rectangle([x0, y + 56, x0 + w, y + 56 + bh - 46], radius=8, fill=color)
        d.text((x0 + w + 18, y + 52), f"{val:,} tok/s", fill=TEXT, font=_font(26, bold=True))
        if val == best:
            d.text((x0 + w + 18, y + 84), "baseline", fill=GREEN, font=_font(17))
        y += bh + gap

    d.text((PAD_L - 40, H - PAD_B + 6), "1.79x faster than PyTorch SDPA - measured, not estimated",
           fill=MUTED, font=_font(20))
    save(img, out_dir, "speedup.png")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--log", nargs="*", default=None,
                    help="one or more training logs (default: en-train-v2*.log)")
    ap.add_argument("--out", default=os.path.join(here, "docs"))
    args = ap.parse_args()

    paths = args.log
    if not paths:
        paths = sorted(
            glob.glob(os.path.join(here, "en-train-v2*.log")), key=os.path.getmtime
        )
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        sys.exit(f"no training logs found in {here} (expected en-train-v2*.log)")

    steps, losses, lrs, toks, total = parse_logs(paths)
    print("logs: " + ", ".join(os.path.basename(p) for p in paths))
    print(f"parsed {len(steps)} samples, step {steps[0]} -> {steps[-1]} of {total:,}")
    chart_loss(steps, losses, lrs, total, args.out)
    chart_throughput(steps, toks, args.out)
    chart_speedup(args.out)
    print("done")


if __name__ == "__main__":
    main()
