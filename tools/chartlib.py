#!/usr/bin/env python3
"""Shared drawing primitives for the project charts.

Authoring only - numpy + Pillow, nothing added to the crate's dependencies.
make_charts.py predates this file and still carries its own copy; new charts
should import from here.
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# palette (GitHub dark)
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
YELLOW = (210, 168, 60)

W, H = 1600, 780
PAD_L, PAD_R, PAD_T, PAD_B = 110, 150, 96, 92


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = ["segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]
    root = os.environ.get("WINDIR", r"C:\Windows")
    for n in names:
        p = os.path.join(root, "Fonts", n)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                pass
    return ImageFont.load_default()


def mono(size: int) -> ImageFont.FreeTypeFont:
    root = os.environ.get("WINDIR", r"C:\Windows")
    for n in ["consola.ttf", "cour.ttf"]:
        p = os.path.join(root, "Fonts", n)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                pass
    return font(size)


def canvas(title: str, subtitle: str, note: str = "", w: int = W, h: int = H) -> Image.Image:
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)
    d.text((PAD_L - 40, 30), title, fill=TEXT, font=font(38, bold=True))
    d.text((PAD_L - 40, 74), subtitle, fill=MUTED, font=font(21))
    if note:
        f = font(18)
        tw = d.textlength(note, font=f)
        d.text((w - PAD_R + 10 - tw, 78), note, fill=MUTED, font=f)
    return img


def text(img, xy, s, color=MUTED, size=19, bold=False, anchor=None):
    d = ImageDraw.Draw(img)
    d.text(xy, s, fill=color, font=font(size, bold), anchor=anchor)


def legend(img: Image.Image, entries, x: int, y: int, size: int = 20):
    d = ImageDraw.Draw(img)
    f = font(size)
    for color, label in entries:
        d.rectangle([x, y + 8, x + 26, y + 16], fill=color)
        d.text((x + 36, y), label, fill=TEXT, font=f)
        x += 36 + d.textlength(label, font=f) + 40


def nice_ticks(lo: float, hi: float, target: int = 6):
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
    ticks, v = [], start
    while v <= hi + 1e-9:
        ticks.append(round(v, 6))
        v += step
    return ticks


class Axes:
    def __init__(self, img, box, xlim, ylim, fmt="{:,.0f}"):
        self.img = img
        self.d = ImageDraw.Draw(img)
        self.x0, self.y0, self.x1, self.y1 = box
        self.xlim = xlim
        self.ylim = ylim
        self.fmt = fmt

    def px(self, x):
        a, b = self.xlim
        return self.x0 + (x - a) / (b - a) * (self.x1 - self.x0)

    def py(self, y):
        a, b = self.ylim
        return self.y1 - (y - a) / (b - a) * (self.y1 - self.y0)

    def grid_y(self, ticks):
        for t in ticks:
            y = self.py(t)
            self.d.line([(self.x0, y), (self.x1, y)], fill=GRID, width=1)
            lbl = self.fmt.format(t)
            f = font(19)
            w = self.d.textlength(lbl, font=f)
            self.d.text((self.x0 - 14 - w, y - 10), lbl, fill=MUTED, font=f)

    def grid_x(self, ticks):
        for t in ticks:
            x = self.px(t)
            self.d.line([(x, self.y0), (x, self.y1)], fill=GRID, width=1)

    def x_labels(self, ticks, y: int, fmt="{:,.0f}"):
        f = font(19)
        for t in ticks:
            x = self.px(t)
            self.d.line([(x, self.y1), (x, self.y1 + 6)], fill=AXIS, width=1)
            lbl = fmt.format(t)
            self.d.text((x - self.d.textlength(lbl, font=f) / 2, y), lbl, fill=MUTED, font=f)

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


def save(img: Image.Image, out_dir: str, name: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    img.save(path, "PNG", optimize=True)
    print(f"  wrote {path} ({os.path.getsize(path) / 1024:.0f} KB)")
