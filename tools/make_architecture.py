#!/usr/bin/env python3
"""Draw the architecture diagram used by the README.

Authoring only, numpy + Pillow, nothing added to the crate.

    python tools/make_architecture.py [--out docs]

Writes docs/architecture.png. Reading order is left to right: tokens, embedding,
one transformer block (expanded vertically), then the tied head and the loss.
The two residual adds are drawn as a bus on the left of the stack, because that
is the only way to keep the arrows readable at this width.
"""

from __future__ import annotations

import argparse
import math
import os

from PIL import Image, ImageDraw

from chartlib import (
    AXIS,
    BG,
    BLUE,
    GREEN,
    MUTED,
    ORANGE,
    PANEL,
    PURPLE,
    RED,
    TEXT,
    YELLOW,
    font,
    mono,
    save,
)

W, H = 1800, 1160
INK = (10, 14, 20)


def box(d, rect, fill, title, sub=None, sub2=None, tsize=21):
    x0, y0, x1, y1 = rect
    d.rounded_rectangle([x0, y0, x1, y1], radius=10, fill=fill)
    cx = (x0 + x1) / 2
    tw = d.textlength(title, font=font(tsize, bold=True))
    d.text((cx - tw / 2, y0 + 11), title, fill=INK, font=font(tsize, bold=True))
    y = y0 + 11 + tsize + 7
    for s in (sub, sub2):
        if s:
            w = d.textlength(s, font=font(15))
            d.text((cx - w / 2, y), s, fill=INK, font=font(15))
            y += 19


def head(d, tip, ang, color, size=11, width=3):
    for s in (+1, -1):
        d.line(
            [tip, (tip[0] - size * math.cos(ang - s * 0.5), tip[1] - size * math.sin(ang - s * 0.5))],
            fill=color,
            width=width,
        )


def down(d, x, y0, y1, color=AXIS, width=3):
    d.line([(x, y0), (x, y1)], fill=color, width=width)
    head(d, (x, y1), math.pi / 2, color, width=width)


def right(d, y, x0, x1, color=AXIS, width=3):
    d.line([(x0, y), (x1, y)], fill=color, width=width)
    head(d, (x1, y), 0, color, width=width)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out", default=os.path.join(here, "docs"))
    a = ap.parse_args()

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    d.text((60, 34), "StonefishAI v2 - one forward pass", fill=TEXT, font=font(38, bold=True))
    d.text(
        (60, 80),
        "222,331,904 parameters - RMSNorm, SwiGLU, RoPE, weight tying, 16 layers, ctx 512",
        fill=MUTED,
        font=font(21),
    )
    note = "every box below is our own code"
    d.text((W - 70 - d.textlength(note, font=font(19)), 40), note, fill=GREEN, font=font(19))

    # ---------------------------------------------------------------- input
    box(d, (60, 180, 320, 260), (120, 130, 140), "token ids", "u16, BPE-16k", "batch 8 x 512")
    down(d, 190, 260, 310)
    box(d, (60, 310, 320, 410), BLUE, "Embedding", "16384 x 1024", "shared with the head")

    # ---------------------------------------------------------------- block panel
    px0, px1 = 400, 1120
    py0, py1 = 150, 1000
    d.rounded_rectangle([px0, py0, px1, py1], radius=16, outline=AXIS, width=2)
    d.text((px0 + 24, py0 + 16), "TransformerBlock", fill=TEXT, font=font(24, bold=True))
    lab = "x 16 layers"
    d.text((px1 - 24 - d.textlength(lab, font=font(22, bold=True)), py0 + 18), lab,
           fill=ORANGE, font=font(22, bold=True))

    cx = (px0 + px1) / 2
    st_w, st_x0, st_x1 = 520, cx - 260, cx + 260

    stages = [
        (230, 300, YELLOW, "RMSNorm", "per token, no bias, eps 1e-6"),
        (340, 450, PURPLE, "Causal attention", "8 heads x d_head 128 - RoPE applied inside"),
        (490, 548, (150, 160, 170), "+ residual", None),
        (590, 660, YELLOW, "RMSNorm", "same shape"),
        (700, 812, GREEN, "SwiGLU", "1024 -> 2816 -> 1024, silu(xW1) * (xW3)"),
        (854, 912, (150, 160, 170), "+ residual", None),
    ]
    for y0, y1, col, title, sub in stages:
        h = y1 - y0
        box(d, (st_x0, y0, st_x1, y1), col, title, sub, tsize=21 if h > 60 else 19)

    # vertical connectors between the stages
    for (y0a, y1a, _, _, _), (y0b, _, _, _, _) in zip(stages, stages[1:]):
        down(d, cx, y1a, y0b)

    # residual bus on the left: two taps into the two "+ residual" stages
    bus_x = st_x0 - 66
    d.line([(px0 + 40, 265), (bus_x, 265)], fill=GREEN, width=3)
    d.line([(bus_x, 265), (bus_x, 519)], fill=GREEN, width=3)
    right(d, 519, bus_x, st_x0, GREEN, 3)
    d.line([(px0 + 40, 519), (bus_x, 519)], fill=GREEN, width=3)
    d.line([(bus_x, 519), (bus_x, 883)], fill=GREEN, width=3)
    right(d, 883, bus_x, st_x0, GREEN, 3)
    d.text((px0 + 30, 200), "residual stream", fill=GREEN, font=font(16))

    # in / out of the block
    right(d, 350, 320, px0, AXIS, 3)
    d.line([(px0 + 30, 350), (st_x0 - 120, 350)], fill=AXIS, width=2)
    d.text((st_x1 - 150, 930), "block output", fill=MUTED, font=font(17))
    d.text((st_x1 - 150, 952), "(8, 512, 1024)", fill=MUTED, font=font(15))

    # ---------------------------------------------------------------- head
    hx0, hx1 = 1220, 1740
    down(d, cx, 912, 960)
    right(d, 960, cx, hx0)
    d.line([(hx0, 960), (hx0, 250)], fill=AXIS, width=1)

    box(d, (hx0, 230, hx1, 300), YELLOW, "RMSNorm", "final")
    down(d, (hx0 + hx1) / 2, 300, 340)
    box(d, (hx0, 340, hx1, 430), BLUE, "Head", "1024 x 16384, tied", "no separate matrix")
    down(d, (hx0 + hx1) / 2, 430, 470)
    box(d, (hx0, 470, hx1, 580), RED, "Cross-entropy", "16384-way softmax", "backward: softmax - onehot")
    d.text((hx0, 596), "loss (nat/token)", fill=MUTED, font=font(18))

    # ---------------------------------------------------------------- notes
    ny = 1046
    d.rounded_rectangle([60, ny - 4, W - 60, H - 14], radius=12, fill=PANEL)
    notes = [
        ("Weight tying", "the head is the embedding - 16.8M parameters saved"),
        ("Fused attention", "the B*S*S score matrix is never written to memory"),
        ("KV cache", "generation O(n); matches the full forward pass to 5e-8"),
        ("Zero dependencies", "no ML framework, no CUDA crate - our own FFI"),
    ]
    col_w = (W - 160) / 2
    for i, (t, s) in enumerate(notes):
        cxx = 84 + (i % 2) * col_w
        cyy = ny + 12 + (i // 2) * 42
        d.text((cxx, cyy), t, fill=GREEN, font=font(18, bold=True))
        d.text((cxx + 210, cyy + 1), s, fill=MUTED, font=font(17))

    save(img, a.out, "architecture.png")


if __name__ == "__main__":
    main()
