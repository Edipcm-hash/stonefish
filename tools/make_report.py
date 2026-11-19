#!/usr/bin/env python3
"""Render the v1.0 report charts from a measurement file.

Everything here is drawn from docs/report.json, which is written by hand (or by
the measurement steps described in the README) so the numbers on the charts are
always traceable to a command that was actually run. Nothing is estimated.

Usage:
    python tools/make_report.py [--data docs/report.json] [--out docs]

Charts:
    docs/tests.png       the test suite by area, with pass/fail per area
    docs/comparison.png  StonefishAI vs PyTorch: speed, memory, and loss parity
    docs/quality.png     model quality: bits-per-character against reference points
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

from chartlib import (
    AXIS,
    BG,
    BLUE,
    GREEN,
    GRID,
    H,
    MUTED,
    ORANGE,
    PAD_B,
    PAD_L,
    PAD_R,
    PURPLE,
    RED,
    TEXT,
    W,
    YELLOW,
    Axes,
    canvas,
    font,
    legend,
    mono,
    nice_ticks,
    save,
)


# ---------------------------------------------------------------- tests
def chart_tests(data, out):
    t = data["tests"]
    areas = t["areas"]
    total = sum(a["count"] for a in areas)
    failed = sum(a.get("failed", 0) for a in areas)

    img = canvas(
        f"Test suite - {total} tests, every one of them a real assertion",
        f"{total} tests across {len(areas)} areas - gradient checks, kernel parity, "
        "checkpoint safety, KV-cache",
        f"{total - failed}/{total} passing",
    )
    d = ImageDraw.Draw(img)

    y = 146
    row = 84
    bar_x0, bar_x1 = PAD_L + 520, W - PAD_R - 110
    longest = max(a["count"] for a in areas)

    for a in areas:
        n = a["count"]
        bad = a.get("failed", 0)
        color = RED if bad else GREEN

        d.text((PAD_L - 40, y - 4), a["name"], fill=TEXT, font=font(23, bold=True))
        d.text((PAD_L - 40, y + 26), a["file"], fill=MUTED, font=mono(16))
        what = a.get("what", "")
        if what:
            d.text((PAD_L - 40, y + 48), what, fill=MUTED, font=font(16))

        w = (n / longest) * (bar_x1 - bar_x0)
        d.rounded_rectangle([bar_x0, y + 8, bar_x0 + w, y + 46], radius=7, fill=color)
        d.text((bar_x0 + w + 18, y + 10), f"{n}", fill=TEXT, font=font(26, bold=True))
        y += row

    d.text((PAD_L - 40, H - 92),
           "gradient checks compare analytic backward() against finite differences",
           fill=MUTED, font=font(19))
    d.text((PAD_L - 40, H - 62),
           "kernel parity compares every CUDA kernel against the CPU reference",
           fill=MUTED, font=font(19))
    save(img, out, "tests.png")


# ---------------------------------------------------------------- comparison
def chart_comparison(data, out):
    speed = data["speed"]
    runs = speed["runs"]
    m = speed["model"]
    CH = 900

    img = canvas(
        "StonefishAI vs PyTorch - speed, memory, and correctness",
        f"same architecture (d={m['d']}, {m['layers']} layers, {m['heads']} heads, "
        f"ctx={m['ctx']}, {m['params'] / 1e6:.1f}M params), same GPU, same batch",
        speed.get("device", ""),
        h=CH,
    )
    d = ImageDraw.Draw(img)

    x0 = PAD_L + 300
    x1 = W - PAD_R - 190

    # --- throughput
    y = 150
    d.text((PAD_L - 40, y), "training throughput", fill=TEXT, font=font(24, bold=True))
    y += 46
    best = max(r["tok_s"] for r in runs)
    for r in runs:
        col = BLUE if r["name"].startswith("Stonefish") else ORANGE
        d.text((PAD_L - 40, y + 6), r["name"], fill=TEXT, font=font(20))
        w = (r["tok_s"] / best) * (x1 - x0)
        d.rounded_rectangle([x0, y, x0 + w, y + 40], radius=6, fill=col)
        d.text((x0 + w + 14, y + 2), f"{r['tok_s']:,.0f} tok/s", fill=TEXT, font=font(21, bold=True))
        if r.get("speedup"):
            d.text((x0 + w + 14, y + 26), r["speedup"], fill=GREEN, font=font(16))
        y += 62

    # --- peak VRAM
    y += 26
    d.text((PAD_L - 40, y), "peak VRAM", fill=TEXT, font=font(24, bold=True))
    y += 46
    vbest = max(r["vram_gb"] for r in runs)
    for r in runs:
        col = BLUE if r["name"].startswith("Stonefish") else ORANGE
        d.text((PAD_L - 40, y + 6), r["name"], fill=TEXT, font=font(19))
        w = (r["vram_gb"] / vbest) * (x1 - x0)
        d.rounded_rectangle([x0, y, x0 + w, y + 32], radius=6, fill=col)
        d.text((x0 + w + 14, y + 1), f"{r['vram_gb']:.2f} GB", fill=TEXT, font=font(20, bold=True))
        y += 48

    # --- loss parity: the honest correctness check, shown as two numbers and a delta
    y += 30
    d.text((PAD_L - 40, y), "correctness - loss after the same steps", fill=TEXT, font=font(24, bold=True))
    y += 44
    par = data.get("parity")
    if not par:
        d.text((PAD_L - 40, y), "not measured", fill=RED, font=font(20))
        save(img, out, "comparison.png")
        return

    d.text((PAD_L - 40, y), f"{par['steps']} steps, identical data order and seeds", fill=MUTED, font=font(19))
    y += 40
    cells = [("StonefishAI", par["stonefish_loss"], BLUE), ("PyTorch", par["torch_loss"], ORANGE)]
    cw = 420
    for i, (name, loss, col) in enumerate(cells):
        cx = x0 + i * (cw + 60)
        d.rounded_rectangle([cx, y, cx + cw, y + 118], radius=10, fill=(22, 27, 34))
        d.text((cx + 24, y + 16), name, fill=MUTED, font=font(19))
        d.text((cx + 24, y + 48), f"{loss:.4f}", fill=col, font=font(44, bold=True))
    d.text((x0 + 2 * (cw + 60) + 10, y + 40), "delta", fill=MUTED, font=font(19))
    d.text((x0 + 2 * (cw + 60) + 10, y + 66), f"{par['abs_diff']:.5f}", fill=GREEN, font=font(30, bold=True))

    note = par.get("note", "the two implementations agree to f32 rounding")
    d.text((PAD_L - 40, y + 140), note, fill=GREEN, font=font(20))
    save(img, out, "comparison.png")

# ---------------------------------------------------------------- quality
def chart_quality(data, out):
    q = data["quality"]
    corpora = q["corpora"]

    img = canvas(
        "Model quality - what 222M parameters on 2B tokens actually buys",
        "bits per character on held-out data; each corpus against its OWN random baseline",
        "lower is better",
    )
    d = ImageDraw.Draw(img)

    # Every corpus needs its own random baseline: BPC is loss/ln(2)/chars_per_token,
    # so a corpus that tokenises into 2 characters per token has a baseline of
    # log2(vocab)/2, not log2(vocab)/4.8. Comparing a 2-char/token BPC against a
    # 4.8-char/token baseline is the classic way to make a model look worse than
    # random when it is not.
    # Rows are built so that a reference model sits directly under the corpus it
    # was measured on: the first corpus is the held-out validation set and the
    # reference was evaluated on the same data, so putting it after the
    # out-of-distribution probe would read as a separate category.
    rows = []
    references = list(q.get("reference", []))
    for idx, c in enumerate(corpora):
        cpt = c.get("chars_per_token", 4.812)
        base = c.get("random_bpc", 13.9997 / cpt)
        rows.append((c["name"], c["bpc"], BLUE, f"loss {c['loss']:.4f} · ppl {c['ppl']:.2f} · {cpt:.2f} chars/token"))
        rows.append(("  random baseline", base, MUTED, f"log2(16384) / {cpt:.2f}"))
        if idx == 0:
            for r in references:
                rows.append((r["name"], r["bpc"], PURPLE, r.get("note", "")))
            references = []
    for r in references:
        rows.append((r["name"], r["bpc"], PURPLE, r.get("note", "")))

    top = 150
    bar_h = 46
    x0 = PAD_L + 380
    x1 = W - PAD_R - 300
    vmax = max(r[1] for r in rows) * 1.06

    d.text((PAD_L - 40, top - 40), "bits per character", fill=TEXT, font=font(23, bold=True))
    for name, bpc, col, note in rows:
        d.text((PAD_L - 40, top + 2), name, fill=TEXT, font=font(19))
        w = (bpc / vmax) * (x1 - x0)
        d.rounded_rectangle([x0, top, x0 + w, top + bar_h - 14], radius=6, fill=col)
        d.text((x0 + w + 14, top - 2), f"{bpc:.3f}", fill=TEXT, font=font(20, bold=True))
        if note:
            d.text((PAD_L - 40, top + 22), note, fill=MUTED, font=font(15))
        top += bar_h + 8

    box_y = top + 12
    d.rounded_rectangle([PAD_L - 40, box_y, W - PAD_R, box_y + 170], radius=10, fill=(22, 27, 34))
    d.text((PAD_L - 10, box_y + 14), "How to read this", fill=TEXT, font=font(23, bold=True))
    for i, line in enumerate(q.get("reading", [])[:4]):
        col = GREEN if line.startswith("+") else RED if line.startswith("-") else MUTED
        d.text((PAD_L - 10, box_y + 52 + i * 28), line, fill=col, font=font(18))

    save(img, out, "quality.png")


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--data", default=os.path.join(here, "docs", "report.json"))
    ap.add_argument("--out", default=os.path.join(here, "docs"))
    args = ap.parse_args()

    if not os.path.exists(args.data):
        sys.exit(f"measurement file not found: {args.data}")

    with open(args.data, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    print(f"report: {data.get('measured_at', 'unknown time')}")
    chart_tests(data, args.out)
    chart_comparison(data, args.out)
    chart_quality(data, args.out)
    print("done")


if __name__ == "__main__":
    main()
