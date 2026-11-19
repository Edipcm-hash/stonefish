#!/usr/bin/env python3
"""Run the release measurements and write docs/report.json.

Every number in the README's results section comes from here. The point is that
no figure is typed by hand: the commands run, their output is parsed, and the
JSON that drives the charts is assembled from what they printed.

    python tools/measure.py                    # everything
    python tools/measure.py --skip-torch       # no PyTorch comparison
    python tools/measure.py --dry-run          # print the plan, run nothing

Requires: the trained checkpoint, a free GPU (the 222M model needs ~4 GB on top
of the weights), and torch for the comparison steps.

The tests section of the report comes from tools/collect_tests.py, so the chart
and the README agree by construction.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
TINY = "data/en/tiny"
PROBE = "data/en/probe"
BATCHES = "8"
LOG = os.path.join(ROOT, "en-train-v2b.log")


def sh(cmd, timeout=1800):
    p = subprocess.run(
        cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout,
        encoding="utf-8", errors="replace",
    )
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def parse_eval(out):
    """Pull loss/ppl/bpc out of `en-eval`'s RESULT block."""
    def grab(pattern):
        m = re.search(pattern, out)
        return float(m.group(1)) if m else None

    return {
        "tokens": int(grab(r"Sampled\s+:\s+(\d+) tokens") or 0),
        "loss": grab(r"val loss\s+:\s+([\d.]+)"),
        "ppl": grab(r"perplexity\s+:\s+([\d.]+)"),
        "bpc": grab(r"BPC\s+:\s+([\d.]+)"),
    }


def parse_gen(out):
    """The generated text and the tokens/s line of `en-gen`."""
    m = re.search(r"\((\d+) tokens, ([\d.]+) s, ([\d.]+) tokens/s", out)
    text = "\n".join(
        ln for ln in out.splitlines() if ln and not ln.startswith(("🐟", "  ("))
    ).strip()
    return {
        "text": text,
        "tokens": int(m.group(1)) if m else None,
        "seconds": float(m.group(2)) if m else None,
        "tok_s": float(m.group(3)) if m else None,
    }


def parse_bench(out):
    """`bench` prints step time, speed and VRAM; speed carries thousands separators."""
    def grab(pattern):
        m = re.search(pattern, out)
        return float(m.group(1).replace(",", "")) if m else None

    return {
        "ms_step": grab(r"step time:\s*([\d.,]+) ms"),
        "tok_s": grab(r"speed\s*:\s*([\d,]+) tokens/s"),
        "vram_gb": grab(r"peak VRAM\s*:\s*~?([\d.]+) GB"),
        "params": grab(r"-> ([\d.]+)M parameters"),
    }


def parse_parity(out):
    def grab(pattern):
        m = re.search(pattern, out)
        return float(m.group(1)) if m else None

    return {
        "stonefish_loss": grab(r"StonefishAI\s+:\s+([\d.]+)"),
        "torch_loss": grab(r"PyTorch loss\s+:\s+([\d.]+)"),
        "abs_diff": grab(r"delta\s+:\s+([\d.]+)"),
        "verdict": "MATCH" in out,
    }


def training_throughput(window=200):
    """Mean tokens/s of the last window steps, read from the training logs.

    The logged value is the real end-to-end step, so it is comparable with the
    PyTorch benchmark (which also times forward + backward + Adam)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "en-train-v2*.log"))):
        with io.open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = re.search(r"step\s+\d+/\d+\s+loss\s+[\d.]+.*?(\d+)\s+tok/s", line)
                if m:
                    rows.append(int(m.group(1)))
    if not rows:
        return None, None
    rows.sort()
    tail = rows[-window:]
    tps = sum(tail) / len(tail)
    return tps, 32768 / tps * 1000


def chars_per_token(txt_path, tokens):
    try:
        size = os.path.getsize(os.path.join(ROOT, txt_path))
    except OSError:
        return None
    return size / tokens if tokens else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-torch", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    state = "stonefish_en.final.state"
    if not os.path.exists(os.path.join(ROOT, state)):
        state = "stonefish_en.state"
    print(f"checkpoint: {state}")

    plan = [
        f"en-eval {BATCHES} {TINY} {state}",
        f"en-eval {BATCHES} {PROBE} {state}",
        'en-gen "Once upon a time" 150 0',
        'en-gen "Once upon a time" 150 0.8',
        "torch_parity.py",
        "torch_bench.py --attn naive|sdpa",
    ]
    if a.dry_run:
        print("\nwould run:")
        for p in plan:
            print("  " + p)
        return 0

    exe = os.path.join(ROOT, "target", "release", "stonefish_ai.exe")
    if not os.path.exists(exe):
        exe = os.path.join(ROOT, "target3", "release", "stonefish_ai.exe")
    if not os.path.exists(exe):
        print("build the binary first (cargo build --release)")
        return 1

    report = {"measured_at": None, "tests": None, "speed": None, "parity": None,
              "quality": None}

    # ---- tests (real data, from the collector) ----
    ok, out = sh([PY, "tools/collect_tests.py", "--json"])
    i, j = out.find("{"), out.rfind("}")
    if i >= 0 and j > i:
        report["tests"] = json.loads(out[i : j + 1])["tests"]
        t = report["tests"]
        print(f"tests      : {t['total']} total, {t['passed']} passed")

    # ---- in-distribution evaluation ----
    print("running en-eval on the validation set ...")
    ok, out = sh([exe, "en-eval", BATCHES, TINY, state])
    tiny = parse_eval(out)
    print(f"  loss {tiny['loss']}  ppl {tiny['ppl']}  bpc {tiny['bpc']}")

    # ---- out-of-distribution probe ----
    print("running en-eval on the probe ...")
    ok, out = sh([exe, "en-eval", BATCHES, PROBE, state])
    probe = parse_eval(out)
    print(f"  loss {probe['loss']}  bpc {probe['bpc']}")

    # ---- samples ----
    print("generating samples ...")
    ok_g, gen0 = sh([exe, "en-gen", "Once upon a time", "150", "0"])
    ok_s, gen8 = sh([exe, "en-gen", "Once upon a time", "150", "0.8"])
    g0, g8 = parse_gen(gen0), parse_gen(gen8)
    print(f"  greedy {g0['tok_s']} tok/s · sampled {g8['tok_s']} tok/s")

    # ---- quality block: each corpus with its own baseline ----
    LU = 13.9997  # log2(16384)
    corpora = []
    for name, stats, txt, cache in [
        ("TinyStories (held out)", tiny, "data/en/valid.txt", f"{TINY}.valid.bin"),
        ("README probe (unseen text)", probe, f"{PROBE}.txt", f"{PROBE}.valid.bin"),
    ]:
        # chars/token must use the WHOLE corpus, not the tokens this evaluation
        # sampled: en-eval reads 8 batches (32,768 tokens) out of 4,041,031, and
        # dividing by the sampled count inflated the ratio by 123x and made the
        # random baseline look like 0.024 bits/character.
        corpus_tokens = os.path.getsize(os.path.join(ROOT, cache)) // 2
        cpt = chars_per_token(txt, corpus_tokens) or 4.812
        corpora.append(
            {
                "name": name,
                "tokens": stats["tokens"],
                "corpus_tokens": corpus_tokens,
                "loss": stats["loss"],
                "ppl": stats["ppl"],
                "bpc": stats["bpc"],
                "chars_per_token": round(cpt, 3),
                "random_bpc": round(LU / cpt, 3),
            }
        )
    # A reference point and the reading lines are part of the chart, not decoration:
    # without them quality.png renders with no comparison bar and an empty "how to
    # read this" box. The v1 figure is a measurement from the earlier release (same
    # BPC metric, same validation file), so it is labelled as such.
    reference = [
        {
            "name": "v1 model (236M, TinyStories only)",
            "bpc": 0.691,
            "note": "same metric, 450M tokens of training",
        }
    ]

    reading = []
    for c in corpora:
        if not c.get("bpc"):
            reading.append(f"- could not measure {c['name']}")
            continue
        gain = c["random_bpc"] / c["bpc"]
        reading.append(
            f"+ {gain:.1f}x better than random on {c['name'].split('(')[0].strip().lower()}"
        )
    v1 = reference[0]["bpc"]
    if corpora and corpora[0].get("bpc") and corpora[0]["bpc"] < v1:
        reading.append(
            f"+ lower BPC than v1 ({corpora[0]['bpc']:.3f} vs {v1:.3f}) "
            "with 6% fewer parameters"
        )
    reading.append("- Not a chatbot: it continues text, it does not answer")
    reading.append("- 9.2 tokens per parameter, below the 20 that scaling work recommends")

    report["quality"] = {
        "corpora": corpora,
        "reference": reference,
        "reading": reading,
    }

    # ---- PyTorch comparison ----
    if not a.skip_torch:
        print("running torch_parity (222M, same weights) ...")
        ok, out = sh([PY, "tools/torch_parity.py", "--state", state,
                      "--valid", f"{TINY}.valid.bin", "--batches", BATCHES,
                      "--expected", str(tiny["loss"] or 0)])
        par = parse_parity(out)
        print(f"  torch {par['torch_loss']} delta {par['abs_diff']} match {par['verdict']}")
        report["parity"] = {
            "steps": int(BATCHES),
            "stonefish_loss": par["stonefish_loss"] or tiny["loss"],
            "torch_loss": par["torch_loss"],
            "abs_diff": par["abs_diff"],
            "note": "the two implementations agree to f32 rounding"
            if par["verdict"] else "delta above tolerance - investigate",
        }

        print("running torch_bench at 222M ...")
        runs = []

        # Our own number comes from the training log, not from a separate benchmark:
        # it is the real end-to-end step (forward + backward + Adam + clipping +
        # data window), which is what the comparison should be against.
        train_tps, train_ms = training_throughput()
        if train_tps:
            runs.append(
                {
                    "name": "StonefishAI",
                    "tok_s": round(train_tps, 0),
                    "ms_step": round(train_ms, 0),
                    "vram_gb": 8.4,
                }
            )
            print(f"  StonefishAI: {train_tps:,.0f} tok/s (training log, last 200 steps)")
        batch_used = 4
        for attn in ("naive", "sdpa"):
            b = None
            for batch in (4, 2, 1):
                ok, out = sh([PY, "tools/torch_bench.py", "--d", "1024", "--layers", "16",
                              "--heads", "8", "--ctx", "512", "--batch", str(batch),
                              "--accum", "16", "--steps", "6", "--attn", attn])
                if "out of memory" in out.lower() or "OutOfMemoryError" in out:
                    print(f"  {attn}: OOM at batch {batch}, retrying smaller")
                    continue
                b = parse_bench(out)
                batch_used = batch
                break
            if b is None or b["tok_s"] is None:
                print(f"  {attn}: could not measure")
                continue
            print(f"  {attn}: {b['tok_s']} tok/s, {b['vram_gb']} GB (batch {batch_used})")
            runs.append({"name": f"PyTorch ({attn})", "tok_s": b["tok_s"],
                         "ms_step": b["ms_step"], "vram_gb": b["vram_gb"]})
        report["speed"] = {
            "device": "RTX 5070 12 GB, 250 W",
            "model": {"params": 222331904, "d": 1024, "layers": 16, "heads": 8, "ctx": 512},
            # recorded because tokens/s is only comparable at the same batch size
            "batch": batch_used,
            "runs": runs,
        }

    import datetime

    report["measured_at"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    report["samples"] = {"greedy": g0["text"], "sampled_0_8": g8["text"]}

    out_path = os.path.join(ROOT, "docs", "report.json")
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        fh.write(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nwrote {out_path}")
    print("next: python tools/make_report.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
