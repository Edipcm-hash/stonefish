#!/usr/bin/env python3
"""Check the README's headline numbers against the artifacts they describe.

The README makes a small number of claims that can be verified mechanically
instead of trusted: the kernel count, the parameter count, the vocabulary, the
token budget, the test count, the source size. Each one is either read out of a
file or recomputed, and anything whose input is missing (the gitignored corpora
and checkpoints) is reported as skipped rather than guessed.

    python tools/check_numbers.py

Exit code 1 if a claim disagrees with its artifact.
"""

from __future__ import annotations

import glob
import io
import os
import re
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

failures = 0
skipped = 0


def ok(what, claimed, actual, unit=""):
    global failures
    match = claimed == actual
    mark = "OK  " if match else "FAIL"
    print(f"  {mark}  {what:34s} {actual}{unit}" + ("" if match else f"   (README says {claimed})"))
    if not match:
        failures += 1


def skip(what, why):
    global skipped
    skipped += 1
    print(f"  skip  {what:34s} {why}")


def claim(pattern, path):
    """Pull a number out of a README."""
    with io.open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    m = re.search(pattern, text)
    return int(m.group(1).replace(".", "").replace(",", "")) if m else None


def main() -> int:
    print("headline numbers vs artifacts\n")

    # --- CUDA kernels: count the __global__ definitions ---
    cu = io.open("kernels/stonefish.cu", encoding="utf-8", errors="replace").read()
    kernels = len(re.findall(r"__global__\s+void\s+\w+", cu))
    ok("CUDA kernels", claim(r"(\d+) CUDA çekirdeği", "README-Tr.md"), kernels)  # tr:sample - literal matched in README-Tr.md

    # --- Op variants ---
    ag = io.open("src/autograd.rs", encoding="utf-8", errors="replace").read()
    i, j = ag.find("pub enum Op"), ag.find("\n}", ag.find("pub enum Op"))
    ops = len(re.findall(r"^    [A-Z][A-Za-z0-9_]*", ag[i:j], re.M))
    ok("autograd Op variants", claim(r"(\d+) op,", "README-Tr.md"), ops)

    # --- source size ---
    src = sorted(set(glob.glob("src/**/*.rs", recursive=True)))
    lines = sum(sum(1 for _ in io.open(f, encoding="utf-8", errors="replace")) for f in src)
    ok("Rust source lines", claim(r"\*\*([\d.]+) satır Rust\*\*", "README-Tr.md"), lines)  # tr:sample - literal matched in README-Tr.md

    # --- tests ---
    tf = glob.glob("tests/*.rs")
    total = 0
    for f in tf:
        text = io.open(f, encoding="utf-8", errors="replace").read()
        total += len(re.findall(r"#\[test\]", text))
    lib = io.open("src/lib.rs", encoding="utf-8", errors="replace").read()
    for f in glob.glob("src/**/*.rs", recursive=True):
        text = io.open(f, encoding="utf-8", errors="replace").read()
        total += len(re.findall(r"#\[test\]", text))
    # README claims the CUDA-build total, which includes gated tests
    claimed_tests = claim(r"\*\*(\d+) test\*\* CUDA", "README-Tr.md")
    ok("tests (all #[test] in tree)", claimed_tests, total)

    # --- tools ---
    tools = len(glob.glob("tools/*.py"))
    ok("Python tools", claim(r"(\d+) Python aracı", "README-Tr.md"), tools)  # tr:sample - literal matched in README-Tr.md

    # --- vocabulary, from the BPE file ---
    try:
        with open("data/en/tiny.bpe", "rb") as fh:
            fh.read(8)
            fh.read(4)
            merges = struct.unpack("<I", fh.read(4))[0]
        ok("vocabulary", 16384, 256 + merges)
    except OSError:
        skip("vocabulary", "data/en/tiny.bpe missing")

    # --- parameter count, from the checkpoint size ---
    for name in ("stonefish_en.final.state", "stonefish_en.state"):
        if os.path.exists(name):
            size = os.path.getsize(name)
            # 8 magic + 4 ver + 20 cfg + 8 adam_t + 4 n_tensors + 146 headers
            params = (size - (8 + 4 + 20 + 8 + 4) - 146 * 8) // 4
            ok("parameters", claim(r"\| Parametre \|[^|]*\| \*\*([\d.]+)\*\*", "README-Tr.md"), params)  # tr:sample - the literal this pattern matches in README-Tr.md
            break
    else:
        skip("parameters", "no checkpoint present")

    # --- token budget, from the u16 cache ---
    try:
        train = os.path.getsize("data/en/tiny.train.bin") // 2
        valid = os.path.getsize("data/en/tiny.valid.bin") // 2
        ok("training tokens", claim(r"TinyStories \+ Gutenberg, ([\d.]+) token", "README-Tr.md"), train)
        ok("validation tokens", 4041031, valid)
    except OSError:
        skip("token budget", "data/en/*.bin missing")

    print()
    if failures:
        print(f"{failures} claim(s) disagree with their artifact")
        return 1
    print(f"all claims match ({skipped} skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
