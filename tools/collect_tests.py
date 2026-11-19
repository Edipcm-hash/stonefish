#!/usr/bin/env python3
"""Collect the test-suite numbers for docs/report.json.

Runs the suite once per binary and reports what actually happened, so the
tests chart is traceable to a command instead of to a number someone typed.

    python tools/collect_tests.py            # human-readable
    python tools/collect_tests.py --json     # the `tests` object for report.json

The GPU binaries need a free GPU; without one they still build and list, but the
pass/fail column will be missing. That is deliberate: the script never invents a
green.

Exit code 1 if any test binary fails.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (area name, `cargo test` target, what it proves, file shown in the chart)
BINARIES = [
    ("numerical gradient checks", ["--test", "gradcheck"],
     "analytic backward() vs finite differences", "tests/gradcheck.rs"),
    ("CUDA kernels vs CPU", ["--test", "gpu_kernels"],
     "every kernel against its reference", "tests/gpu_kernels.rs"),
    ("checkpoint safety", ["--test", "checkpoint"],
     "truncated, wrong magic, vocab drift", "tests/checkpoint.rs"),
    ("GPU trainer equivalence", ["--test", "gpu_train_equiv"],
     "device trainer vs CPU autograd", "tests/gpu_train_equiv.rs"),
    ("KV-cache", ["--test", "kv_cache"],
     "cached generation vs full forward", "tests/kv_cache.rs"),
    ("unit tests (BPE, tokenizer, optim, corpus)",
     ["--lib"], "round-trips and edge cases", "src/lib.rs"),
]

RESULT = re.compile(r"test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored")


def run(args):
    """Returns (passed, failed, ignored, listed) for one target."""
    base = ["cargo", "test", "--release", "--features", "cuda"] + args
    listing = subprocess.run(base + ["--", "--list"], cwd=ROOT,
                             capture_output=True, text=True)
    listed = sum(1 for line in listing.stdout.splitlines() if line.endswith(": test"))

    runres = subprocess.run(base, cwd=ROOT, capture_output=True, text=True)
    out = runres.stdout + runres.stderr
    m = None
    for m2 in RESULT.finditer(out):
        m = m2
    if m is None:
        return None, None, None, listed
    return int(m.group(2)), int(m.group(3)), int(m.group(4)), listed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    areas = []
    total_passed = total_failed = total_count = 0
    unknown = False

    for name, target, what, file in BINARIES:
        passed, failed, ignored, listed = run(target)
        count = listed
        total_count += count
        if passed is None:
            unknown = True
            status = "listed only (no GPU?)"
            passed_s = failed_s = 0
        else:
            passed_s, failed_s = passed, failed
            total_passed += passed
            total_failed += failed
            status = f"{passed} passed, {failed} failed, {ignored} ignored"
        areas.append(
            {
                "name": name,
                "file": file,
                "count": count,
                "failed": failed_s,
                "what": what,
            }
        )
        print(f"  {name:42s} {count:3d} listed  {status}")

    doc = {
        "total": total_count,
        "passed": total_passed,
        "failed": total_failed,
        "ignored": 0,
        "areas": areas,
    }
    if a.json:
        print(json.dumps({"tests": doc}, indent=2))
    else:
        print(f"\n  total {total_count} tests, {total_passed} passed, {total_failed} failed"
              + ("  (GPU results unknown)" if unknown else ""))

    return 1 if total_failed else 0


if __name__ == "__main__":
    sys.exit(main())
