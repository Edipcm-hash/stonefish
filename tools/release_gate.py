#!/usr/bin/env python3
"""The release gate: every check that must be green before a tag.

One command instead of nine remembered ones, and a table that says exactly what
failed instead of making you scroll a build log.

    python tools/release_gate.py            # run everything
    python tools/release_gate.py --fast     # skip the test run (slowest step)

Exit code 1 if any check fails.

This mirrors docs/RELEASE.md section 0. If a check is added there, add it here.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def run(cmd, env=None, shell=False):
    """Returns (ok, combined_output)."""
    p = subprocess.run(
        cmd, cwd=ROOT, capture_output=True, text=True, env=env, shell=shell
    )
    return p.returncode == 0, (p.stdout or "") + (p.stderr or "")


def check_git_clean():
    ok, out = run(["git", "status", "--porcelain"])
    if not ok:
        return False, "git status failed"
    lines = [ln for ln in out.splitlines() if ln.strip()]
    return (not lines), "clean" if not lines else f"{len(lines)} uncommitted change(s)"


def check_zero_deps():
    ok, out = run(["cargo", "tree", "--no-default-features"])
    if not ok:
        return False, "cargo tree failed"
    lines = [ln for ln in out.splitlines() if ln.strip()]
    return len(lines) == 1, f"{len(lines)} line(s): " + (lines[0].strip() if lines else "")


def check_tests(env):
    ok, out = run(["cargo", "test", "--release", "--features", "cuda"], env=env)
    total = failed = 0
    for line in out.splitlines():
        if "test result:" in line:
            parts = line.replace(",", " ").split()
            for i, p in enumerate(parts):
                if p == "passed;":
                    total += int(parts[i - 1])
                if p == "failed;":
                    failed += int(parts[i - 1])
    if not ok or failed:
        return False, f"{total} passed, {failed} failed"
    return total > 0, f"{total} passed, 0 failed"


def check_clippy(env, features):
    cmd = ["cargo", "clippy", "--all-targets"]
    cmd += ["--features", "cuda"] if features else ["--no-default-features"]
    cmd += ["--", "-D", "warnings"]
    ok, out = run(cmd, env=env)
    if ok:
        return True, "clean"
    errs = [ln for ln in out.splitlines() if ln.startswith("error")]
    return False, (errs[0][:80] if errs else "unknown error")


def check_fmt():
    ok, out = run(["cargo", "fmt", "--all", "--", "--check"])
    return ok, "clean" if ok else "diff (run: cargo fmt --all)"


def check_script(name):
    ok, out = run([PY, os.path.join("tools", name)])
    first = out.strip().splitlines()[-1] if out.strip() else "no output"
    return ok, first[:90]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip the test suite")
    a = ap.parse_args()

    env = dict(os.environ)
    env.setdefault("CARGO_TARGET_DIR", "target3")

    checks = [
        ("git clean", check_git_clean),
        ("zero dependencies", check_zero_deps),
        ("fmt", check_fmt),
        ("clippy (cuda)", lambda: check_clippy(env, True)),
        ("clippy (cpu)", lambda: check_clippy(env, False)),
        ("language guard", lambda: check_script("check_no_turkish.py")),
        ("links + anchors", lambda: check_script("check_docs.py")),
        ("headline numbers", lambda: check_script("check_numbers.py")),
    ]
    if not a.fast:
        checks.insert(2, ("tests (75)", lambda: check_tests(env)))

    failures = 0
    print(f"release gate - {len(checks)} checks\n")
    for name, fn in checks:
        try:
            ok, detail = fn()
        except Exception as exc:  # a broken check must not look green
            ok, detail = False, f"check crashed: {exc}"
        mark = "OK  " if ok else "FAIL"
        print(f"  {mark}  {name:20s} {detail}")
        if not ok:
            failures += 1

    print()
    if failures:
        print(f"{failures} check(s) failed - do not tag")
        return 1
    print("all checks green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
