"""Audit: report code items that are declared but never referenced.

Static only (no compilation), so it can run while the GPU is busy training.
Counts word-boundary occurrences across src/, tests/ and examples/ and flags
functions/methods whose name appears once (the definition itself).

Usage:  python tools/audit_dead_code.py
"""

import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Names that are legitimately called from outside the source tree (test harness,
# trait impls, entry points) or are too generic to judge statically.
IGNORE = {
    "new", "main", "fmt", "default", "from", "drop", "clone", "forward",
    "backward", "parameters", "deref", "eq", "hash", "display", "next",
}


def read_all(pattern):
    parts = []
    for path in glob.glob(pattern, recursive=True):
        with open(path, encoding="utf-8") as f:
            parts.append(f.read())
    return "\n".join(parts)


def main():
    src_files = glob.glob(os.path.join(ROOT, "src", "**", "*.rs"), recursive=True)
    src = read_all(os.path.join(ROOT, "src", "**", "*.rs"))
    other = read_all(os.path.join(ROOT, "tests", "*.rs")) + read_all(
        os.path.join(ROOT, "examples", "*.rs")
    )

    dead = []
    for path in src_files:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            m = re.match(r"^\s*(?:pub(?:\(crate\))?\s+)?(?:async\s+)?fn\s+([a-z_][a-z0-9_]*)", line)
            if not m:
                continue
            name = m.group(1)
            if name in IGNORE:
                continue
            # skip #[test] functions (they are called by the harness)
            window = "".join(lines[max(0, i - 2): i + 1])
            if "#[test]" in window:
                continue
            esc = re.escape(name)
            hits = len(re.findall(rf"\b{esc}\b", src)) + len(re.findall(rf"\b{esc}\b", other))
            if hits <= 1:
                dead.append((os.path.relpath(path, ROOT), i + 1, name))

    if dead:
        print(f"unreferenced functions: {len(dead)}")
        for rel, line, name in dead:
            print(f"  {rel}:{line}  {name}")
    else:
        print("OK: no unreferenced functions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
