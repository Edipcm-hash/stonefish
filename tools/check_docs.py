#!/usr/bin/env python3
"""Check that every relative link in the READMEs points at something real.

Catches the failure mode that matters before a release: an image or source file
renamed, moved or never committed, leaving a broken link on the repository front
page. External URLs are not fetched (CI has no business making network calls for
a link check); only relative paths are verified.

    python tools/check_docs.py

Exit code 1 lists the broken links.
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ["README.md", "README-Tr.md", "README-Eng.md", "docs/MODEL-CARD.md", "CHANGELOG.md"]

# ![alt](target) or [text](target)
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def main() -> int:
    broken = []
    checked = 0

    for rel in FILES:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            broken.append((rel, "<file missing>"))
            continue
        base = os.path.dirname(path)
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for lineno, line in enumerate(fh, 1):
                for target in LINK.findall(line):
                    t = target.strip()
                    if t.startswith(("http://", "https://", "mailto:", "#")):
                        continue
                    t = t.split("#", 1)[0].split("?", 1)[0]
                    if not t:
                        continue
                    checked += 1
                    resolved = os.path.normpath(os.path.join(base, t))
                    if not os.path.exists(resolved):
                        broken.append((f"{rel}:{lineno}", target))

    if broken:
        print(f"broken links: {len(broken)} of {checked} checked")
        for where, target in broken:
            print(f"  {where}  ->  {target}")
        return 1

    print(f"OK: {checked} relative links resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
