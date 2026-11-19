#!/usr/bin/env python3
"""Check that every link in the documentation points at something real.

Two kinds of link, two kinds of failure:

* relative paths - an image or source file renamed, moved or never committed,
  which leaves a broken image on the repository front page;
* heading anchors - the table of contents drifting away from the sections, which
  silently turns a working link into a no-op.

External URLs are not fetched (CI has no business making network calls for a
link check).

    python tools/check_docs.py

Exit code 1 lists everything that does not resolve.
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = [
    "README.md",
    "README-Tr.md",
    "README-Eng.md",
    "docs/MODEL-CARD.md",
    "docs/REPRODUCING.md",
    "docs/RELEASE.md",
    "docs/huggingface-README.md",
    "CHANGELOG.md",
]

# ![alt](target) or [text](target)
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def github_slug(heading: str) -> str:
    """GitHub's heading-anchor rule.

    Lowercase, drop everything that is not a letter/number/space/hyphen/
    underscore, then replace EACH space with a hyphen. The one-to-one space
    replacement is what makes "a / b" become "a--b" (the slash disappears, both
    spaces stay); collapsing runs of whitespace gives a false positive there.
    """
    t = heading.strip().lower()
    t = re.sub(r"[^\w\s-]", "", t)
    return t.replace(" ", "-")


def main() -> int:
    broken = []
    checked = 0
    anchors_checked = 0

    for rel in FILES:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            broken.append((rel, "<file missing>"))
            continue
        base = os.path.dirname(path)
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()

        headings = set()
        for line in lines:
            m = HEADING.match(line)
            if m:
                headings.add(github_slug(m.group(2)))

        for lineno, line in enumerate(lines, 1):
            for target in LINK.findall(line):
                t = target.strip()
                if t.startswith(("http://", "https://", "mailto:")):
                    continue
                if t.startswith("#"):
                    anchors_checked += 1
                    if t[1:] not in headings:
                        broken.append((f"{rel}:{lineno}", target))
                    continue
                path_part = t.split("#", 1)[0].split("?", 1)[0]
                if not path_part:
                    continue
                checked += 1
                resolved = os.path.normpath(os.path.join(base, path_part))
                if not os.path.exists(resolved):
                    broken.append((f"{rel}:{lineno}", target))

    if broken:
        print(
            f"broken links: {len(broken)} "
            f"({checked} paths, {anchors_checked} anchors checked)"
        )
        for where, target in broken:
            print(f"  {where}  ->  {target}")
        return 1

    print(f"OK: {checked} relative links resolve, {anchors_checked} anchors resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
