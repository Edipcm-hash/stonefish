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
base_dir = ROOT
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


TABLE_SEP = re.compile(r"^\|[\s:\-|]+\|$")


def check_tables(rel: str, lines) -> list[tuple[str, str]]:
    """A markdown table renders correctly only if every row has the same number
    of pipes as the header. Editing a table by hand is exactly how that breaks,
    and the result is a mangled table on the repository front page."""
    bad = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")
        if line.strip().startswith("|") and i + 1 < len(lines):
            sep = lines[i + 1].strip()
            if TABLE_SEP.match(sep) and "-" in sep:
                width = line.count("|")
                j = i + 1
                while j < len(lines) and lines[j].strip().startswith("|"):
                    if lines[j].strip().count("|") != width:
                        bad.append((f"{rel}:{j + 1}", "table row width"))
                    j += 1
                i = j
                continue
        i += 1
    return bad


LINE_ROW = re.compile(r"\| \[([^\]]+)\]\([^)]+\) \| (\d+) \|")


def check_line_counts(rel: str, text: str) -> list[tuple[str, str]]:
    """The architecture table claims a line count per file.

    Those numbers are the easiest claim in the README to falsify and the easiest to
    forget: adding a function to checkpoint.rs quietly makes the table wrong. This
    reads each linked file and compares."""
    bad = []
    for m in LINE_ROW.finditer(text):
        path, claimed = m.group(1), int(m.group(2))
        full = os.path.join(base_dir, path)
        if not os.path.exists(full):
            bad.append((f"{rel}: table row", f"{path} does not exist"))
            continue
        with open(full, "r", encoding="utf-8", errors="replace") as fh:
            real = sum(1 for _ in fh)
        if real != claimed:
            bad.append((f"{rel}: table row", f"{path} says {claimed}, is {real}"))
    return bad


TREE_ROW = re.compile(
    r"^([\u2502 ]*)[\u251c\u2514]\u2500\u2500 ([\w.\-/]+/?)\s*(?:#(.*))?$"
)

# The tree lists data/ and runs/ with a comment saying they are not committed.
# Those two are gitignored by design, so they exist on a development machine and
# not in a fresh clone - which is exactly where this check has to pass.
NOT_COMMITTED = re.compile(
    r"git d\u0131\u015f\u0131|git disi|not in git|not committed|gitignore", re.IGNORECASE
)


def check_tree(rel: str, text: str) -> list[tuple[str, str]]:
    """The directory tree in section 13 names real paths.

    A tree diagram is a claim about the repository, and it silently rots when a
    file is renamed or a directory is added. This walks the diagram, tracking the
    parent from the indentation, and stats each entry - except entries whose own
    comment says they are not committed, which are absent in CI on purpose."""
    bad = []
    stack: list[str] = []
    for line in text.splitlines():
        m = TREE_ROW.match(line)
        if not m:
            continue
        prefix, name, comment = m.group(1), m.group(2), m.group(3) or ""
        depth = len(prefix) // 4
        stack = stack[:depth]
        if name == "StonefishAI/":
            stack = []
            continue
        bare = name.rstrip("/")
        full = "/".join(stack + [bare]) if stack else bare
        if name.endswith("/"):
            stack.append(bare)
        if bare == "...":
            continue
        if NOT_COMMITTED.search(comment):
            continue
        if not os.path.exists(os.path.join(base_dir, full)):
            bad.append((f"{rel}: tree", f"{full} does not exist"))
    return bad


def main() -> int:
    broken = []
    checked = 0
    anchors_checked = 0
    tables_checked = 0

    for rel in FILES:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            broken.append((rel, "<file missing>"))
            continue
        base = os.path.dirname(path)
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()

        headings = set()
        seen_headings = {}
        in_fence = False
        for lineno, line in enumerate(lines, 1):
            # A line starting with # inside a fenced block is a shell comment, not a
            # heading - counting those would both invent anchors and hide real
            # duplicates.
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            m = HEADING.match(line)
            if not m:
                continue
            slug = github_slug(m.group(2))
            if slug in seen_headings:
                # GitHub renames the second one to "slug-1", so an anchor link to
                # the bare slug silently jumps to the first - worth reporting.
                broken.append((f"{rel}:{lineno}", f"duplicate heading '{m.group(2).strip()}'"))
            else:
                seen_headings[slug] = lineno
            headings.add(slug)

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

        broken.extend(check_tables(rel, lines))
        broken.extend(check_line_counts(rel, "".join(lines)))
        broken.extend(check_tree(rel, "".join(lines)))
        tables_checked += 1

    if broken:
        print(
            f"broken docs: {len(broken)} "
            f"({checked} paths, {anchors_checked} anchors, {tables_checked} files)"
        )
        for where, target in broken:
            print(f"  {where}  ->  {target}")
        return 1

    print(
        f"OK: {checked} relative links resolve, {anchors_checked} anchors resolve, "
        f"tables in {tables_checked} files are well formed"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
