"""Guard: no Turkish text may appear outside README-Tr.md.

Turkish-specific characters are written as unicode escapes so that this file
itself contains no Turkish characters - otherwise the check would flag itself.

Run from the repository root:  python tools/check_no_turkish.py
Exit code 1 if anything is found, so it can gate CI.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# c, g-breve, dotless i, o-umlaut, s-cedilla, u-umlaut and their uppercase forms
TURKISH = re.compile(r"[\u00E7\u011F\u0131\u00F6\u015F\u00FC\u00C7\u011E\u0130\u00D6\u015E\u00DC]")

EXTENSIONS = (".rs", ".cu", ".py", ".yml", ".toml", ".md", ".txt")
# Extensionless files that are still part of the project.
EXACT_FILES = {".gitignore", ".gitattributes"}
SKIP_DIRS = {"target", "target2", "data", "runs", ".git"}
ALLOWED = {"README-Tr.md"}


def main():
    hits = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn in ALLOWED or not (fn.endswith(EXTENSIONS) or fn in EXACT_FILES):
                continue
            path = os.path.join(dirpath, fn)
            try:
                with open(path, encoding="utf-8") as f:
                    for i, line in enumerate(f, 1):
                        if TURKISH.search(line):
                            hits.append((os.path.relpath(path, ROOT), i, line.strip()))
            except (UnicodeDecodeError, OSError):
                continue

    if hits:
        print(f"Turkish text found outside README-Tr.md: {len(hits)} line(s)")
        for rel, i, line in hits[:50]:
            print(f"  {rel}:{i}  {line[:100]}")
        return 1
    print("OK: no Turkish text outside README-Tr.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
