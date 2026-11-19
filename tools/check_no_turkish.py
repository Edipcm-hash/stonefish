#!/usr/bin/env python3
"""Fail if Turkish text appears outside README-Tr.md.

Three passes, because each one alone misses real cases:

1. Turkish-specific characters (c-cedilla, g-breve, dotless i, s-cedilla,
   u-umlaut, o-umlaut). Catches most prose but misses words written without
   diacritics - which is how most short labels in code look.

2. A curated list of Turkish-only stems with no English homograph, matched as
   word prefixes: Turkish agglutinates, so "boyut" also appears as "boyutu" and
   "boyutlar", and "kaydedil" as "kaydedildi" or "kaydedilemedi". Matching a
   stem plus up to six trailing letters catches those without matching English
   (the stems themselves do not occur in English).

3. A per-line opt-out: a line containing `tr:sample` is skipped, used for the one
   place the README quotes the Turkish model's own output as evidence.

This is a net, not a proof of absence: a Turkish sentence whose words are all
absent from the list will pass. The README says as much.

Exit code 1 with a line list when anything is found.
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALLOWED = {"README-Tr.md", "tools/check_no_turkish.py"}
EXTENSIONS = (".rs", ".cu", ".py", ".yml", ".toml", ".md", ".txt")
# Extensionless files that are still part of the project.
EXACT_FILES = {".gitignore", ".gitattributes"}
SKIP_DIRS = {".git", "__pycache__", "data", "docs", "runs", "node_modules"}
# target, target2, target3 ... are build directories (and `cargo package`
# output lands inside them).
SKIP_PREFIXES = ("target",)

# Turkish characters that do not occur in English text.
TURKISH_CHARS = re.compile(r"[çğıöşüÇĞİÖŞÜ]")

# Turkish-only stems: no English homograph, so a prefix hit is a real leak.
WORDS = [
    # nouns / technical
    "testi", "bellek", "adim", "hata", "kayit", "egitim", "cikti", "ornek",
    "deger", "boyut", "katman", "gradyan", "kayip", "dosya", "ayar", "surum",
    "toplam", "ortalama", "hesap", "kontrol", "baslangic", "bitis", "sonuc",
    "girdi", "cikis", "parametre", "bagimlilik", "calisma", "olcum",
    "karsilastir", "dogrula", "gozlem", "korpus", "zirve", "tepe",
    # adjectives / adverbs
    "buyuk", "kucuk", "yeni", "dogru", "yanlis", "guncel", "ayni", "farkli",
    "gerekli", "mumkun", "degil", "uzerinde", "arasinda", "olarak", "icin",
    "sonra", "simdi", "birlikte", "yaklasik", "bilinmeyen", "gecersiz",
    # verb stems (agglutinate heavily)
    "kaydedil", "hesapla", "yazil", "okun", "bulun", "ekle", "silin",
    "guncellen", "baslatil", "tamamlan", "denen", "atlan", "secil",
    "kullanil", "calis", "degistir", "kaldir", "acikla", "uyar", "mesaj",
    "sifir", "dolu", "yolu",
]
# Deliberately absent: short particles that live inside identifiers or English
# words (ve, bir, bu, ne, cok, bos, son, on, bin, en, az).

# Stem + up to six trailing letters covers Turkish suffixes.
WORD_RE = re.compile(r"\b(" + "|".join(WORDS) + r")\w{0,6}\b", re.IGNORECASE)


def scan_line(line: str) -> str | None:
    if "tr:sample" in line:  # documented quote of the Turkish model's output
        return None
    m = TURKISH_CHARS.search(line)
    if m:
        return f"diacritic '{m.group(0)}'"
    m = WORD_RE.search(line)
    if m:
        return f"word '{m.group(0)}'"
    return None


def main() -> int:
    hits = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [
            d
            for d in dirnames
            if d not in SKIP_DIRS and not d.startswith(SKIP_PREFIXES)
        ]
        for fn in filenames:
            if fn in ALLOWED or fn == "check_no_turkish.py":
                continue
            if not (fn.endswith(EXTENSIONS) or fn in EXACT_FILES):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, ROOT).replace("\\", "/")
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    for i, line in enumerate(fh, 1):
                        why = scan_line(line)
                        if why:
                            hits.append((rel, i, why, line.rstrip()[:110]))
            except OSError as exc:
                print(f"could not read {rel}: {exc}", file=sys.stderr)

    if hits:
        print(f"Turkish text found outside README-Tr.md: {len(hits)} line(s)")
        for rel, i, why, text in hits:
            print(f"  {rel}:{i}  [{why}]  {text}")
        return 1

    print("OK: no Turkish text outside README-Tr.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
