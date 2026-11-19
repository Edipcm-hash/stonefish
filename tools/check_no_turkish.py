#!/usr/bin/env python3
"""Fail if Turkish text appears outside README-Tr.md.

Two passes, because the first one alone is not enough:

1. Turkish-specific characters (c-cedilla, g-breve, dotless i, s-cedilla,
   u-umlaut, o-umlaut). Catches most prose but misses words written without
   diacritics - which is exactly how the leaks that survived the first pass
   looked: "CUDA testi", "Bellek", "gradyan".

2. A curated word list of Turkish-only tokens that have no English homograph.
   Whole-word, case-insensitive. This is what catches "testi" and "bellek";
   it deliberately excludes short particles (ve, bir, bu, ne, cok) that would
   produce false positives inside identifiers and English sentences.

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
SKIP_DIRS = {".git", "target", "__pycache__", "data", "docs", "runs", "node_modules"}

# Turkish characters that do not occur in English text.
TURKISH_CHARS = re.compile(r"[çğıöşüÇĞİÖŞÜ]")

# Turkish-only words: no English homograph, so a whole-word hit is a real leak.
WORDS = [
    # nouns / technical
    "testi", "bellek", "adim", "hata", "kayit", "egitim", "cikti", "ornek",
    "deger", "boyut", "katman", "gradyan", "kayip", "dosya", "ayar", "surum",
    "toplam", "ortalama", "hesap", "kontrol", "baslangic", "bitis", "sonuc",
    "girdi", "cikis", "katmani", "katmanlar", "parametre", "bagimlilik",
    "calisma", "olcum", "karsilastirma", "dogrulama", "gozlem",
    # adjectives / adverbs with no English twin
    "buyuk", "kucuk", "yeni", "eski", "dogru", "yanlis", "guncel", "ayni",
    "farkli", "gerekli", "mumkun", "degil", "uzerinde", "arasinda", "olarak",
    "icin", "sonra", "simdi", "birlikte", "yaklasik",
    # verb forms that only make sense in Turkish
    "kaydedildi", "hesaplandi", "yazildi", "okundu", "bulundu", "eklendi",
    "silindi", "guncellendi", "baslatildi", "tamamlandi", "denendi",
    "atlandi", "secildi", "kullanilir", "kullanilan", "kullanilarak",
    "calisiyor", "calismiyor", "calistir", "degistir", "kaldir", "ekle",
    "aciklama", "uyari", "bilgi", "mesaj", "sifir", "bos", "dolu", "zirve", "tepe", "son loss", "yolu", "kaydediliyor", "kaydedilir",
]
WORD_RE = re.compile(r"\b(" + "|".join(WORDS) + r")\b", re.IGNORECASE)


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
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
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
