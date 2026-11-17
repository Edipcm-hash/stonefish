"""Applies line-by-line Turkish -> English translation.

Dictionary: tools/_tr_XXX.tsv files (TAB separated: TURKISH <TAB> ENGLISH).
Works line by line: if the trimmed form of a line is in the dictionary it is
replaced by its English version, keeping the indentation.

Usage:
    python tools/apply_tr.py            # uygula + rapor
    python tools/apply_tr.py --dry-run  # report only
"""

import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXTENSIONS = (".rs", ".cu", ".py", ".yml", ".toml", ".md")
SKIP_DIRS = {"target", "target2", "data", ".git", "runs"}
SKIP_FILES = {"README-Tr.md", "_tr_lines.txt"}

TR_CHARS = re.compile(r"[\u00E7\u011F\u0131\u00F6\u015F\u00FC\u00C7\u011E\u0130\u00D6\u015E\u00DC]")
# Turkish-specific words with no English homograph (low false-positive risk)
TR_WORDS = re.compile(
    r"\b(bir|ve|için|ile|bu|olarak|değil|yok|hata|boyut|satır|sütun|değer|sonra|önce|"
    r"gibi|daha|çok|veya|eğer|zaman|katman|çekirdek|bellek|tampon|gradyan|ağırlık|"
    r"eğitim|dizi|örnek|toplam|ortalama|sonuç|giriş|çıkış|adım|hız|süre|yaz|döner|"
    r"kullan|yapılır|edilir|olur|olan|olduğu|tarafından|üzerinde|içinde|arasında|"
    r"kadar|sadece|ayrıca|ancak|çünkü|böylece|şekilde|durum|işlem|değişken|döngü|"
    r"koşul|sınır|kontrol|hesap|kaydet|yükle|sil|ekle|başlat|bitir|açık|kapalı)\b"
)


def load_dict():
    d = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "tools", "_tr_[0-9]*.tsv"))):
        with open(path, encoding="utf-8") as f:
            for ln in f:
                ln = ln.rstrip("\n")
                if not ln.strip() or "\t" not in ln:
                    continue
                tr, en = ln.split("\t", 1)
                d[tr.strip()] = en
    return d


def is_turkish(line):
    return bool(TR_CHARS.search(line) or TR_WORDS.search(line))


def targets():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn in SKIP_FILES or not fn.endswith(EXTENSIONS):
                continue
            yield os.path.join(dirpath, fn)


def main():
    dry = "--dry-run" in sys.argv
    d = load_dict()
    print(f"dictionary: {len(d)} lines")

    replaced = 0
    remaining = {}
    for path in targets():
        rel = os.path.relpath(path, ROOT)
        with open(path, encoding="utf-8") as f:
            lines = f.read().split("\n")
        changed = False
        for i, ln in enumerate(lines):
            key = ln.strip()
            if key in d and d[key] != key:
                indent = ln[: len(ln) - len(ln.lstrip())]
                lines[i] = indent + d[key]
                replaced += 1
                changed = True
        if changed and not dry:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write("\n".join(lines))
        # remaining Turkish lines
        left = [ln for ln in lines if is_turkish(ln)]
        if left:
            remaining[rel] = left

    print(f"lines replaced: {replaced}")
    total_left = sum(len(v) for v in remaining.values())
    print(f"Turkish lines left: {total_left} (files: {len(remaining)})")
    for rel in sorted(remaining, key=lambda k: -len(remaining[k])):
        print(f"  {len(remaining[rel]):4d}  {rel}")
    # write the leftovers to a file for review
    with open(os.path.join(ROOT, "tools", "_tr_remaining.txt"), "w", encoding="utf-8") as f:
        for rel in sorted(remaining):
            for ln in remaining[rel]:
                f.write(ln.strip() + "\n")


if __name__ == "__main__":
    main()
