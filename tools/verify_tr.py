"""Çeviri doğrulaması (derleme yapmadan).

Kontroller:
  1) Her TSV çiftinde `{}` yer tutucu sayısı korunmuş mu? (biçim dizgisi bozulması)
  2) `"` (çift tırnak) sayısı korunmuş mu? (string yapısı)
  3) Yorum işareti (`//`, `///`, `#`) korunmuş mu? (kod satırı yanlışlıkla yorum olmasın)
  4) Çevrilen tüm dosyalarda `"` sayısı çift mi? (dengesiz string yok)
"""

import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_pairs():
    pairs = []
    for path in sorted(glob.glob(os.path.join(ROOT, "tools", "_tr_[0-9]*.tsv"))):
        with open(path, encoding="utf-8") as f:
            for ln in f:
                ln = ln.rstrip("\n")
                if not ln.strip() or "\t" not in ln:
                    continue
                tr, en = ln.split("\t", 1)
                pairs.append((tr.strip(), en))
    return pairs


def main():
    pairs = load_pairs()
    problems = 0
    for tr, en in pairs:
        # 1) yer tutucu sayısı
        if tr.count("{") != en.count("{"):
            print(f"YER TUTUCU  TR[{tr}]  EN[{en}]")
            problems += 1
        # 2) tırnak sayısı
        if tr.count('"') != en.count('"'):
            print(f"TIRNAK      TR[{tr}]  EN[{en}]")
            problems += 1
        # 3) yorum işareti
        for marker in ("///", "//", "#"):
            if tr.startswith(marker) != en.startswith(marker):
                print(f"YORUM       TR[{tr}]  EN[{en}]")
                problems += 1
                break

    print(f"\nçift sayısı: {len(pairs)} · sorun: {problems}")

    # 4) dosyalarda dengeli tırnak
    bad_files = []
    for path in glob.glob(os.path.join(ROOT, "src", "**", "*.rs"), recursive=True) + glob.glob(
        os.path.join(ROOT, "tests", "*.rs")
    ):
        with open(path, encoding="utf-8") as f:
            for i, ln in enumerate(f, 1):
                # yorum satırlarını atla
                code = ln.split("//")[0]
                if code.count('"') % 2 != 0:
                    bad_files.append((os.path.relpath(path, ROOT), i, ln.strip()))
    print(f"dengesiz tırnaklı satır: {len(bad_files)}")
    for rel, i, ln in bad_files[:10]:
        print(f"  {rel}:{i}  {ln[:90]}")


if __name__ == "__main__":
    main()
