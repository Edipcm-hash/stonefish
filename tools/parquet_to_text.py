"""Parquet Gutenberg shard'larını düz metne çevirir (veri hazırlığı; proje bağımlılığı değil)."""
import sys, glob, os
import pyarrow.parquet as pq

def main():
    files = sorted(glob.glob("data/gb/*.parquet"))
    if not files:
        print("parquet yok")
        return
    out_path = "data/gb/books.txt"
    total_chars = 0
    total_rows = 0
    with open(out_path, "w", encoding="utf-8", newline="\n") as out:
        for f in files:
            pf = pq.ParquetFile(f)
            cols = pf.schema_arrow.names
            text_col = "text" if "text" in cols else cols[0]
            for batch in pf.iter_batches(batch_size=2000, columns=[text_col]):
                for t in batch.column(0).to_pylist():
                    if not t:
                        continue
                    t = t.replace("\r\n", "\n").replace("\r", "\n")
                    out.write(t)
                    if not t.endswith("\n"):
                        out.write("\n")
                    out.write("\n")
                    total_chars += len(t) + 2
                    total_rows += 1
            print(f"{os.path.basename(f)}: {total_rows} belge, {total_chars/1e6:.1f}M karakter", flush=True)
    print(f"TOPLAM: {total_rows} belge, {total_chars/1e6:.1f}M karakter -> {out_path}")

if __name__ == "__main__":
    main()
