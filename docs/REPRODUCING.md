# Reproducing the numbers in the README

Every table in the README comes from one of the commands below, run on the
machine described in §3 (one RTX 5070, 12 GB, 250 W). Nothing is estimated; if a
number is not in this file, it is not in the README.

Run them in this order — later steps assume the checkpoint from the earlier ones.

## 0. Prerequisites

```bash
cargo build --release                     # needs CUDA for the GPU paths

# data (not committed: 4 GB of token cache would bloat the repo)
python tools/parquet_to_text.py           # Gutenberg parquet -> text, optional
cargo run --release -- en-prepare         # BPE + u16 caches under data/en/
```

`en-prepare` writes `data/en/tiny.{bpe,train.bin,valid.bin}`. Re-running it is
idempotent unless the corpus changed, in which case the vocabulary hash in the
cache changes and training refuses to start — that is deliberate.

## 1. Tests

```bash
cargo test --release                       # 75 tests
cargo test --release --test gradcheck      # 32 numerical gradient checks
cargo test --release --test gpu_kernels    # 16 kernel-vs-CPU checks
```

Expected: `75 passed; 0 failed`. If the GPU is busy, the four GPU test binaries
still run but slower; they are not skipped.

## 2. Train (optional — this is the 56-hour step)

```bash
cargo run --release -- en-train 2047457313   # total tokens
```

The run writes `stonefish_en.state` every 500 steps (atomic: temp file,
`sync_all`, rename) and `stonefish_en.final.state` at the end. To resume, pass the
same token budget again: the trainer reads `adam_t` from the checkpoint.

## 3. Evaluate

```bash
cargo run --release -- en-eval 8 data/en/tiny stonefish_en.final.state
```

Prints, for the first 8 batches of 8×512 tokens:

```
Model: d=1024, layers=16, heads=8, ctx=512, vocab=16384 (step 62483)
loss : 3.xxxx   ppl: xx.xx   BPC: 0.xxxx
```

BPC is `loss / ln(2) / (bytes(valid.txt) / tokens(valid.bin))`, so it does not
depend on the tokeniser and can be compared with other models.

## 4. Generate

```bash
cargo run --release -- en-gen "Once upon a time" 150 0.0   # greedy
cargo run --release -- en-gen "Once upon a time" 150 0.8   # sampled

# The same command works without CUDA (`--no-default-features`): the CPU path
# loads the SFEN001 checkpoint into the CPU model and samples there at ~3 tok/s.
```

The README quotes both. Greedy shows what the model is most confident about;
temperature 0.8 shows what it does when allowed to wander. Nothing is filtered.

## 5. Cross-framework parity

The check that makes the correctness claim:

```bash
cargo run --release -- en-eval 8 data/en/tiny stonefish_en.final.state
python tools/torch_parity.py --state stonefish_en.final.state \
    --valid data/en/tiny.valid.bin --batches 8 --expected <the loss printed above>
```

`tools/torch_parity.py` parses the `SFEN001` checkpoint directly, rebuilds the
same architecture in PyTorch, loads the weights (transposing, because our `W` is
`(in, out)` and `nn.Linear` stores `(out, in)`), and evaluates the same batches.
It prints `MATCH` when the two losses differ by less than 2e-3.

## 6. Speed

**Measure on an idle GPU.** Every GPU number in this file was taken on a card
with nothing else running. Sharing the card with another CUDA process does not
scale the timings by a constant: a small matmul that takes 0.19 ms alone measured
2.99 ms while a training run owned the card - a 16x difference - because each
call waits for the other context's kernels. The CPU numbers are unaffected, so
the contamination is easy to miss: the comparison silently inverts.



```bash
python tools/torch_bench.py --d 1024 --layers 16 --heads 8 --ctx 512 \
    --batch 4 --accum 16 --steps 6 --attn naive
python tools/torch_bench.py ... --attn sdpa
```

The Rust side of the same measurement is the training log itself: the `tok/s`
column the trainer prints every 20 steps, averaged over the run.

## 7. Charts

```bash
# one command for the whole measurement: runs en-eval twice, generates two
# samples, compares against PyTorch, and writes docs/report.json from the
# parsed output - no number is typed by hand
python tools/measure.py

python tools/make_charts.py        # loss-curve.png, throughput.png, speedup.png
python tools/make_report.py        # tests.png, comparison.png, quality.png
python tools/make_architecture.py  # architecture.png
```

`make_charts.py` reads `en-train-v2*.log` from the repository root and merges the
two logs by step (the run was restarted once, so neither file holds the whole
curve). `make_report.py` reads `docs/report.json`, which the steps above fill in.

## 8. Guards

```bash
python tools/check_no_turkish.py   # no Turkish outside README-Tr.md
python tools/check_docs.py         # every relative link resolves
```

Both run in CI. The language guard has two passes: Turkish-specific characters,
and a word list of Turkish-only tokens — the second exists because the first
missed short labels written without diacritics, which is most of them in code.
