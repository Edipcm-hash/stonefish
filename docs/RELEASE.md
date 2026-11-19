# Cutting the v1.0 release

The release is deliberately boring: every claim is already measured and every
number in the README has a command behind it (`docs/REPRODUCING.md`). What is
left is packaging and the parts a human has to approve.

## 0. Before anything else

```bash
git status                 # must be clean
cargo tree                 # must be one line
cargo test --release       # 74 tests
cargo clippy --all-targets --features cuda -- -D warnings
cargo clippy --all-targets --no-default-features -- -D warnings
cargo fmt --all -- --check
python tools/check_no_turkish.py
python tools/check_docs.py
```

All seven green. If any of them is red, the release waits — that is the whole
point of having them.

## 1. The measurement that must exist in the README

These numbers cannot be copied from an older run:

| Number | Command | Lives in |
|---|---|---|
| val loss, perplexity, BPC | `en-eval data/en/tiny stonefish_en.final.state 8` | §6.2, §7 |
| out-of-distribution loss | `en-eval data/en/probe stonefish_en.final.state 8` | §7 |
| generation samples | `en-gen "Once upon a time" 150 0` and `... 0.8` | §6.2 |
| throughput at 222M | training log, `tok/s` column | §7.4 |
| PyTorch speed and parity | `tools/torch_bench.py`, `tools/torch_parity.py` | §7.5 |
| charts | `tools/make_charts.py`, `tools/make_report.py` | `docs/*.png` |

## 2. Tag

```bash
git tag -a v1.0.0 -m "v1.0.0 - 222M from-scratch language model, zero dependencies"
git push origin main --tags
```

Only after the repository is switched from private to public (or the tag will
point at a commit nobody can see).

## 3. GitHub release

Title: **v1.0.0 — a language model from scratch, zero dependencies**

Body: start from `CHANGELOG.md`, then put the four numbers a reader can check in
the first screen:

```
222,331,904 parameters · 2.05B tokens · 74 tests · 0 dependencies
val loss X.XXXX · BPC 0.XXXX · 10,7xx tokens/s on one RTX 5070
```

Attach: the loss curve and the comparison chart. **Do not attach the 848 MB
checkpoint** to a GitHub release — use Hugging Face (below) and link it.

## 4. Weights

The checkpoint is `SFEN001`, 848 MB, and cannot be rebuilt without 56 hours of
GPU time, so it belongs somewhere durable:

- Hugging Face model repo (`Edipcm-hash/stonefish-222m`), file
  `stonefish_en.final.state`, plus a copy of `docs/MODEL-CARD.md` as the card.
- Mention the tokeniser: `data/en/tiny.bpe` (126 KB) must ship with the weights,
  because the checkpoint alone cannot decode.

Do not put the weights in git. `.gitignore` already blocks `*.state`.

## 5. Repository settings

- **About**: description from `Cargo.toml`, topics `rust llm transformer autograd
  cuda from-scratch zero-dependencies gpt bpe machine-learning`.
- **Pin** the repository on the profile, otherwise a fresh account shows nothing.
- **Enable Issues** — a from-scratch project will get "how do I…" questions, and
  that is a good problem.
- **Social preview**: `docs/architecture.png` works at that aspect ratio.

## 6. Announcement

One paragraph, no superlatives, the numbers that can be verified:

> I wrote a language model from scratch in Rust — autograd, BPE, transformer,
> CUDA kernels — with zero dependencies. 222M parameters trained on 2.05B tokens
> on one RTX 5070. 74 tests, 32 of them numerical gradient checks. It is 1.8×
> faster than PyTorch on the same GPU at the same architecture, and it is **not**
> a chat assistant: it continues text.

Then the honest part, because it is what makes the post credible: what it cannot
do, and what the next step (bf16) would buy.

## 7. After the release

- [ ] README's status line updated from "training" to the measured numbers
- [ ] `Cargo.toml` version already 1.0.0
- [ ] Tag matches the commit that was tested
- [ ] Hugging Face weights downloadable and the model card renders
- [ ] The verify-it-yourself table still passes on a fresh clone
