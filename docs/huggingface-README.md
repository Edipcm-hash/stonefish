---
license: mit
language:
  - en
tags:
  - text-generation
  - rust
  - from-scratch
  - zero-dependencies
  - cuda
  - rmsnorm
  - swiglu
  - rope
datasets:
  - roneneldan/TinyStories
  - sedthh/gutenberg_english
library_name: stonefish_ai
pipeline_tag: text-generation
---

# StonefishAI v2 — 222M

A 222,331,904-parameter English language model trained from scratch on a corpus of
TinyStories and Project Gutenberg, by an engine with **zero dependencies**:
autograd, byte-level BPE, the transformer and the CUDA kernels are all in the
[repository](https://github.com/Edipcm-hash/stonefish), written by hand.

This repository contains the weights only. You need the inference code from the
main repository to run them — see *How to use* below.

## Model

| | |
|---|---|
| Architecture | decoder-only transformer, pre-norm |
| Parameters | 222,331,904 |
| Layers / width / heads | 16 / 1024 / 8 (d_head 128) |
| Context | 512 tokens |
| Vocabulary | 16,384 (byte-level BPE) |
| Normalisation | RMSNorm, no bias, eps 1e-6 |
| Activation | SwiGLU, hidden 2816 |
| Position | RoPE, theta 10,000 |
| Tying | output head reuses the token embedding |
| Precision | fp32 |
| Checkpoint | `stonefish_en.final.state` (`SFEN001` format, 848 MB) |

## Training

| | |
|---|---|
| Data | TinyStories v2 + Project Gutenberg (English) |
| Tokens | 2,047,457,313 (exactly one epoch) |
| Steps | 62,483 |
| Batch | 4 sequences x gradient accumulation 16 = 32,768 tokens/step |
| Optimiser | Adam (0.9 / 0.999, eps 1e-6) |
| Schedule | 4e-4 peak, 1,000-step warmup, cosine decay, clip 1.0 |
| Dropout | 0.1 |
| Hardware | one RTX 5070 (12 GB), 250 W, ~56 hours |
| Throughput | ~10,700 tokens/s |

That is 9.2 tokens per parameter, less than half of what scaling studies
recommend — this model is deliberately *compute-limited*, not data-limited.

## How to use

The weights are useless without the tokenizer: `tiny.bpe` (126 KB) must sit next
to the checkpoint.

```bash
git clone https://github.com/Edipcm-hash/stonefish
cd stonefish
cargo build --release            # no CUDA? add --no-default-features

# put stonefish_en.final.state and data/en/tiny.bpe in place, then:
cargo run --release -- en-gen "Once upon a time" 150 0.8
cargo run --release -- en-eval 8 data/en/tiny stonefish_en.final.state
```

With CUDA the generation path uses the KV-cache on the GPU (~100+ tokens/s).
Without it, the CPU path loads the same checkpoint into the CPU model and runs
through the autograd engine at roughly 3 tokens/s.

## Intended use and limits

**Use it for:** text continuation in English, and as a reference implementation
of a language model with no framework underneath.

**Do not use it for:** questions and answers, chat, facts, code generation,
translation, or anything with consequences. It was never fine-tuned for
instruction following, and it states invented facts fluently.

It is also **not** safety-tuned: a hostile prompt can elicit biased or unpleasant
continuations, and the training data (children's stories plus a mostly
19th-century Western canon) carries the blind spots of its sources.

## Evaluation

See the main repository's README, section 7, for the measured validation loss,
perplexity, bits-per-character and generation samples — with the exact command
that produced each number. The numbers are not duplicated here on purpose: two
copies drift.

## Licence

MIT for the code and these weights. The training corpora keep their own licences
(TinyStories: CDLA-Sharing-1.0; Gutenberg: public domain in the US), which is why
no data is redistributed.
