# 🐟 StonefishAI

[![CI](https://github.com/Edipcm-hash/stonefish/actions/workflows/ci.yml/badge.svg)](https://github.com/Edipcm-hash/stonefish/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Dependencies: 0](https://img.shields.io/badge/dependencies-0-brightgreen.svg)](#14-how-zero-dependencies-works)

**A from-scratch autograd engine and language-model stack in Rust — ZERO dependencies.**

**222 M parameters · 2.05 billion tokens · 75 tests · 0 dependencies** — on one
RTX 5070, with no ML library underneath.

> **What this is:** a language-model engine written from scratch - autograd, tokenizer,
> transformer, CUDA kernels - that trains and runs on one consumer GPU with zero
> dependencies.
>
> **What this is not:** a chat assistant. The 222M model continues English text; it does
> not answer questions, and it states invented facts with confidence.

```console
$ cargo tree
stonefish_ai v1.0.0
```

Everything from `Tensor` and topological backprop to optimizers and CUDA kernels is written by hand in this repository. **`[dependencies]` is empty.** The CUDA driver, cuBLAS and NVRTC are loaded dynamically at runtime (`LoadLibraryW` / `dlopen`) — not even `cudarc`, only the operating system's library loader.

On machines without CUDA, `--no-default-features` runs everything on the CPU.

> Turkish version: [README-Tr.md](README-Tr.md)

> **Status:** v2 is **trained and measured**: 222M parameters, exactly one epoch =
> 2.05B tokens. val loss **2.0227** · perplexity **7.56** · BPC **0.6064**
> bits/character (random baseline 2.909) · **10,772 tok/s** on one RTX 5070 ·
> 75 tests passing. Details in section 7.4.
>
> **Weights:** [huggingface.co/StonefishAI/stonefish-222m](https://huggingface.co/StonefishAI/stonefish-222m)
> — the 848 MB checkpoint plus the 126 KB vocabulary. Without the vocabulary the
> checkpoint cannot decode.

---

## Contents

1. [What it can / cannot do](#1-what-it-can--cannot-do)
2. [Quick start](#2-quick-start)
3. [Hardware and measured limits](#3-hardware-and-measured-limits)
4. [Architecture](#4-architecture)
5. [Design decisions](#5-design-decisions)
6. [The two model pipelines](#6-the-two-model-pipelines)
7. [Measured results](#7-measured-results)
8. [Time-budget profile](#8-time-budget-profile)
9. [Tests](#9-tests)
10. [Bugs found during development](#10-bugs-found-during-development)
11. [Known limitations](#11-known-limitations)
12. [Roadmap - what is done, what is left](#12-roadmap---what-is-done-what-is-left)
13. [Directory layout](#13-directory-layout)
14. [How zero dependencies works](#14-how-zero-dependencies-works)
15. [**Timeline — from 15 June to today**](#15-timeline--from-15-june-to-today)

---

## 1. What it can / cannot do

### ✅ What it can do (all measured)

| Capability | Measure |
|---|---|
| Handwritten digit classification (MNIST) | **98.14 %** test accuracy (`train 20 0.5`) |
| English language model (v2, released) | **222.3 M parameters**, exactly 1 epoch = **2.05 B tokens** |
| Quality | held-out **BPC 0.6064** (random baseline 2.909) · perplexity **7.56** |
| English generation | 512-token context, temperature/top-k, **KV-cache** so generation is O(n) |
| Turkish character model (v0.5, historical) | 842 K parameters, val CE **1.6230** (baseline 4.91) |
| Gradient verification | **32 tests**, analytic <-> numerical by central difference |
| GPU training | device-resident; **10,772 tok/s** at 222 M |
| Checkpoints | config+vocab embedded, **atomic** writes, resume |

### ❌ What it cannot do (the honest list)

- **Not a chat assistant.** No instruction following, Q&A, translation, summarisation or code generation.
- **No world knowledge.** "What is the capital of France?" gets prose, not an answer.
- **No reasoning.** No arithmetic, no logical chains, no multi-step inference.
- **It confabulates.** The output is grammatical and factually unreliable: it writes invented facts fluently.
- **Small and undertrained.** 222 M parameters on 2.05 B tokens is 9.2 tokens per parameter, less than half of what scaling work recommends.
- **English only.** The Turkish model was a character-level toy and is not part of this release.
- **512-token context.** Beyond that the window slides; the model never saw longer.
- **Slow on a CPU.** `en-gen` runs without a GPU, at about 3 tokens/s; the GPU path
  reaches **160-185 tokens/s** (measured).
- **Not meaningfully faster than PyTorch.** 2.6 % ahead at 222 M, 1.3 GB worse on
  memory. The gain is not speed; see 7.5.

### Why it exists

I could have defined this model in about ten lines of PyTorch. I did not, because the
question this project asks is not "train a language model" but **"understand the
layers that make one run"**:

- How does `backward()` actually work, and where do the gradients come from?
- What does writing a CUDA kernel look like without a wrapper like `cudarc`?
- Which decisions get 222M parameters into a 12 GB card?
- Is the number right, or does it merely run?

None of that is learned by using a library. The whole project is the answer to those
four questions. The model is the by-product.

In one line: **it was built to take the pinball machine apart, not to play it.** But
the parts turned out to be a machine that plays: 2 billion tokens of training, 75
tests, and a loss that matches PyTorch to 1.8e-05.

### Where to start reading

10,349 lines is intimidating; in the right order, five files are enough.

| # | File | What it teaches |
|---|---|---|
| 1 | `src/tensor.rs` | a plain Rust 2D matrix - matmul, transpose, slicing |
| 2 | `src/autograd.rs` | the `Op` enum and topological `backward()`; gradients for 19 ops |
| 3 | `src/transformer.rs` | RMSNorm, SwiGLU, RoPE, attention, GPT |
| 4 | `src/gpu_train.rs` | the device-resident trainer: hand-written forward and backward |
| 5 | `src/sys/` | our own CUDA FFI - from `LoadLibraryW` to `cublasSgemm` |

All of it is cross-checked by tests; `tests/gradcheck.rs` recomputes every op's
gradient by central difference. If you get stuck reading the code, that test probably
has the answer.

---

## 2. Quick start

```bash
# Build (on a machine without CUDA: --no-default-features)
cargo build --release

# List every mode
./target/release/stonefish_ai help

# ⚡ ONE-COMMAND DEMO - trains from scratch and generates in 90 s, no downloads
./target/release/stonefish_ai demo 90

# --- MNIST MLP ---
./target/release/stonefish_ai train 20 0.5   # epochs, lr -> 98.14 %
./target/release/stonefish_ai eval           # measure on the test set

# --- Turkish GPT (character level, historical) ---
./target/release/stonefish_ai gpt-train 6 8                       # epochs, batch
./target/release/stonefish_ai gpt-gen "Aksham erken iner" 300 0.8 # prompt, tokens, temp

# --- English GPT (BPE-16k, 222 M) ---
./target/release/stonefish_ai en-prepare             # learn BPE + build the u16 cache
./target/release/stonefish_ai en-train 2047457313    # token budget (~54 h, one GPU)
./target/release/stonefish_ai en-eval 8 data/en/tiny # loss / perplexity / BPC
./target/release/stonefish_ai en-gen "Once upon a time" 150 0.8

# --- Measurement ---
./target/release/stonefish_ai test-gpt-profile  # where the time goes
./target/release/stonefish_ai test-gpu-native   # GPU-native trainer speed
./target/release/stonefish_ai test-matmul-bench # CPU vs GPU at real shapes
./target/release/stonefish_ai test-gpu-bench-big
```

### Data requirements

| File | What | Where |
|---|---|---|
| `data/train-images-idx3-ubyte` + 3 siblings | MNIST | [yann.lecun.com/exdb/mnist](http://yann.lecun.com/exdb/mnist/) |
| `data/turkish.txt` | Turkish corpus (plain text) | your own text |
| `data/en/train.txt` | English corpus: **TinyStories v2 + Gutenberg** (~4.1 GB) | the two commands below, appended |
| `data/en/valid.txt` | English validation (TinyStories valid only, ~18 MB) | TinyStories |

> **Careful:** the released v2 model was trained on **both corpora together**
> (exactly one epoch = 2.05B tokens). Training on TinyStories alone gives a
> different, smaller model with a higher validation loss. To reproduce section 7's
> numbers exactly, run both steps.

`data/`, `*.bin`, `*.state`, `*.bpe` are **not committed** (`.gitignore`).

TinyStories:
```powershell
$b = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
Invoke-WebRequest "$b/TinyStoriesV2-GPT4-train.txt" -OutFile data/en/train.txt
Invoke-WebRequest "$b/TinyStories-valid.txt"        -OutFile data/en/valid.txt
```

Gutenberg (parquet shards on the Hub; the repository has the converter):
```powershell
# 1. download the shards (sedthh/gutenberg_english)
huggingface-cli download sedthh/gutenberg_english --repo-type dataset --local-dir data/gb
# 2. parquet -> plain text (needs pyarrow; data prep, not a project dependency)
python tools/parquet_to_text.py          # -> data/gb/books.txt
# 3. append to the corpus
Get-Content data/gb/books.txt | Add-Content data/en/train.txt
```

### 2.1 Verify every claim yourself

> Every number in this document can be reproduced: [docs/REPRODUCING.md](docs/REPRODUCING.md). Model details: [docs/MODEL-CARD.md](docs/MODEL-CARD.md). Release procedure: [docs/RELEASE.md](docs/RELEASE.md).


No number in this README asks you to trust us. The commands below reproduce
each claim on your own machine.

| Claim | Command | Expected |
|---|---|---|
| No dependencies | `cargo tree` | one line: `stonefish_ai v1.0.0` |
| 75 tests pass | `cargo test --release` | `75 passed; 0 failed` |
| Gradients are correct | `cargo test --release --test gradcheck` | `32 passed` |
| CUDA kernels match the CPU | `cargo test --release --test gpu_kernels` | `16 passed` |
| GPU trainer matches CPU autograd | `cargo test --release --test gpu_train_equiv` | `2 passed` |
| KV cache matches the full forward pass | `cargo test --release --test kv_cache` | `2 passed` |
| Checkpoints survive corruption attempts | `cargo test --release --test checkpoint` | `5 passed` |
| From-scratch training works | `cargo run --release -- demo 90` | text in 90 s |
| Our own FFI works | `cargo run --release -- test-ffi` | every diff is `0.0` |
| The profile table is real | `cargo run --release -- test-gpt-profile` | a table is printed (on a CUDA build the GEMM and transfer buckets are non-zero) |
| The model agrees with PyTorch | `python tools/torch_parity.py --state stonefish_en.final.state --valid data/en/tiny.valid.bin --batches 8 --expected <value from en-eval>` | `MATCH` (< 2e-3) |
| No Turkish text | `python tools/check_no_turkish.py` | `OK` |
| `en-gen` works without a GPU | `cargo run --release --no-default-features -- en-gen "Once upon a time" 10 0` | text comes out (~3 tok/s) |
| The charts come from this data | `python tools/make_charts.py` | `docs/*.png` rewritten |

**Measured (idle RTX 5070):** 18,907 steps in 99 s · **389,304 tok/s** · final loss
**0.71** · 1,783,680 parameters · output: *"I thank you, gentle lady; and so indeed it
with his new doublet in true arms?"*

**Honest warning - this is an engine demo, not a model claim.** 18,907 steps x 2,048
tokens = 38.7 M tokens, which is **~54 epochs** of the 716,800-character Shakespeare
slice. The model is fluent here because it largely **memorises** the corpus. That the
text looks like Shakespeare shows the engine works; it does not show that the model
generalises. The real model (section 7.4) was trained on 2 billion tokens, seeing each
piece of text once.

**Note:** it also runs on the CPU (a 12K-character slice, ~11,000 tok/s). The budget
assumes an **idle card**: sharing it with another training run drops the rate from
389,000 to ~12,000 tok/s and the text gets rough.
`test-ffi` and the GPU tests need the CUDA build; `--no-default-features` skips them. With
`--no-default-features` everything runs on the CPU and the GPU tests are
skipped.

---

## 3. Hardware and measured limits

Development/test machine: **RTX 5070 (12 GB), 32 GB DDR5, MSI Trident AS (~10 L SFF)**, Windows.

| Limit | Value | Consequence |
|---|---|---|
| VRAM | 12 GB | **~250M parameter** ceiling in fp32 (v2: 222M, 8.4 GB) |
| RAM | 32 GB | 2B tokens **do not fit** without the streaming pipeline (u16 + disk cache solves it) |
| Thermals | SFF chassis, VBIOS cap 250 W | 76-80 C at 250 W, no throttling. A 210 W limit was tried as an experiment (76 C, 5 % slower); the driver reset it and it stayed at 250 W |
| Time | single GPU | **222M model + 2.05B tokens = 56 hours** (measured) |

Measured training speeds:

| Model | Step time | Throughput |
|---|---|---|
| 842K (GPU-native) | 4.2 ms | **244,698 tok/s** |
| 842K (CPU+cuBLAS) | 120.8 ms | 8,476 tok/s |
| 236M, v1 (GPU-native) | ~3.27 s | ~10,000 tok/s |
| **222M, v2 (GPU-native)** | **3.04 s** | **10,772 tok/s** |

The 222M row is the mean of the last 200 samples over 62,483 steps; the whole run
averaged 10,558 tok/s (the first hours carried the data-window bug and my own
compile load).

---

## 4. Architecture

![Architecture - one forward pass](docs/architecture.png)

| File | Lines | Responsibility |
|---|---|---|
| [src/lib.rs](src/lib.rs) | 46 | Module root, `VERSION`, the clippy allows with reasons |
| [src/tensor.rs](src/tensor.rs) | 182 | Matrix + `Tensor` (Rc<RefCell<..>>), the xorshift RNG |
| [src/autograd.rs](src/autograd.rs) | 1106 | 19 ops, topological `backward()`, fused causal attention |
| [src/nn.rs](src/nn.rs) | 117 | `Linear`, save/load for the MNIST model |
| [src/optim.rs](src/optim.rs) | 196 | SGD, Adam with bias correction, `clip_grad_norm` |
| [src/loss.rs](src/loss.rs) | 9 | Softmax + cross-entropy |
| [src/data.rs](src/data.rs) | 89 | MNIST idx reader, text to sequences |
| [src/tokenizer.rs](src/tokenizer.rs) | 150 | Character-level tokenizer |
| [src/bpe.rs](src/bpe.rs) | 313 | Byte-level BPE: merge learning, encode/decode |
| [src/corpus.rs](src/corpus.rs) | 306 | u16 token cache, streaming window reader |
| [src/transformer.rs](src/transformer.rs) | 290 | v2 GPT reference on the CPU (RMSNorm/SwiGLU/RoPE) |
| [src/checkpoint.rs](src/checkpoint.rs) | 400 | GPT checkpoint v2 (magic/version/config/vocab/tensors) |
| [src/trainer.rs](src/trainer.rs) | 219 | One training interface over the CPU and device trainers |
| [src/gpu.rs](src/gpu.rs) | 326 | cuBLAS GEMM (plain/transposed/device/strided-batched) |
| [src/gpu_kernels.rs](src/gpu_kernels.rs) | 603 | NVRTC compile, PTX cache, kernel wrappers |
| [src/gpu_train.rs](src/gpu_train.rs) | 1467 | Device-resident GPT trainer (the fast path) |
| [src/en_train.rs](src/en_train.rs) | 596 | English pipeline: `prepare`, `train`, `eval`, `generate` |
| [src/kv_cache.rs](src/kv_cache.rs) | 363 | KV-cache generation (O(n) until the context fills) |
| [src/demo.rs](src/demo.rs) | 224 | One-command demo, CPU or GPU |
| [src/probe.rs](src/probe.rs) | 60 | Tokenize unseen text so `en-eval` can score it |
| [src/profile.rs](src/profile.rs) | 133 | Lightweight profiling counters |
| [src/main.rs](src/main.rs) | 1739 | CLI (27 modes), the training loops, the smoke tests |
| [src/sys/mod.rs](src/sys/mod.rs) | 202 | Dynamic loader (LoadLibraryW / dlopen) |
| [src/sys/cuda.rs](src/sys/cuda.rs) | 202 | CUDA driver API |
| [src/sys/cublas.rs](src/sys/cublas.rs) | 85 | cuBLAS bindings |
| [src/sys/nvrtc.rs](src/sys/nvrtc.rs) | 124 | Runtime compilation |
| [src/sys/dev.rs](src/sys/dev.rs) | 564 | Buffers, views, kernel arguments, GEMM |
| [src/sys/selftest.rs](src/sys/selftest.rs) | 238 | `test-ffi`: alloc, copy, PTX, kernel, cuBLAS |
| [kernels/stonefish.cu](kernels/stonefish.cu) | 711 | CUDA kernels |

Measured total: **10,349 lines of Rust** (`src/`) + 2,322 lines of tests + 150 of examples + **711 lines of CUDA**, **0 dependencies**.

---

## 5. Design decisions

### 5.1 Batching by row packing (no Tensor3D)

A `(batch, seq, d_model)` tensor is stored as a **`(batch·seq, d_model)` matrix**:

```
(B, S, D) ──pack──▶ (B·S, D)
```

RMSNorm, Linear, SwiGLU, embedding and softmax are already **row-wise**, so this transformation is free. The only exception is attention, handled inside `Op::CausalAttention` with `batch`/`seq`/`n_heads` metadata.

**Result:** one engine instead of two parallel APIs, real batch gradients, a single `backward()` call. The old `Tensor3D` type was removed entirely.

### 5.2 Fused causal multi-head attention

`Tensor::causal_attention(q, k, v, batch, seq, n_heads)` is **a single node**:

- splits `d_model` into `n_heads` slices (`d_head = d_model/n_heads`)
- `1/√d_head` scaling
- causal mask
- per-head softmax
- **the backward for all of it**

Attention weights are cached as `(batch·n_heads·seq, seq)`; dQ/dK/dV are produced in one pass. Because scaling and masking live *inside* the op, they cannot be mis-composed outside.

### 5.3 Row-major GEMM → cuBLAS (transpose trick)

cuBLAS is column-major. For row-major `C(m,n) = A·B`:

```
Cᵀ = Bᵀ · Aᵀ   →  operands swap, expressed via transa/transb
```

`dev_gemm_batched` performs this mapping for `Nn/Nt/Tn`; `lda` = **B's row stride**, `ldb` = **A's row stride**, `ldc` = C's row stride. Transposes are therefore **never materialized** (the old code took a `(1024×128)` strided copy on every backward).

### 5.4 CUDA kernels: NVRTC instead of nvcc

This machine has no host C++ compiler (`cl.exe`), so `nvcc` fails. **NVRTC** only compiles device code, so it works:

- runtime compilation (~100-300 ms, once)
- PTX cached under `target/gpu-cache/` (key: source + arch + version)
- arch read from the device (`compute_120` → RTX 5070)

### 5.5 Two trainers, one interface

| | CPU + cuBLAS | GPU-native |
|---|---|---|
| What | autograd graph | hand-written forward/backward |
| Where | `autograd.rs` + `trainer.rs` | `gpu_train.rs` |
| Role | **reference** (gradient tests) | tied to CPU via equivalence tests |
| Speed | 8,476 tok/s | **244,698 tok/s** |

`gpt-train` automatically selects the GPU path when CUDA is present, otherwise falls back to CPU.

### 5.6 Tokenizer: characters first, BPE later

A character-level model spends most of its capacity learning **spelling**. With byte-level BPE-16k the same window covers **4-5× more text** (measured: 4.7 characters/token).

BPE learning uses **incremental pair counting**: after each merge only words containing that pair are rescanned → 16k merges in **7 seconds** (the naive approach took hours).

### 5.7 Data pipeline: u16 + disk cache

`Vec<usize>` = 8 bytes per token → 2 B tokens would need **16 GB of RAM**. Instead:

- tokenized corpus is written to disk as **u16** (2 bytes/token)
- training reads through a **window** (`len*8`, clamped to 4 KB - 1 MB); the whole file is never loaded
- tokenization: **114-148 MB/s** (word-cached; depends on how repetitive the
  corpus is - measured with `cargo run --release --example bpe_bench`: 114 MB/s
  on TinyStories, 148 MB/s on a more repetitive text)

---

## 6. The two model pipelines

### 6.1 Turkish (character level)

| | |
|---|---|
| Model | `d_model=128`, 4 layers, 4 heads (`d_head=32`), ctx=128 |
| Vocab | 136 characters |
| Parameters | **842,632** |
| Corpus | `data/turkish.txt` — 2,870,158 characters, 22,423 sequences |
| Checkpoint | `stonefish_gpt.bin` (v2: magic + config + vocab + tensors) |
| Training | 6 epochs, 16,620 steps, batch 8 |

Training curve (GPU-native, 70 seconds):

| Epoch | train CE | val CE |
|---|---|---|
| 0 | 2.1881 | 1.8943 |
| 1 | 1.7973 | 1.7624 |
| 2 | 1.6868 | 1.6939 |
| 3 | 1.6145 | 1.6538 |
| 4 | 1.5591 | 1.6309 |
| 5 | **1.5215** | **1.6230** |

Reproducibility: hyperparameters + seeds + per-epoch metrics are written to a
`runs/<ts>.json` manifest. The two manifests in the repository confirm it: `run`
holds `model_seed: 0xDEADBEEF`, `shuffle_seed: 0xA5A51234`, `wall_seconds: 69.6`;
`epochs` holds each epoch's train/val CE; and `best_val_ce: 1.622985` is the last
row of the table above.

### 6.2 English (BPE-16k)

Two English models exist: **v1** (the previous version, measured but never published)
and **v2** (the model released here). Both were measured on the same validation file,
so they are directly comparable.

| | **v1 - previous, never published** | **v2 - released** |
|---|---|---|
| Architecture | LayerNorm + GELU + absolute positions | **RMSNorm + SwiGLU + RoPE + weight tying + dropout** |
| Parameters | 235,571,200 | **222,331,904** |
| Corpus | TinyStories, 450.9 M tokens | **TinyStories + Gutenberg, 2,047,457,313 tokens** |
| Tokens/parameter | 1.9 | **9.2** |
| Training | 13,732 steps, 745 min | 62,483 steps, ~54 hours |
| **BPC** | 0.6913 | **0.6064** |
| Status | completed, measured - **never published** | **completed, measured, ready to publish** |

v2 gets a better BPC **with 6 % fewer parameters**: the architecture round (RoPE,
RMSNorm, SwiGLU, tying) and 4.5x more data, together.

#### v1 (previous version, never published) - measured

| | |
|---|---|
| Model | `d_model=1024`, 16 layers, 8 heads (`d_head=128`), ctx=512 |
| Vocabulary | **16,384** (byte-level BPE) |
| Checkpoint | `stonefish_en.state` (`SFEN001`, every 500 steps) |
| VRAM | 8.4 GB / 12 GB · ~10,000 tok/s |
| Training | 13,732 steps, 745 minutes, 10,061 tok/s average |

**Measured quality (`valid.bin`, 3,928,064 tokens):**

| Metric | Value |
|---|---|
| val loss | **2.3700** nats/token |
| perplexity | **10.70** |
| **BPC** | **0.6913 bits/character** |
| final train loss | ~1.95 (gap 0.42 -> mild overfitting) |
| random baseline | 9.70 |

**Sample generation (temperature 0, greedy):**

> Once upon a time-out lived a little girl named Lily. She loved to play with her toys and make up stories. One day, she found a big box in her room. She was very excited and wanted to see what was inside.
> Lily opened the box and found a lot of fun toys inside. She played with the toys and had a lot of fun. But then, she heard a loud noise outside. She looked out the window and saw a big, scary monster! The monster was very big and scary. Lily was scared and ran back to her room.

**Dialogue formatting (temperature 0.8):**

> "Yes! I love to play with my car," Tommy said.
> "But I need your toy car," Jane said.

The model learned the `<|endoftext|>` story boundary and quoted dialogue formatting on its own.

---

## 7. Measured results

> **Measurement conditions.** Every number below was taken on an **idle card**:
> RTX 5070 (12 GB), driver 621.24, fp32, TF32 off. Sharing the card with another
> CUDA process corrupts the measurement - the same matmul reads 2.99 ms instead of
> 0.19 ms, and the GPU appears *slower* than the CPU. The command for each row is
> in `docs/REPRODUCING.md`.

### 7.1 MNIST MLP (784→128→10)

```bash
cargo run --release -- train 20 0.5
```

SGD lr=0.5, batch 64, 20 epochs → **98.14 %** test accuracy (10,000 images).

> **Why lr=0.5, and why the README used to say 0.01:** `SoftmaxCE`'s backward
> once summed over the batch; when the `1/batch` fix landed, the gradient became
> 64x smaller. With the same 20 epochs, lr=0.01 now stalls at **94.51 %**
> (measured) while lr=0.5 reproduces the old result (98.14 %). The earlier
> 98.12 % was measured *before* that fix - both numbers are correct, they belong
> to different revisions.

### 7.2 GPU vs CPU matmul (RTX 5070, after warm-up)

| GEMM | CPU | GPU | Speed-up |
|---|---|---|---|
| `(1024×128)@(128×128)` attn proj | 1.119 ms | 0.189 ms | **5.9×** |
| `(1024×128)@(128×512)` ff1 | 5.697 ms | 0.975 ms | **5.8×** |
| `(1024×512)@(512×128)` ff2 | 5.310 ms | 0.284 ms | **18.7×** |
| `(1024×128)@(128×136)` lm head | 1.150 ms | 0.210 ms | **5.5×** |
| `(128×128)@(128×128)` inference (B=1) | 0.132 ms | 0.055 ms | **2.4×** |

1024³: CPU 2.16 s / GPU-only **67.5 µs** / transfer+matmul 2.23 ms.

### 7.3 End-to-end training (6 epochs, same model)

| Path | Step time | Throughput | 6 epochs |
|---|---|---|---|
| CPU + cuBLAS (autograd) | 120.1 ms | 8,476 tok/s | **54.4 min** |
| **GPU-native (device-resident)** | **4.2 ms** | **244,698 tok/s** | **70 s** |

Two ratios, both correct, measuring different things:

- **28.6x per step** (120.1 ms -> 4.2 ms; 8,476 -> 244,698 tok/s).
- **46.6x wall clock** (54.4 min -> 70 s), because on the CPU path every
  epoch's validation also runs on the CPU, while the GPU path validates on
  the device too.

Val CE: CPU 1.6337 / GPU 1.6230 (fast-math + summation order).

### 7.4 English v2 (the released model)

| | |
|---|---|
| Model | d_model=1024, 16 layers, 8 heads (d_head=128), ctx=512 |
| Parameters | **222,331,904** |
| Vocabulary | 16,384 (byte-level BPE, 16,128 merges) |
| Corpus | TinyStories v2 + Gutenberg, exactly 1 epoch = **2,047,457,313 tokens** |
| Training | 62,483 steps, ~49 hours, one RTX 5070 (250 W) |
| Throughput | last 200 steps: **10,772 tok/s** |
| VRAM | 8.4 GB / 12 GB · 66 °C · GPU 99 % |

**Weights:** [StonefishAI/stonefish-222m](https://huggingface.co/StonefishAI/stonefish-222m)
(`stonefish_en.final.state` + `tiny.bpe`). Once downloaded:

```bash
cargo run --release -- en-gen "Once upon a time" 150 0.8
cargo run --release -- en-eval 8 data/en/tiny stonefish_en.final.state
```

**Quality** (valid.bin, 8 batches = 32,768 tokens):

| Metric | Value | Random baseline |
|---|---|---|
| val loss | **2.0227** nat/token | 9.7041 (ln 16,384) |
| perplexity | **7.56** | 16,384 |
| **BPC** | **0.6064** bits/character | **2.909** |

**Out of distribution** (this README's English text, never seen in training, 20,514 tokens):

| Metric | Value | Its own baseline |
|---|---|---|
| val loss | **4.5849** nat/token | 9.7041 |
| perplexity | **98.00** | 16,384 |
| **BPC** | **3.2914** bits/character | **6.966** |

> **Why is val loss lower than train loss?** Two reasons: (1) about three
> quarters of the training mix is Gutenberg while the validation set is
> TinyStories only - the model is markedly better at children's stories than at
> books (measured: 1.95 vs 2.90 on the first batches). (2) Dropout is on during
> training and off during evaluation. The weighted average of the mix is 2.68;
> the ~3.10 in the log is that with dropout.
>
> **The probe's BPC is not comparable with 2.909:** BPC's denominator is the
> corpus's chars-per-token ratio (TinyStories 4.812, probe 2.010). Each corpus is
> shown against its own baseline.

**Generation** (temperature 0, greedy, no filter):

> Once upon a timetime, there was a little girl named Lily. She loved to play with her toys and eat yummy food. One day, she found a big box in her room. She was very excited and wanted to see what was inside.
> Lily opened the box and found a big, soft teddy bear. She was so happy and hugged her teddy bear. She said, "Thank you, teddy bear, for being so kind to me." The teddy bear smiled and said, "You're welcome, Lily. I am glad you like it."
> From that day on, Lily and the teddy bear were best friends. They played together every day and had lots of fun. And they lived happily ever after.
> <|endoftext|>
> Once upon a time, there was a little girl named Lily. She had a big box of toys. One day, she found a new toy in her box. It was a small, red car.

**Generation** (temperature 0.8, top-k 40):

> Once upon a time? The family was very sad. They needed help to find a way to get home and stay cool. They looked around and finally found a hidden door. The family was very excited to see what was behind the door. They opened the door and saw a bright light and a bright light.
> The family was so happy to be home and they thanked the family for being so kind. They continued to walk home and had a wonderful time together.
> <|endoftext|>
> One day, a little boy named Tim went to the museum with his mom. They saw a big box with a picture of a big, scary dog. Tim saw a man with a funny hat and a funny hat. He asked his mom, "Can we buy the dog?"
> His mom said, "Yes, let's buy the dog." So, they bought the dog and brought it home. Tim

Seven samples - the good and the bad, unfiltered: [docs/SAMPLES.md](docs/SAMPLES.md)

**PyTorch parity:** same weights, same 8 batches (32,768 tokens) →
StonefishAI **2.022700** · PyTorch **2.022718** · delta **1.8e-05** (threshold 2e-3).
Two independent implementations agree to f32 rounding.

### 7.5 Against PyTorch - same GPU, same architecture

![StonefishAI vs PyTorch - 222 M parameters](docs/comparison.png)

`tools/torch_bench.py` rebuilds the v2 architecture (RMSNorm + SwiGLU + RoPE +
weight tying) exactly in PyTorch; same batch (4 x gradient accumulation 16), same
GPU, same warm-up.

**At 222 M parameters** (the released model):

| Implementation | tok/s | ms/step | VRAM |
|---|---|---|---|
| **StonefishAI** | **10,772** | 3,042 | 8.40 GB |
| PyTorch (`F.scaled_dot_product_attention`) | 10,502 | 3,120 | 7.10 GB |
| PyTorch (naive attention) | 9,996 | 3,278 | 7.60 GB |

At this size the gap is **2.6 %** over PyTorch's SDPA. Honestly, that is far below
the small-model figure: at 222 M both sides are bound by the same GEMMs, so
hand-written kernels stop paying off. **On memory we are worse** - 8.40 GB against
7.10 GB - because our activations are reused less aggressively.

At 35.7 M parameters the same comparison was **1.79x** (20,487 vs 10,810 tok/s):
there the bottleneck was per-kernel memory traffic rather than GEMM efficiency,
which is why the advantage is large on small models and small on large ones.

![The same measurement at 35.7 M parameters](docs/speedup.png)

---

## 8. Time-budget profile

> **Historical measurement.** This is the profile from the day we decided to
> write the GPU-native trainer: the model was still **v1** (GELU + LayerNorm)
> and training ran on the CPU. v2 replaced GELU with SwiGLU and LayerNorm with
> RMSNorm, so `test-gpt-profile` prints different rows today. The table stays
> because these numbers are the *reason* the decision was made.

`test-gpt-profile` (842 K model, real config, 10-step average):

| Item | ms/step | share |
|---|---|---|
| attention fwd+bwd (CPU) | 35.3 | 29.5 % |
| matmul **transfer** (CPU↔GPU) | 21.1 | 17.6 % |
| matmul **real GEMM** | **1.0** | **0.8 %** |
| GELU fwd+bwd | 17.7 | 14.8 % |
| LayerNorm fwd+bwd | 7.4 | 6.1 % |
| intermediate tensor allocation | 6.0 | 5.0 % |
| adam.step / CE / zero_grad | 3.4 | 2.9 % |
| other (alloc, packing, residuals) | ~27 | ~23 % |
| **total** | **~120** | 100 % |

**Key insight:** a step issues 75 matmul calls but the real GEMM total is 1 ms. So "keep GEMMs on the GPU" (eliminating transfers) buys at most 17 %; the real win comes from **moving the computation itself to the GPU**. The GPU-native path does exactly that: 28.6x per step, 46.6x wall clock (7.3).

---

## 9. Tests

**75 tests** in the CUDA build, **55** in the CPU-only build. The other 20 are
kernel comparisons and the device trainer, so they need CUDA (on a machine
without a GPU, `cargo test --no-default-features` runs 55).

| Group | Count | What it proves |
|---|---|---|
| [tests/gradcheck.rs](tests/gradcheck.rs) | 789 | analytic <-> numerical gradient by **central difference**, for every op |
| [tests/gpu_kernels.rs](tests/gpu_kernels.rs) | 1109 | 33 CUDA kernels and the device GEMM modes, against the CPU reference |
| [tests/checkpoint.rs](tests/checkpoint.rs) | 155 | round-trip, bad magic, truncated file, vocab mismatch, drift |
| [tests/gpu_train_equiv.rs](tests/gpu_train_equiv.rs) | 159 | GPU trainer <-> CPU autograd: **every gradient** plus the loss curve of 5 Adam steps |
| [tests/kv_cache.rs](tests/kv_cache.rs) | 110 | cached generation <-> full forward pass |
| Unit tests (`src/`) | 18 | tokenizer (4), optimizer/clip (4), BPE (4), corpus (2), LR schedule (3), SFEN order (1) |

Gradient coverage: `add`/`sub` (including broadcast), `mul_elem`, `matmul`
(including square), `matmul+transpose`, `scale`, `relu`, `sigmoid`, `transpose`,
`sum`, `softmax_ce`, `softmax_ce_ids` (sparse), `softmax`, `mask_causal`, `gelu`,
`layer_norm`, `rms_norm` (forward and the gamma gradient), `rope` (forward and
gradient), `swiglu`, `embedding`, `causal_attention` (single head, multi head and
batched), an end-to-end tiny GPT, the causality contract, and the identity that a
batched gradient equals the sum of single-sequence gradients.

```bash
cargo test --release                       # CUDA path
cargo test --release --no-default-features  # CPU path
```

---

## 10. Bugs found during development

This section is the most valuable part of the project: **every bug was real, most were caught by tests, all were fixed.** Some cost hours; one (the head stride) cost a day.

### 10.1 Silent numerical bugs - the dangerous class

These ran without crashing and produced **wrong numbers**. None of them raised an error; all were exposed by gradient checks.

| # | Symptom | Root cause | Fix | How it was caught |
|---|---|---|---|---|
| 1 | Attention gradients looked plausible but the loss would not fall | Backward used `q` instead of `dOut` for `dW` | Rewritten as `dWq = dOutᵀ·x̂` | **3 gradient tests red** after vectorisation |
| 2 | Attention scores always 0 → softmax 0.5/0.5 | Batched score buffer offset assumed `h·S²`, correct is **`h·B·S²`** | Stride corrected | `test-attn` output: every row identical |
| 3 | `dAttn/dWo` wrong from the start | Residual gradient: the attention branch's upstream is `dOut + dx_ln2`, not `dOut` | Both branches summed | 6 of 32 gradient tests red |
| 4 | GPU GEMM results accumulated between steps | `dev_gemm` passed `beta = 1.0` (C = αAB + βC) instead of 0.0 | `beta = 0.0` | `GemmMode` test: multiplying the same matrix twice gave 4× |
| 5 | `test-gpt-profile` reported **0 ms** for the two largest buckets | The FFI migration dropped `matmul_gpu`'s profiling hooks (`t_start`/`transfer_ns` gone) | Hooks restored, upload/download measured separately | The profile table did not match the README's numbers |

### 10.2 Bugs in the tests themselves - traps that send you the wrong way

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 6 | `mask_causal` gradient test permanently red | The mask was `-1e9` → loss ≈ 1e9 → `f32` ulp(1e9) ≈ 64 **swallows** `ΔL ≈ 1e-3` | Masked weights zeroed; the test now checks itself |
| 7 | Linearity broke in gradcheck | `eps` calibration: embedding init 0.02 + LayerNorm → `eps = 1e-3` is 10 % of a row | Inputs scaled ×50; `grad_probe` for parameter-wise measurement |
| 8 | Batch gradient never equalled the sum of single-sequence gradients | Wrong factor (**B³**) in the test | Derived `L_batch = (1/B)·ΣL_b`, test fixed |

### 10.3 Performance bugs - correct but slow

| # | Symptom | Root cause | Fix | Gain |
|---|---|---|---|---|
| 9 | Tokenization "hung" (minutes per 64 MB) | Every merge scanned the whole buffer → **O(n²)** | Word-based encode + word cache | **148 MB/s** |
| 10 | A training step did ~4 GB of disk I/O | The data window read 64 MB per miss | Window = `len·8` (4 KB – 1 MB) | **5,679 → 9,966 tok/s** (+75 %) |
| 11 | RoPE called `powf`/`cos`/`sin` per element | No table | cos/sin table cached | Measurable step-time drop |
| 12 | `rmsnorm_param_grad` used one thread per column | No `atomicAdd` race, but no parallelism either | Row chunking + atomic accumulation | Norm gradient ~4× |
| 13 | A training step took 177 ms | Profile: **75 matmuls per step**, real GEMM 1 ms | Removed upstream clones, added `matmul_bt/ta`, vectorised attention backward | **177 → 120 ms** (+47 %) |

### 10.4 Architecture bugs

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 14 | KV-cache generation did not match the full forward pass (**4e-2** off) | The cache path forgot the **attention scale** (1/√d_head) | Scale added → difference **5e-8** |
| 15 | The FFN was needlessly wide in tiny models | `swiglu_hidden` rounded every size up to 256 | Size-dependent step (8/64/256) |
| 16 | A checkpoint could be corrupted by a power cut | Written directly to the target file | **tmp → `sync_all` → `rename`** (atomic) |

### 10.5 Process and tooling bugs

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 17 | Two processes wrote the same cache → "Access denied" | A killed job's child process survived | Process inventory + single-writer rule |
| 18 | An **orphan training process** from 13:33 would have overwritten the checkpoint with the old (v1) format | Two trainings running at once | Process check, separate `out_prefix` |
| 19 | `cargo clippy --fix` rewrote all of `data.rs` | Line endings (CRLF→LF) | Reverted; `.gitattributes` added |
| 20 | **The tree did not build after the translation** (3 errors) | A lost comma in a `panic!` format string; the head of a `gpu.rs` hook missing; `corpus.rs`'s `Seek` import removed as "unused" | All three fixed - **caught before publishing** |
| 21 | The language guard said "OK" while **42 lines of Turkish** remained | The guard only looked for the six Turkish-specific letters; diacritic-free Turkish (the smoke-test and memory labels) slipped through | Second pass: a Turkish-only word list with no English homographs |
| 22 | `target3/` tried to enter git | `.gitignore` only had `/target/` and `target2/` | `/target*/` + `__pycache__/` |

### 10.6 Outside the project, found the same day

| # | Finding | Effect |
|---|---|---|
| 23 | The machine has **no antivirus** (Ghost Spectre image) | Known, accepted risk - documented in the README |

---

## 11. Known limitations

These are v2's real limits - the model that ships - not v1's. The earlier
version's gaps (RoPE, SwiGLU, RMSNorm, weight tying, `en-eval`, `en-gen`) are
closed in v2.

- **The KV cache resets when the window fills** - generation is O(n) until the
  512-token context is full; after that the sliding window is re-encoded per
  token, i.e. O(n·512). A ring buffer is the right fix; it is not implemented.
- **Context is 512 tokens** - longer text needs window sliding, and the model
  never saw anything beyond 512 during training.
- **fp32 training** - no bf16, which is what caps the model at ~250M on 12 GB.
- **No gradient checkpointing** - activations are kept; required for a bigger model.
- **9.2 tokens per parameter** - less than half of what scaling work recommends
  (~20). More training time or a smaller model would have been better on this
  GPU; it was a budget decision.
- **Single GPU** - no data-parallel path; the only way up is a bigger card.
- **Single-threaded CPU path** - no `rayon`; the CPU path is a reference and a
  fallback, not a production path.
- **No SFT/DPO** - the model does not behave like an assistant; it continues text.
- **English only** - the Turkish model (v0.5) was a character-level toy and is not
  part of this release.
- **Synchronous checkpoint writes** - writing 848 MB pauses the run for ~1-2 s;
  measured and accepted, every 500 steps.
- **Single-sequence generation** - no batching, no speculative decoding.
- **CPU generation is slow** - `en-gen` runs without a GPU, but through the
  autograd path at ~3 tokens/s; the GPU path is well above 100 tokens/s.

---

## 12. Roadmap - what is done, what is left

This section used to be a "to do" list, and time proved it wrong: all three of its
phases finished in v1.0. Below is the real state of things.

### 12.1 Done (in v1.0)

| Phase | What | Result |
|---|---|---|
| **A - validation** | `en-eval`, `en-gen`, sample report | ✅ BPC **0.6064** · ppl **7.56** · OOD BPC **3.2914** · 7 samples (`docs/SAMPLES.md`) |
| **B - data** | Project Gutenberg + TinyStories | ✅ **2.05 B tokens**, exactly one epoch |
| **C - architecture** | KV-cache, RoPE, RMSNorm, SwiGLU, weight tying, dropout | ✅ all of them; the KV-cache matches the full forward pass to **5e-8** |
| **C - correctness** | same weights in PyTorch | ✅ loss differs by **1.8e-05** (threshold 2e-3) |

### 12.2 v1.5 - the engineering release

v1.0 proved **model quality**: BPC 0.6064, a 1.8e-05 difference from PyTorch. But the
measurements exposed three weak spots, and v1.5 should close them - not with a new
architecture, but with better engineering of the same model.

#### A - Performance

| # | Work | What it does | Effort |
|---|---|---|---|
| **A1** | **bf16 mixed precision** ⭐ | Either 2x speed or 2x model in the same 12 GB. `__half` kernels + `cublasGemmEx`, fp32 master weights. bf16 has the same exponent range as fp32, so **no loss scaling is needed** - which was fp16's whole problem | 1-2 weeks |
| **A2** | **Close the VRAM gap** | PyTorch uses 7.10 GB, we use 8.40. The difference is most likely materialising the 16,384-way logits for every token (134 MB + 134 MB of probs) and not reusing activation buffers. Chunked cross-entropy + buffer reuse | 3-5 days |
| **A3** | **Ring-buffer KV-cache** | Past 512 tokens generation degrades to O(n*512). A ring buffer with a RoPE position offset gives constant memory and arbitrary length | 2-3 days |
| **A4** | **Gradient checkpointing** | Recompute activations in the backward pass -> room for a **400M model** (20-30 % slower) | ~1 week |

#### B - Usability

| # | Work | What it does | Effort |
|---|---|---|---|
| **B1** | **Checkpoint v3 - one file** | Config + vocabulary + merge table + weights in a single `.sfa` file. Ends the "which file did I need?" question at release time | 2-3 days |
| **B2** | **Sampling improvements** | Repetition penalty (kills the greedy loop and the "the old woman and the old woman" repetition), top-p, stop strings. **The cheapest work with the largest effect on perceived quality** | 1 day |
| **B3** | **`en-eval`/`en-gen` tests** | Neither path runs in CI today; a small checkpoint plus two tests is enough | 2 days |

#### C - Measurement honesty

| # | Work | What it does | Effort |
|---|---|---|---|
| **C1** | **A real held-out Gutenberg split** | The validation set is TinyStories only while 77 % of the training mix is Gutenberg. `en-eval --split gutenberg` would make the long explanation in 7.4 unnecessary | 2 days |
| **C2** | **VRAM profile** | `test-gpt-profile` measures time, not **memory**. It is the prerequisite for A2 | 1 day |

#### Targets

| Metric | v1.0 | **v1.5 target** |
|---|---|---|
| Throughput (222 M) | 10,772 tok/s | **>= 20,000** (bf16) |
| VRAM (222 M) | 8.40 GB | **<= 7.5 GB** (ahead of PyTorch) |
| Largest model | 222 M | **400 M** |
| Generation past 512 | degrades | **constant memory** |
| Checkpoint | 2 files | **1 file** |
| Sampling | temperature + top-k | + repetition penalty + top-p |
| Evaluation | TinyStories only | + held-out Gutenberg |
| Tests | 75 | **+4** |

#### Suggested order

```
1. B2 sampling         (1 day)    cheapest win, visible immediately
2. C2 VRAM profile     (1 day)    the measurement A2 needs
3. A2 VRAM fix         (4 days)
4. B1 checkpoint v3    (3 days)
5. A3 ring KV-cache    (3 days)
6. A1 bf16             (2 weeks)  the big one, the big payoff
7. A4 400M + ckpt      (1 week)   possible once bf16 lands
8. C1 Gutenberg eval   (2 days)
9. B3 tests            (2 days)
```

About **6 weeks** of evenings. This is a scope estimate, not a promise; the order
follows the dependencies.

### 12.3 v2.0 - two candidates

| Candidate | Why it is interesting | The hard part |
|---|---|---|
| **A Turkish model** (50-100 M, real corpus) | A from-scratch Turkish language model is rare; the strongest differentiator | Not the code - the **data**: clean, appropriately licensed, large enough |
| **A browser demo** (10-20 M, WebGPU/WASM) | "No install, click and it runs" is a strong effect | Fitting the model into a browser: quantisation plus a WASM/WGSL backend |

Both rest on v1.5's engineering: a browser demo needs bf16/quantisation, a Turkish
model needs a solid data pipeline.

### 12.4 Deliberately **not** done

| What | Why not |
|---|---|
| **SFT / DPO / chat** | This project is not an assistant. SFT needs 1B+ parameters and human-labelled data; doing SFT on 222M would be teaching it to imitate knowledge it does not have |
| **Multi-GPU** | The target was one consumer card; a data-parallel path would change that premise |
| **Fine-tuning a base model** | The entire project rests on the from-scratch claim; that step would void it |
| **Copying FlashAttention** | We have our own fused attention and it is not the bottleneck at 222M; copying it would add no learning value |

### 12.5 The realistic ceiling

On one RTX 5070 (12 GB, fp32) the reachable endpoint is **~250-300M parameters** and
**~4 B tokens**. That means "a real from-scratch engine plus an honest language
model" - not "a usable assistant". A real assistant needs 1B+ parameters and ~20B
tokens: cloud GPUs (8xA100 for ~2 days) or fine-tuning an existing base model. Both
are outside this project's claim.

---

## 13. Directory layout

```
StonefishAI/
├── Cargo.toml            # version 1.0.0, empty [dependencies], cuda feature (default)
├── LICENSE               # MIT
├── CHANGELOG.md          # release notes
├── README-Tr.md          # Turkish
├── README-Eng.md         # this file
├── README.md             # language picker
├── kernels/
│   └── stonefish.cu      # 711 lines, 33 CUDA kernels (compiled by NVRTC)
├── src/                  # 28 files, 10,349 lines
│   ├── autograd.rs       # 19 ops, topological backward()
│   ├── sys/              # our own CUDA FFI (1,415 lines)
│   ├── gpu_train.rs      # GPU-native trainer
│   ├── en_train.rs       # English LM training and evaluation
│   ├── bpe.rs            # byte-level BPE-16k
│   └── ...               # tensor, transformer, trainer, checkpoint, corpus
├── tests/                # 6 files, 75 tests (2,322 lines)
├── examples/             # bpe_bench.rs, grad_probe.rs
├── tools/                # 15 Python tools: measurement, charts, guards, release gate
├── docs/                 # MODEL-CARD, REPRODUCING, RELEASE, SAMPLES,
│                         #   HF card, report.json, 7 charts
├── .github/workflows/    # CI: 5 jobs (CPU, CUDA, zero deps, docs, language)
├── data/                 # (not in git) MNIST + corpora
└── runs/                 # (not in git) training manifests
```

`src/sys/` holds our own CUDA FFI: the driver, cuBLAS and NVRTC are loaded at
runtime through `LoadLibraryW`/`dlopen`. The test count depends on the feature set
(75 with CUDA, 55 without).

**Engine principles:** written from scratch · gradients numerically verified ·
decisions made by measurement · every defect locked down by a test, a profile or a
measurement · runs without CUDA.

---

## 14. How zero dependencies works

`[dependencies]` is empty. GPU access is entirely through our own FFI layer:

```
src/sys/mod.rs    LoadLibraryW (Windows) / dlopen (Linux) + GetProcAddress
                  → function pointers resolved lazily (OnceLock)
src/sys/cuda.rs   cuInit, cuCtxCreate, cuMemAlloc/HtoD/DtoH/Memset,
                  cuModuleLoadData, cuModuleGetFunction, cuLaunchKernel
src/sys/cublas.rs cublasCreate, cublasSetStream, cublasSgemm,
                  cublasSgemmStridedBatched
src/sys/nvrtc.rs  nvrtcCreateProgram, nvrtcCompileProgram, nvrtcGetPTX
```

**Why it matters:**

1. **No CUDA SDK at build time** — no import `.lib`, no `build.rs`
2. **Graceful degradation** — `LoadLibraryW` returns `None`, everything runs on CPU
3. **Auditable** — 1,289 lines of FFI instead of a 400-line crate's abstractions over the same API
4. **Verified** — `stonefish_ai test-ffi` checks alloc + copy + PTX load + kernel launch + cuBLAS SGEMM against CPU references (measured diff: **0.0**)

Kernel arguments are prepared by `Kargs`: each argument is written into one `u64` slot (device pointer = 8 bytes, i32/f32 = low 4 bytes) and `cuLaunchKernel` receives the addresses of those slots — zero allocation, zero copy.

---

## 15. Timeline — from 15 June to today

The project started on **15 June 2026**. The table below lists every step since then, in order, with its measurements.

> **Source note:** 15 June – 15 November was not under version control; that period was reconstructed from the development log and its run output. From 16 November onwards every step has a commit, a measurement and a test.

### 15 June – 15 November 2026 · v0.1 → v0.6 (no version control)

Five months, by hand, without a single ML library. Every version built on the previous one, and at every step the **output was read** - the question was never "does it run" but "what does it write".

#### v0.1 - Tensor + autograd: XOR

`tensor.rs` (a pure-Rust 2D matrix: `matmul` as a triple loop, `add/sub/mul_elem/scale/transpose/map`, our own xorshift RNG), `autograd.rs` (`Op` enum, topologically ordered `backward()`, chain rule), `nn.rs` (`Linear`), `loss.rs` (MSE), `optim.rs` (SGD), `main.rs` (XOR training).

```
0 XOR 0 = 0.0000  ✅     1 XOR 0 = 1.0000  ✅
0 XOR 1 = 1.0000  ✅     1 XOR 1 = 0.0000  ✅
```

**First victory:** a from-scratch neural network learned XOR.

#### v0.2 - Softmax + cross-entropy

`SoftmaxCE` added as a single numerically stable op (forward `-log(softmax[target])`, backward `softmax - onehot`). Three-class synthetic data:

```
CE loss = 1.100955 -> 0.000017        Accuracy: 60/60 = 100.0% ✅
```

#### v0.3 - MNIST, save/load, optimizer comparison

`data.rs` (IDX binary parser written from scratch on `std::fs`, 0..1 normalization, one-hot labels), `nn.rs` (`save_model`/`load_model` - our own binary format: `SFISH001` magic + version + layers), `main.rs` (`--train/--predict/--eval`).

| Optimizer | LR | Epochs | Test accuracy |
|---|---|---|---|
| SGD | 0.1 | 10 | 97.66 % (stuck) |
| SGD | 0.01 | 10 | 98.06 % |
| **SGD** | **0.01** | **20** | **98.22 %** |
| Adam | 0.001 | 10 | 97.66 % |
| Adam | 0.003 | 20 | 97.69 % |

The model was saved (`stonefish.bin`, 407 KB) and reproduced the same result in a separate run.

> **Lesson:** on simple problems like MNIST, Adam can be *worse* than SGD. Hyperparameters matter.

#### v0.4a - Tokenizer

`tokenizer.rs` (character level, text <-> id), `data.rs` (`load_text`, `make_sequences`). Shakespeare, 1.1 MB:

```
Vocab: 65        Round-trip: ✅        Sequences: 17,428
```

#### v0.4b - Transformer ops: Softmax, GELU, LayerNorm

All three added with their backward passes and verified **numerically**:

```
softmax([1,2,3]) = [0.0900, 0.2447, 0.6652] ✅
gelu([-1,0,1])   = [-0.1588, 0.0000, 0.8412] ✅
ln([1,2,3])      = [-1.2247, 0.0000, 1.2247] ✅
```

#### v0.4c - Embedding (token + position)

```
token_emb grad: 10x8 ✅    pos_emb grad: 16x8 ✅    unused token gradient: 0 ✅
```

#### v0.4d - Attention

The `MaskCausal` op (blocks the future) + `AttentionHead` (Q, K, V, mask, softmax, output projection):

```
Output: 4x8 ✅    No gradient is zero ✅
```

#### v0.4e - GPT model + training

`TransformerBlock` (pre-norm + attention + residual + FFN) + `GPT` (embedding + N blocks + final LayerNorm + head), `gpt_train()` (Adam, 17k steps), `gpt_generate()` (sampling).

| Epoch | Loss | Generated text |
|---|---|---|
| 1 | 4.17 -> **2.32** | `ROMEO: Go the, sever, onren'd fot heine fomp whit, Efrastau wre cagenter...` - nonsense, but the **format is right** (`ROMEO:` and character names) |
| 3 | **1.86** | `ROMEO: He the forling of, the for helf: Foroct! you him, stays cale.` - real Shakespeare characters like `AUTOLYCUS:`, real words like `thou`, `Who`, `fair` |

#### v0.5 - Fixing Adam, growing the model

**Problem:** the bigger model (d=128, 4 layers, multi-head) + Adam -> loss **stuck at 2.5 and climbing**.

**Diagnosis:** SGD worked (1.88), Adam did not -> `eps` too small (1e-8) and `lr` too high (5e-4). The multi-head "summation" approach was also wrong.

**Fix:** `eps = 1e-6`, `lr = 1e-4`, `heads = 1`. Result **loss 1.64**:

```
ROMEO: He will retorm or, the entermenn goes / shew chas of unwy can myself hops / Hast aly set.
QUEEN ELIZABETH: Or the I weep, queen, in Pawien'd theselves disholy gond / Officy thange me...
```

`QUEEN ELIZABETH` (a real character), `He will retorm` (almost meaningful), `shew` (Shakespeare's "show") - the sentence structure has settled.

#### v0.5 - Turkish data

`PoemsJson.json` (5 MB) -> `turkish.txt` (~3-4 MB via PowerShell). **Vocab 140** (65 for Shakespeare), **22,434 sequences** (2.5x Shakespeare).

| Epoch | Loss |
|---|---|
| 0 | 2.2215 |
| 1 | 1.8643 |
| 2 | **1.7542** <- target ~1.75, hit exactly |

Despite a vocabulary more than twice as large, the loss nearly caught up with Shakespeare - the Turkish data is much richer.

```
Prompt: "Aksham" (evening)
Aksham dikleri, camib irerde / Korkmalar cemlatlayacaksin, yashim"nemeder, aldir,
Gulumse con susuzlugu / Sonra ve daha boyle intirili birakmaketli chikraliyor shehri... <!-- tr:sample -->
```

Meaningless, but **structurally Turkish**: suffixes, stems and punctuation are right.

#### v0.6 - CUDA integration

CUDA 13.5 + RTX 5070 installed, `cudarc 0.19.10` added. `gpu.rs` (`GpuMatrix`, `matmul` via cuBLAS, CPU wrapper), `Tensor::matmul` routed to the GPU.

| Test | Result |
|---|---|
| `test-cuda` | ✅ RTX 5070 recognised |
| `test-gpu-matmul` | ✅ same result as the CPU |
| `test-gpu-bench-big` | ✅ 1024^3 matmul: **55,000x** |
| `test-gpu-train` | ✅ GPU training loop works |
| `gpt-train` (GPU) | ✅ same loss (1.75) - but **GPU utilisation 2 %** |

> **Key observation:** GPU matmul works, but **transfer cost dominates**. Full integration is needed. (Measured on 16 November: a step issues 75 matmuls yet the real GEMM time is 1 ms - 0.8 % of the 120 ms total.)

#### v0.6+ - Autograd fixes

- **1/batch** added to the `SoftmaxCE` backward (a critical correctness bug)
- `LayerNorm`'s `eps` embedded in `Op::LayerNorm(f32)`
- Topological sort via `HashSet`: **O(N^2) -> O(N)**
- `Sigmoid`, `Softmax`, `LayerNorm`, `SoftmaxCE` cached
- `MatMul` forward on the GPU

**After five months:** a working autograd engine, 98.22 % on MNIST, a GPT writing in two languages, GPU matmul - and **zero tests**.

**Invisible problems** (found the next day): batching was fake, the GPU was used for only half of training, `Tensor3D` was entirely dead code, the checkpoint did not store the vocabulary, and there were 0 tests.

### 16 November 2026 · day 1 — audit and hardening (8 commits)

| Time | Step | Measurement / result |
|---|---|---|
| 14:22 | **Git initialised**, `.gitignore`, v0.6 baseline commit | 11 source files, 0 tests |
| 14:24 | Library/binary split + **first gradient tests** | 0 → **21 tests**; `mask_causal` failed (the *test* was wrong: the `-1e9` mask exceeded `f32` resolution) |
| 14:33 | **Real batching** (row packing) + fused attention + **checkpoint v2** + dead-code removal | `Tensor3D` (−250 lines) deleted; magic + config + vocab embedded |
| 14:59 | README (design decisions + measurements) | — |
| 15:01 | Clean clippy + LF line endings | 9 → **0 warnings** |
| 16:00 | **A proper training loop**: warmup + cosine, gradient clipping, best-val checkpoint, manifest | 6 epochs, 16,626 steps, **54.4 min**, val CE **1.6337** |
| 16:08 | **Profile → 3 optimizations** | Step **177 → 120 ms** (+47 % tok/s) |
| 16:32 | **GPU-native trainer** (device-resident, 15 kernels that day, 33 now) | 6 epochs **54 min → 70 s** = **46.7×**; 244,698 tok/s; val CE **1.6230** |
| 17:14 | **English pipeline**: BPE-16k + u16 disk cache + 236 M model | TinyStories 2.1 GB → 450.9 M tokens, 4.7 chars/token |

**End of day 1:** 40 tests, no dead code, training 46.7x faster wall clock, the first English model training.

### 17 November 2026 · day 2 — zero dependencies, v2 architecture, measurement (15 commits)

| Time | Step | Measurement / result |
|---|---|---|
| 04:36 | README-Tr + README-Eng (13 sections, roadmap) | 2 × 380 lines |
| 05:41 | `en-eval` + `en-gen` + first real measurement | val loss **2.3700** · perplexity **10.70** · **BPC 0.6913** |
| 05:48 | **ZERO DEPENDENCIES** — `cudarc` removed, our own CUDA/cuBLAS/NVRTC FFI | `cargo tree` → **a single line**; `test-ffi` diff **0.0** |
| 05:56 | v2 kernels: RMSNorm, SwiGLU, RoPE, dropout | each verified against a CPU reference |
| 05:59 | **One-command demo** | 1.8 M model in 51 s, **352,494 tok/s**, writes Shakespeare |
| 06:09 | **v2 architecture** (CPU + GPU together) | 235.6 M → **222.3 M** parameters (weight tying −16.8 M); gradient equivalence kept |
| 06:22 | **Data window bug** found and fixed | 5,679 → **9,966 tok/s** (~4 GB of disk traffic per step) |
| 06:30 | **KV-cache** | Generation O(n²) → O(n); **5e-8** from the full forward pass |
| 11:37 | `cargo fmt` + CI (3 jobs) + **atomic checkpoints** | A power cut can no longer corrupt a checkpoint |
| 11:46 | **PyTorch comparison** | **20,487 tok/s** vs PyTorch 11,456 (naive) / 10,810 (SDPA) = **1.79–1.90×** |
| 13:59 | **Entire codebase translated to English** | 1,082 lines, 22 commits; **0** Turkish outside README-Tr |
| 14:04 | Dead-code audit | 6 items removed; the profiler's lost hook restored; 855 MB of dead artifacts deleted |

### 17 – 19 November 2026 · the 2-billion-token run

Mixed corpus: **TinyStories 462 M + Gutenberg 1,585 M = 2,047,457,313 tokens** (tokenized in 2 minutes).

```
Model  : d=1024, 16 layers, 8 heads (d_head=128), ctx=512, BPE-16,384
         -> 222,331,904 parameters
Batch  : 4 x gradient accumulation 16 = 32,768 tokens/step
Plan   : 62,483 steps - max_lr 4e-4 - 1000-step warmup + cosine - clip 1.0
Time   : ~54 hours - 10,562 tokens/s - 8.2 GB VRAM - 66-80 C - 250 W
```

![Loss curve and learning-rate schedule](docs/loss-curve.png)

![Training throughput](docs/throughput.png)

Loss curve (from the log, step by step):

| Step | Loss | Note |
|---|---|---|
| 20 | 9.54 | random baseline ln(16384) = 9.70 |
| 1,440 | 5.20 | warmup finished |
| 5,640 | 4.48 | |
| 8,380 | 4.01 | 13.4 % |
| 15,000 | 3.65 | 24 % |
| 25,000 | 3.41 | 40 % |
| 35,000 | 3.27 | 56 % |
| 45,000 | 3.19 | 72 % |
| 55,000 | 3.14 | 88 % |
| 62,483 | 3.11 | last step |

> This is the **training** loss (dropout on, ~77 % of the mix is Gutenberg). The
> validation loss is lower - section 7.4 explains why.

### 19 November 2026 - day 3: the pre-release audit (40+ commits)

The release package was audited while training ran. The rule: **every claim is
verified by running it**, being written down is not enough. What turned up:

| Claim | Said | Actually |
|---|---|---|
| MNIST accuracy | 98.12 % | **94.51 %** with the same hyperparameters - the `1/batch` fix left lr 64x too small; `lr=0.5` gives 98.14 % |
| Section 1, "what it can do" | "**no KV-cache** -> O(n^2)" | there is one, with two passing tests |
| Quick start | `train` (98.12 %), `en-train 450M`, "236 M" | `train 20 0.5`, `en-train 2047457313`, 222 M |
| Architecture table | line counts for 20 files | all re-measured (10,328 lines) |
| Section 13 tree | version 0.7.0, "6,300 lines", "62 tests" | 1.0.0, 10,328, 75 |
| Model card | "~5.5 days", a held-out Gutenberg row, baseline BPC 2.28 | ~54 hours, row removed, baseline 2.909 |
| Section 7.3 speed-up | "46.7x" (the table yields 28.9x) | **28.6x per step**, **46.6x wall clock** - both stated |
| Data table | TinyStories only | TinyStories **+ Gutenberg** (the model used both) |
| README.md (landing page) | v1 numbers | v2: 222 M, 2.05 B tokens, 75 tests |

**Guards installed in their place** (all running in CI): language · 86 links ·
30 anchors · 71 table widths · every file's line count · 14 headline numbers
against their artifacts · every path in the tree · `data/`+`runs/` still ignored.

**CI went red twice**, both times for the same class of reason: checks that
depended on files present here and absent in a clone. The fix was to build the
exact CI environment with `git archive` and run the checks there (69 files, no
`data/`, no `runs/`). `cargo test` now passes there with 55 tests.

**Early measurement** (step 58,500, PyTorch on CPU, never touching the GPU):
TinyStories valid loss 1.947 · Gutenberg 2.898 · README probe 4.422. All three
below their own random baseline. It also explained the apparent "val loss below
train loss": the training mix is ~77 % Gutenberg, validation is TinyStories only.

### 19 November 2026 - night: training finished, measured

The 62,483rd step completed at 23:26 and `stonefish_en.final.state` (848 MB) was
written. With the GPU free, the measurement ran as one command
(`python tools/measure.py`):

| Measurement | Result |
|---|---|
| TinyStories (held out, 32,768 tokens) | loss **2.0227** · ppl **7.56** · BPC **0.6064** |
| README probe (OOD, 20,514 tokens) | loss **4.5849** · BPC **3.2914** |
| PyTorch parity (same weights, 8 batches) | **1.8e-05** difference (threshold 2e-3) |
| Training throughput | 10,772 tok/s (last 200 steps) |
| PyTorch SDPA / naive | 10,502 / 9,996 tok/s |

Four bugs surfaced in `measure.py` and were fixed: subprocess output was decoded as
**cp1252** (our own emoji crashed it), the bench parser choked on the `"10,076"`
thousands separator, a failed measurement compared `None`, and **chars-per-token used
the wrong denominator** - which made the TinyStories random baseline look like 0.024
bits/character instead of 2.909.

`docs/SAMPLES.md` was added too: seven **unfiltered** generations, two good and five
bad, each with a note about what to notice.

### In numbers: 5 months → today

| | 15 June (v0.6) | Today |
|---|---|---|
| Tests | 0 | **75** |
| Dependencies | 4 crates | **0** |
| Largest model | 842 K parameters | **222 M** |
| Training speed | 5,714 tok/s | **244,698 tok/s** (43×) |
| Data | 2.87 M characters | **2.05 billion tokens** |
| Released model | - | **222 M parameters · BPC 0.6064** (random baseline 2.909) |
| Parity with PyTorch | - | same weights, **1.8e-05** difference |
| Findings | — | **23** — 19 code defects and 4 process/infrastructure problems, each caught by a test, a profile or a measurement |
| Turkish lines (excl. README-Tr) | — | **0** |


---

## 16. Licence and citation

**MIT** - code and weights. The `LICENSE` file is authoritative; this paragraph is
only a summary.

The training data is not redistributed and keeps its own licences: TinyStories
(CDLA-Sharing-1.0), Project Gutenberg (public domain in the US). That is why
`data/` is in `.gitignore`.

```bibtex
@software{stonefishai2026,
  title  = {StonefishAI: a language model from scratch in Rust with zero dependencies},
  author = {Edipcm-hash},
  year   = {2026},
  url    = {https://github.com/Edipcm-hash/stonefish}
}
```
