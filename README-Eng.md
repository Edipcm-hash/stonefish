# 🐟 StonefishAI

**A from-scratch autograd engine and language-model stack in Rust — no ML libraries.**

Everything from `Tensor` and topological backprop to optimizers and CUDA kernels is written by hand in this repository. The only external dependency is `cudarc`, which enables the GPU path (feature-gated; with it off, everything runs on CPU).

> Türkçe sürüm: [README-Tr.md](README-Tr.md)

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
12. [Roadmap — what comes next](#12-roadmap--what-comes-next)
13. [Directory layout](#13-directory-layout)

---

## 1. What it can / cannot do

### ✅ What it can do

| Capability | Measure |
|---|---|
| Handwritten digit classification (MNIST) | **98.12 %** test accuracy |
| Turkish character-level text generation | val CE **1.6230** (random baseline 4.91) |
| English BPE-16k language-model training | 236 M params, 450 M tokens, ~10,000 tok/s |
| Gradient verification | 29 ops checked by central differences |
| GPU training | device-resident, 244,698 tok/s (small model) |
| Checkpoint management | config+vocab embedded, errors instead of panics, resumable |

### ❌ What it cannot do (honest list)

- **It is not a chat assistant.** No instruction following, Q&A, translation, summarization, or code generation.
- **No world knowledge.** It cannot answer "What is the capital of France?"
- **No reasoning.** No arithmetic, logic chains, or multi-step inference.
- **No meaning.** Generated text is grammatical but semantically empty; the model produces statistical continuations.
- **The Turkish model is 842 K params**, the English one 236 M — both in the "undertrained" class (~8.6 M and ~450 M tokens respectively).
- **No KV-cache** → generation is O(n²); 300 tokens ≈ 2.8 s.
- **The English model is not yet validated** — `en-eval` measurement comes after training finishes.

---

## 2. Quick start

```bash
# Build (on a machine without CUDA: --no-default-features)
cargo build --release

# List all modes
./target/release/stonefish_ai help

# --- MNIST MLP ---
./target/release/stonefish_ai train          # train (20 epochs)
./target/release/stonefish_ai eval           # measure on test set → 98.12 %

# --- Turkish GPT (character level) ---
./target/release/stonefish_ai gpt-train 6 8                        # epochs, batch
./target/release/stonefish_ai gpt-gen "Akşam erken iner" 300 0.8   # prompt, tokens, temperature

# --- English GPT (BPE-16k, 236 M) ---
./target/release/stonefish_ai en-prepare           # learn BPE + build u16 cache
./target/release/stonefish_ai en-train 450000000   # token budget

# --- Measurement ---
./target/release/stonefish_ai test-gpt-profile   # where does the time go
./target/release/stonefish_ai test-gpu-native    # GPU-native trainer speed
./target/release/stonefish_ai test-matmul-bench  # CPU vs GPU at real shapes
./target/release/stonefish_ai test-gpu-bench-big
```

### Data requirements

| File | What | Where |
|---|---|---|
| `data/train-images-idx3-ubyte` + 3 siblings | MNIST | [yann.lecun.com/exdb/mnist](http://yann.lecun.com/exdb/mnist/) |
| `data/turkce.txt` | Turkish corpus (plain text) | your own text |
| `data/en/train.txt` | English corpus | TinyStories v2 (~2.1 GB) |
| `data/en/valid.txt` | English validation | TinyStories valid (~18 MB) |

`data/`, `*.bin`, `*.state`, `*.bpe` are **not committed** (`.gitignore`).

Downloading TinyStories:
```powershell
$b = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
Invoke-WebRequest "$b/TinyStoriesV2-GPT4-train.txt" -OutFile data/en/train.txt
Invoke-WebRequest "$b/TinyStories-valid.txt"        -OutFile data/en/valid.txt
```

---

## 3. Hardware and measured limits

Development/test environment: **RTX 5070 (12 GB), 32 GB DDR5, MSI Trident AS (≈10 L SFF)**, Windows.

| Limit | Value | Consequence |
|---|---|---|
| VRAM | 12 GB | with the fp32 trainer, **~250 M parameters** is the ceiling |
| RAM | 32 GB | 3 B tokens would **not fit** without the streaming pipeline (u16 + disk cache solves it) |
| Thermals | SFF chassis | 80 °C at 250 W → **210 W power limit** applied, dropped to 76 °C |
| Time | single GPU | 236 M model + 450 M tokens ≈ **12.5 hours** |

Measured training throughput:

| Model | Step time | Throughput |
|---|---|---|
| 842 K (GPU-native) | 4.2 ms | **244,698 tok/s** |
| 842 K (CPU + cuBLAS) | 120.8 ms | 8,476 tok/s |
| **236 M (GPU-native)** | ~3.27 s | **~10,000 tok/s** |

---

## 4. Architecture

| File | Lines | Responsibility |
|---|---|---|
| [src/lib.rs](src/lib.rs) | 35 | Library root, `VERSION`, justified clippy allows |
| [src/tensor.rs](src/tensor.rs) | 142 | `Matrix` (row-major 2D): matmul, transpose, broadcast add/sub |
| [src/autograd.rs](src/autograd.rs) | 845 | `Tensor`, `Op` (17 variants), topological backward, all ops |
| [src/nn.rs](src/nn.rs) | 99 | `Linear`, MNIST model (de)serialization |
| [src/transformer.rs](src/transformer.rs) | 246 | Embedding, MultiHeadSelfAttention, TransformerBlock, GPT, GptConfig |
| [src/checkpoint.rs](src/checkpoint.rs) | 147 | GPT checkpoint v2 (magic/version/config/vocab/tensors) |
| [src/optim.rs](src/optim.rs) | 167 | SGD, Adam (with bias correction), `clip_grad_norm` |
| [src/loss.rs](src/loss.rs) | 7 | Softmax + cross-entropy |
| [src/data.rs](src/data.rs) | 79 | MNIST idx reader, text → sequences |
| [src/tokenizer.rs](src/tokenizer.rs) | 128 | Character-level tokenizer |
| [src/bpe.rs](src/bpe.rs) | 236 | Byte-level BPE: merge learning + encode/decode |
| [src/corpus.rs](src/corpus.rs) | 251 | u16 token cache, windowed streaming reader |
| [src/gpu.rs](src/gpu.rs) | 362 | cuBLAS GEMM (plain/transposed/device/strided-batched) + CPU fallback |
| [src/gpu_kernels.rs](src/gpu_kernels.rs) | 405 | NVRTC compilation + PTX cache + 15 kernel wrappers |
| [src/gpu_train.rs](src/gpu_train.rs) | 1289 | Device-resident GPT trainer (fast path) |
| [src/en_train.rs](src/en_train.rs) | 270 | English pipeline: `prepare` + `train` |
| [src/profile.rs](src/profile.rs) | 119 | Lightweight profiling counters |
| [src/main.rs](src/main.rs) | 1252 | CLI (20 modes) + training loops + `Trainer` abstraction |
| [kernels/stonefish.cu](kernels/stonefish.cu) | 389 | CUDA kernels |

Total: **~6,300 lines of Rust** + 389 lines of CUDA.

---

## 5. Design decisions

### 5.1 Batching by row packing (no Tensor3D)

A `(batch, seq, d_model)` tensor is stored as a **`(batch·seq, d_model)` matrix**:

```
(B, S, D) ──pack──▶ (B·S, D)
```

LayerNorm, Linear, GELU, bias-add and softmax are already **row-wise**, so this transformation is free. The only exception is attention, handled inside `Op::CausalAttention` with `batch`/`seq`/`n_heads` metadata.

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
| Where | `main.rs` + `autograd.rs` | `gpu_train.rs` |
| Role | **reference** (gradient tests) | tied to CPU via equivalence tests |
| Speed | 8,476 tok/s | **244,698 tok/s** |

`gpt-train` automatically selects the GPU path when CUDA is present, otherwise falls back to CPU.

### 5.6 Tokenizer: characters first, BPE later

A character-level model spends most of its capacity learning **spelling**. With byte-level BPE-16k the same window covers **4-5× more text** (measured: 4.7 characters/token).

BPE learning uses **incremental pair counting**: after each merge only words containing that pair are rescanned → 16k merges in **7 seconds** (the naive approach took hours).

### 5.7 Data pipeline: u16 + disk cache

`Vec<usize>` = 8 bytes per token → 3 B tokens would need **24 GB of RAM**. Instead:

- tokenized corpus is written to disk as **u16** (2 bytes/token)
- training reads through a 64 MB **memory window**; the whole file is never loaded
- tokenization: **148 MB/s** (word-cached)

---

## 6. The two model pipelines

### 6.1 Turkish (character level)

| | |
|---|---|
| Model | `d_model=128`, 4 layers, 4 heads (`d_head=32`), ctx=128 |
| Vocab | 136 characters |
| Parameters | **842,632** |
| Corpus | `data/turkce.txt` — 2,870,158 characters, 22,423 sequences |
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

Reproducibility: hyperparameters + seeds + per-epoch metrics are written to `runs/<ts>.json`.

### 6.2 English (BPE-16k) — **active development**

| | |
|---|---|
| Model | `d_model=1024`, 16 layers, 8 heads (`d_head=128`), ctx=512 |
| Vocab | **16,384** (byte-level BPE) |
| Parameters | **235,571,200** |
| Corpus | TinyStories v2: **450,873,533** train + 3,931,894 valid tokens |
| Compression | 4.7 characters/token |
| Batch | 4 × gradient accumulation 16 = effective 64 sequences (32,768 tokens/step) |
| Plan | 13,732 steps, max_lr=6e-4, 500-step warmup + cosine, clip=1.0 |
| VRAM | 8.4 GB / 12 GB |
| Speed | ~10,000 tok/s → **~12.5 hours** |
| Checkpoint | `stonefish_en.state` (`SFEN001`, every 500 steps) |

**Training completed (2026-11-17):** 13,732 steps, **745 minutes**, 10,061 tok/s average.

**Measured quality (`valid.bin`, 3,928,064 tokens):**

| Metric | Value |
|---|---|
| val loss | **2.3700** nats/token |
| perplexity | **10.70** |
| **BPC** | **0.6913 bits/character** |
| final train loss | ~1.95 (gap 0.42 → mild overfitting) |
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

### 7.1 MNIST MLP (784→128→10)

SGD lr=0.01, batch 64, 20 epochs → **98.12 %** test accuracy (10,000 images, 196 errors).

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

**46.7× speed-up.** Val CE: CPU 1.6337 / GPU 1.6230 (fast-math + summation order).

### 7.4 English 236 M

~10,000 tok/s, 8.4 GB VRAM, 76 °C (at 210 W), GPU 99 % — throughput never dropped over 11 hours.
Loss curve: 9.70 (random) → **~1.97** (at 90 %, still falling).

---

## 8. Time-budget profile

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

**Key insight:** a step issues 75 matmul calls but the real GEMM total is 1 ms. So "keep GEMMs on the GPU" (eliminating transfers) buys at most 17 %; the real win comes from **moving the computation itself to the GPU**. The GPU-native path does exactly that and yields 46.7×.

---

## 9. Tests

**62 tests**, green in both feature configurations:

| Group | Count | What it verifies |
|---|---|---|
| [tests/gradcheck.rs](tests/gradcheck.rs) | 29 | Every op: analytic vs numeric gradient by **central differences** |
| [tests/gpu_kernels.rs](tests/gpu_kernels.rs) | 12 | 15 CUDA kernels + device GEMM modes against CPU reference |
| [tests/gpu_train_equiv.rs](tests/gpu_train_equiv.rs) | 2 | GPU trainer ↔ CPU autograd: **all gradients** + 5-step Adam loss curve |
| [tests/checkpoint.rs](tests/checkpoint.rs) | 5 | Roundtrip, bad magic, truncated file, vocab mismatch, drift |
| Unit tests | 14 | Tokenizer (4), optimizer/clip (4), BPE (4), corpus (2) |

Gradient coverage: `add`/`sub` (incl. broadcast), `mul_elem`, `matmul` (incl. square), `matmul+transpose`, `scale`, `relu`, `sigmoid`, `transpose`, `sum`, `softmax_ce`, `softmax_ce_ids` (sparse), `softmax`, `mask_causal`, `gelu`, `layer_norm`, `embedding`, `causal_attention` (single/multi-head + batched), end-to-end tiny GPT, causality contract, batch gradient = Σ per-sequence gradients.

```bash
cargo test --release                       # CUDA path
cargo test --release --no-default-features  # CPU path
```

---

## 10. Bugs found during development

This list is the most valuable part of the project: **every one was caught by a test.**

| # | Bug | How it was found |
|---|---|---|
| 1 | `mask_causal` test: `-1e9` mask pushed the loss to ~1e9; `f32` ulp(1e9)≈64 swallowed `ΔL≈1e-3` | Gradient test (the *test* was wrong) |
| 2 | `eps` calibration in gradcheck: embedding init 0.02 + LayerNorm → `eps=1e-3` is 10 % of a row, breaking linearity | Parameter-wise measurement with `grad_probe` |
| 3 | Wrong factor in the batch-gradient identity (B³) | Derived `L_batch = (1/B)·ΣL_b` and fixed |
| 4 | Attention backward used `q` instead of `dOut` for `dW` | 3 gradient tests turned red after vectorization |
| 5 | **Head stride**: score-buffer offset assumed `h·S²`, correct is `h·B·S²` | Scores came out zero → softmax 0.5/0.5 |
| 6 | **Residual gradient**: attention branch upstream is `dOut + dx_ln2`, not `dOut` (`res1 = x + proj`) | dAttn/dWo were wrong from the start |
| 7 | `dev_gemm` passed `beta=1.0` instead of `0.0` → results accumulated | GEMM mode test |
| 8 | BPE encode processed the whole 64 MB chunk at once (full vector scan per merge → O(n²)) | Tokenization appeared to hang; fixed with a word cache |
| 9 | Two processes writing the same output file ("Access denied") | A killed job's child process survived |
| 10 | An **orphan training process** started at 13:33 would have overwritten the checkpoint in the old (v1) format | Process inventory |
| 11 | `cargo clippy --fix` rewrote all of `data.rs` due to line endings | Reverted; `.gitattributes` (LF) added |

---

## 11. Known limitations

- **No KV-cache** — generation recomputes the whole context per token (O(n²)).
- **Position drift in generation** — as the window slides, `pos_emb` restarts from 0, so positions do not match training (RoPE fixes this).
- **fp32 training only** — no bf16; this caps VRAM usage at ~250 M parameters.
- **No gradient checkpointing** — activations are kept; needed for larger models.
- **Single-threaded CPU path** — no `rayon`; the CPU path is the reference/fallback.
- **Turkish model processed 17 M tokens** but the corpus is only 2.87 M (3.4 unique tokens/param) → limited diversity, heavy epoch repetition.
- **English model saw 450 M tokens** (1.9 tokens/param) → 1/10 of Chinchilla, i.e. **undertrained**.
- **No SFT/DPO** → the model shows no assistant behaviour.
- **No validation/generation tooling for the English path yet** (`en-eval`, `en-gen` are on the roadmap).
- No weight tying, dropout, RoPE, SwiGLU, or RMSNorm.

---

## 12. Roadmap — what comes next

### Phase A — Validate the English model *(on completion, ~1 hour)*

- [ ] **A1 `en-eval`**: val loss, **perplexity**, **BPC** over `valid.bin` (3.9 M tokens)
  - BPC = loss[token] / (ln2 × chars/token) → the only tokenizer-independent metric
  - Expected band after 450 M tokens: `loss 1.7–2.0`
- [ ] **A2 `en-gen`**: BPE encode → sliding-window forward → temperature/top-k → decode
- [ ] **A3 sample report**: raw generations (temperature 0 and 0.8)

### Phase B — Grow the data *(highest payoff)*

- [ ] **B1** Download and clean Project Gutenberg (~3-5 B tokens; strip headers/footers)
- [ ] **B2** Mix: 60 % TinyStories + 40 % Gutenberg → 2-4 B tokens
- [ ] **B3** Re-tokenize → ~6-8 GB u16 cache
- [ ] **B4** Retrain the same architecture on 2-4 B tokens (~2-3 days)
  - Target: 8-16 tokens/parameter → clear quality jump

### Phase C — Architectural improvements

- [ ] **C1 KV-cache**: 10-50× faster generation (O(n²) → O(n))
- [ ] **C2 RoPE**: rotary positions instead of learned absolute ones → long context, no position drift
- [ ] **C3 RMSNorm** (instead of LayerNorm), **SwiGLU** (FFN), **weight tying** (embed ↔ head), **dropout**
- [ ] **C4 bf16 mixed precision**: 2-3× speed + larger models (`__half` support in kernels)
- [ ] **C5 Memory-efficient/fused attention** (flash-attention style)
- [ ] **C6 Gradient checkpointing**: recompute activations → room for 300-500 M models

### Phase D — Scale

- [ ] **D1** Try 300 M (d=1024, L=20), measuring VRAM
- [ ] **D2** ctx 1024 (currently 512)

### Phase E — Alignment (assistant behaviour)

- [ ] **E1** SFT dataset: 10-50 k `(instruction, response)` pairs *(the hard part)*
- [ ] **E2** Chat template (`<|user|>… <|assistant|>…`) + SFT fine-tuning
- [ ] **E3** DPO / preference training (style, helpfulness, refusal)
- [ ] **E4** Instruction-following evaluation set

### Phase F — Engineering debt

- [ ] **F1** Checkpoint **v3**: embed the BPE merge table (currently a separate `SFEN001` file)
- [ ] **F2** Automated tests for `en-eval`/`en-gen`
- [ ] **F3** Periodic checkpointing for the Turkish path too
- [ ] **F4** Keep both READMEs updated as Phases A-E progress

### Expectation table

| After phase | What you get |
|---|---|
| A | Measured quality + first real English samples |
| B | Fluent, topic-coherent, 2-3 paragraphs |
| C | ~2× speed on the same hardware, long context, less VRAM |
| D | Quality ceiling (hardware limit) |
| E | A model that **answers questions** (no knowledge, but the format) |

> **Honest caveat:** with a single RTX 5070 the reachable end point is ~250-300 M parameters and ~4 B tokens. That means "an impressive from-scratch engine + a real language model" — not "a usable assistant". A real assistant needs 1 B+ parameters and ~20 B tokens → cloud GPUs (8×A100, ~2 days, ~$700) or fine-tuning an existing base model.

---

## 13. Directory layout

```
StonefishAI/
├── Cargo.toml            # version 0.7.0, cuda feature (on by default)
├── README-Tr.md          # Turkish
├── README-Eng.md         # this file
├── kernels/
│   └── stonefish.cu      # 15 CUDA kernels (compiled via NVRTC)
├── src/                  # 18 modules (~6,300 lines)
├── tests/                # 4 files, 62 tests
├── examples/
│   ├── bpe_bench.rs      # BPE tokenization speed benchmark
│   └── grad_probe.rs     # parameter-wise gradient diagnostics
├── data/                 # (untracked) MNIST + corpora
└── runs/                 # (untracked) training manifests
```

**Engine principles:** written from scratch · gradients numerically verified · decisions driven by measurement · every bug locked down by a test · runs without CUDA.
