# Changelog

All notable changes to this project. The v0.x history is reconstructed from the
development log: those five months predate version control, so only the results
are documented, not the individual commits.

## v1.0.0 — 19 November 2026

The first release. A language model written from scratch: hand-written autograd,
byte-level BPE, a RoPE/RMSNorm/SwiGLU transformer, our own CUDA driver FFI, and
a 222M-parameter model trained on 2.05B tokens on a single consumer GPU.

### Headline numbers

| | |
|---|---|
| Parameters | 222,331,904 |
| Training data | 2,047,457,313 tokens (TinyStories + Gutenberg) |
| Dependencies | **0** (`cargo tree` is one line) |
| Tests | **74** |
| Training speed | ~10,700 tokens/s on one RTX 5070 at 250 W |
| Rust | 9,890 lines in src/ (11,362 with tests and examples) |
| CUDA | 711 lines |

### Added

- **Autograd**: reverse-mode over a 19-variant `Op` enum, topological
  `backward()`, 29 numerical gradient checks covering every differentiable op.
- **Batching without a 3D tensor**: `(B,S,D)` is packed into rows of `(B*S,D)`,
  so every linear layer is one GEMM and the attention kernel handles the
  batch/head strides itself.
- **Transformer v2**: RMSNorm (no bias), SwiGLU (`silu(xW1) ⊙ (xW3)W2`),
  RoPE with a cached cos/sin table, weight tying (head = token embedding),
  dropout with a per-layer/per-step hash PRNG.
- **Own CUDA FFI** (`src/sys/`): `LoadLibraryW`/`dlopen` + `GetProcAddress`,
  NVRTC runtime compilation with a PTX disk cache, cuBLAS bindings including
  the strided-batched path. No `cudarc`, no build script, no import library.
- **GPU-native trainer**: parameters, gradients and optimizer state stay on the
  device for the whole run. One step = forward + backward + Adam + clipping
  without a single host round trip.
- **Byte-level BPE tokenizer** (16,384 merges) with a `u16` disk cache, a
  windowed reader so a 4 GB corpus never has to fit in RAM, and a word cache
  during encoding (148 MB/s).
- **KV-cache generation**: O(n) instead of O(n²), verified against the full
  forward pass to 5e-8.
- **Atomic checkpoints**: written to a temp file, `sync_all`, then renamed, so a
  power cut cannot truncate a run.
- **Single-command demo**: `cargo run --release -- demo 90` trains a small GPT
  from scratch and writes Shakespeare in 90 seconds.
- **Comparison and measurement tooling**: `tools/torch_bench.py` (same
  architecture in PyTorch, speed), `tools/torch_parity.py` (same weights, loss
  parity), `tools/make_charts.py` and `tools/make_report.py` (the README charts,
  drawn from the real logs).
- **CI**: four jobs — CPU-only build and tests, CUDA-flagged build, zero-dependency
  verification, and a language guard.

### Changed from v0.6

- `Tensor3D` deleted (−250 lines): batching is row packing.
- Checkpoint format v2 (`SFGPT002` for the CPU model, `SFEN001` for the GPU
  trainer): magic, version, config and vocabulary are inside the file, so a
  mismatched corpus is rejected instead of silently producing garbage.
- GPU matmul moved from `cudarc` onto our own FFI.
- The v1 English model (236M, TinyStories only) is superseded by the v2
  architecture; its measured numbers are kept in the README for reference.

### Fixed

See the README's bug section for the full list with symptoms and root causes.
The ones that mattered most:

- attention backward used `q` where it needed `dOut` for `dW`;
- the batched score buffer used `h·S²` instead of `h·B·S²`;
- `dev_gemm` accumulated (`beta=1.0`) instead of overwriting;
- the data window read 64 MB per miss (~4 GB of I/O per step);
- the KV cache forgot the 1/√d_head attention scale;
- the language guard only caught Turkish written with diacritics;
- three compile errors that the English translation introduced.

### Known limitations

- **Not a chat assistant.** No instruction tuning, no RLHF, no tool use. It
  continues text; it does not answer questions.
- **Small by modern standards.** 222M parameters on 2.05B tokens is about 9.2
  tokens per parameter, well below the ~20 that scaling studies recommend.
- **English only** (the Turkish model from v0.5 was a character-level toy and is
  not part of this release).
- **Single GPU.** The trainer targets one device; there is no data-parallel path.
- **fp32 only.** bf16 mixed precision is the next step and would roughly double
  the reachable model size.

## v0.6 — 15 November 2026

Last version before version control. Autograd, MNIST MLP (98.22 %), a character
GPT for English (loss 1.86) and Turkish (loss 1.75), and cuBLAS matmul through
`cudarc`. No tests.

## v0.1 – v0.5

XOR → softmax cross-entropy → MNIST → character tokenizer → transformer ops →
GPT → the Adam bug (loss stuck at 2.5, fixed by `eps` and `lr`) → Turkish corpus.
Documented in the README timeline.
