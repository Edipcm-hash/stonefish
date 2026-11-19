# 🐟 StonefishAI

From-scratch autograd engine + language-model stack in Rust, **zero dependencies**.

- 🇹🇷 **Turkish documentation:** [README-Tr.md](README-Tr.md)
- 🇬🇧 **English documentation:** [README-Eng.md](README-Eng.md)

At a glance:

| | |
|---|---|
| English GPT (v2, released) | **222,331,904** parameters · 2.05 B tokens |
| MNIST MLP | **98.14 %** test accuracy |
| Training speed | **10,732 tok/s** at 222 M on one RTX 5070 |
| GPU-native trainer | **244,698 tok/s** on the 842 K model (28.6× per step) |
| Tests | **75** — 32 gradient · 16 CUDA kernel · 2 equivalence · 2 KV-cache · 5 checkpoint · 18 unit |
| Dependencies | **none** — `cargo tree` is one line |
| Turkish GPT (v0.5, historical) | 842,632 parameters, val CE **1.6230** |

```bash
cargo build --release
./target/release/stonefish_ai help

# one command: train a small model from scratch and generate, in 90 s
./target/release/stonefish_ai demo 90
```

Not a chat assistant: the model continues English text and writes invented facts
fluently. Both READMEs say exactly what it cannot do.

```console
$ cargo tree
stonefish_ai v1.0.0
```

Everything from `Tensor` to the topological backward pass, the optimisers and the
CUDA kernels is written in this repository. The CUDA driver, cuBLAS and NVRTC are
loaded at runtime through `LoadLibraryW` / `dlopen` — not even `cudarc`, just the
operating system's library loader.

Without CUDA, everything runs on the CPU with `--no-default-features`.
