# 🐟 StonefishAI

From-scratch autograd engine + language-model stack (Rust).

- 🇹🇷 **Turkish documentation:** [README-Tr.md](README-Tr.md)
- 🇬🇧 **English documentation:** [README-Eng.md](README-Eng.md)

At a glance:

| | |
|---|---|
| MNIST MLP | **98.12 %** test accuracy |
| Turkish GPT | 842,632 parameters, val CE **1.6230** |
| English GPT | **235,571,200** parameters, 450 M tokens, ~10,000 tok/s |
| Training speed | **244,698 tok/s** on the 842 K model (GPU-native, 46.7x faster) |
| Tests | **62** (29 gradient + 12 CUDA kernel + 2 equivalence + 5 checkpoint + 14 unit) |
| Dependencies | none - see below |

```bash
cargo build --release
./target/release/stonefish_ai help
```
