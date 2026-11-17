# 🐟 StonefishAI

Sıfırdan yazılmış autograd motoru + dil modeli altyapısı (Rust).

- 🇹🇷 **Türkçe dokümantasyon:** [README-Tr.md](README-Tr.md)
- 🇬🇧 **English documentation:** [README-Eng.md](README-Eng.md)

Hızlı bakış:

| | |
|---|---|
| MNIST MLP | **%98,12** test doğruluğu |
| Türkçe GPT | 842.632 parametre, val CE **1,6230** |
| İngilizce GPT | **235.571.200** parametre, 450 M token, ~10.000 tok/s |
| Eğitim hızı | 842 K modelde **244.698 tok/s** (GPU-native, 46,7× hızlı) |
| Test | **62** (29 gradyan + 12 CUDA çekirdeği + 2 eşdeğerlik + 5 checkpoint + 14 birim) |
| Bağımlılık | yalnız `cudarc` (feature flag ile kapatılabilir) |

```bash
cargo build --release
./target/release/stonefish_ai help
```
