# 🐟 StonefishAI

**Sıfırdan yazılmış bir autograd motoru ve dil modeli altyapısı — Rust, SIFIR bağımlılık.**

```console
$ cargo tree
stonefish_ai v1.0.0
```

`Tensor`'dan topolojik backward'a, optimizer'lardan CUDA çekirdeklerine kadar her şey bu repoda elle yazılmıştır. **`[dependencies]` boştur.** CUDA sürücüsü, cuBLAS ve NVRTC çalışma zamanında dinamik olarak yüklenir (`LoadLibraryW` / `dlopen`) — `cudarc` bile yok, yalnızca işletim sisteminin kütüphane yükleyicisi kullanılır.

CUDA'sı olmayan makinelerde `--no-default-features` ile her şey CPU'da çalışır.

> English version: [README-Eng.md](README-Eng.md)

---

## İçindekiler

1. [Ne yapabiliyor / ne yapamıyor](#1-ne-yapabiliyor--ne-yapamıyor)
2. [Hızlı başlangıç](#2-hızlı-başlangıç)
3. [Donanım ve ölçülen sınırlar](#3-donanım-ve-ölçülen-sınırlar)
4. [Mimari](#4-mimari)
5. [Tasarım kararları](#5-tasarım-kararları)
6. [İki model hattı](#6-iki-model-hattı)
7. [Ölçülen sonuçlar](#7-ölçülen-sonuçlar)
8. [Zaman bütçesi profili](#8-zaman-bütçesi-profili)
9. [Testler](#9-testler)
10. [Geliştirme sırasında bulunan hatalar](#10-geliştirme-sırasında-bulunan-hatalar)
11. [Bilinen sınırlar](#11-bilinen-sınırlar)
12. [Yol haritası — yapılacaklar](#12-yol-haritası--yapılacaklar)
13. [Dizin yapısı](#13-dizin-yapısı)
14. [Sıfır bağımlılık nasıl çalışıyor](#14-sıfır-bağımlılık-nasıl-çalışıyor)

---

## 1. Ne yapabiliyor / ne yapamıyor

### ✅ Yapabildikleri

| Yetenek | Ölçü |
|---|---|
| El yazısı rakam sınıflandırma (MNIST) | **%98,12** test doğruluğu |
| Türkçe karakter-seviyesi metin üretimi | val CE **1,6230** (rastgele taban 4,91) |
| İngilizce BPE-16k dil modeli eğitimi | 236 M parametre, 450 M token, ~10.000 tok/s |
| Gradyan doğrulama | 29 op'un tamamı merkezi farkla sınanmış |
| GPU eğitimi | device-resident, 244.698 tok/s (küçük modelde) |
| Checkpoint yönetimi | config+vocab gömülü, bozuk dosyada panik yok, resume |

### ❌ Yapamadıkları (dürüst liste)

- **Sohbet/asistan değil.** Talimat takibi, soru-cevap, çeviri, özetleme, kod yazma yok.
- **Dünya bilgisi yok.** "Fransa'nın başkenti?" gibi sorulara cevap veremez.
- **Muhakeme yok.** Aritmetik, mantık zinciri, çok adımlı çıkarım yok.
- **Anlam yok.** Üretilen metin dilbilgisel olarak doğru ama anlamsızdır; model istatistiksel devam üretir.
- **Türkçe model 842 K parametre**, İngilizce model 236 M — ikisi de "yetersiz eğitilmiş" sınıfında (sırasıyla ~8,6 M ve ~450 M token görmüş).
- **KV-cache yok** → üretim O(n²); 300 token ≈ 2,8 s.
- **İngilizce model henüz doğrulanmadı** — eğitim tamamlanınca `en-eval` ile ölçülecek.

---

## 2. Hızlı başlangıç

```bash
# Derleme (CUDA'sız makinede: --no-default-features)
cargo build --release

# Tüm modları listele
./target/release/stonefish_ai help

# ⚡ TEK KOMUT DEMO — veri indirmeden, 90 saniyede sıfırdan eğitim + üretim
./target/release/stonefish_ai demo 90

# --- MNIST MLP ---
./target/release/stonefish_ai train          # eğit (20 epoch)
./target/release/stonefish_ai eval           # test setinde ölç → %98,12

# --- Türkçe GPT (karakter seviyesi) ---
./target/release/stonefish_ai gpt-train 6 8                    # epoch, batch
./target/release/stonefish_ai gpt-gen "Akşam erken iner" 300 0.8   # prompt, token, sıcaklık

# --- İngilizce GPT (BPE-16k, 236 M) ---
./target/release/stonefish_ai en-prepare        # BPE öğren + u16 cache üret
./target/release/stonefish_ai en-train 450000000   # token hedefi

# --- Ölçüm ---
./target/release/stonefish_ai test-gpt-profile  # zaman nereye gidiyor
./target/release/stonefish_ai test-gpu-native   # GPU-native eğitici hızı
./target/release/stonefish_ai test-matmul-bench # gerçek şekillerde CPU vs GPU
./target/release/stonefish_ai test-gpu-bench-big
```

### Veri gereksinimleri

| Dosya | Ne | Nereden |
|---|---|---|
| `data/train-images-idx3-ubyte` + 3 kardeşi | MNIST | [yann.lecun.com/exdb/mnist](http://yann.lecun.com/exdb/mnist/) |
| `data/turkce.txt` | Türkçe korpus (düz metin) | kendi metnin |
| `data/en/train.txt` | İngilizce korpus | TinyStories v2 (~2,1 GB) |
| `data/en/valid.txt` | İngilizce validasyon | TinyStories valid (~18 MB) |

`data/`, `*.bin`, `*.state`, `*.bpe` **repoda tutulmaz** (`.gitignore`).

TinyStories indirme komutu:
```powershell
$b = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
Invoke-WebRequest "$b/TinyStoriesV2-GPT4-train.txt" -OutFile data/en/train.txt
Invoke-WebRequest "$b/TinyStories-valid.txt"        -OutFile data/en/valid.txt
```

---

## 3. Donanım ve ölçülen sınırlar

Geliştirme/test ortamı: **RTX 5070 (12 GB), 32 GB DDR5, MSI Trident AS (≈10 L SFF)**, Windows.

| Sınır | Değer | Sonuç |
|---|---|---|
| VRAM | 12 GB | fp32 eğiticisiyle **~250 M parametre** tavanı |
| RAM | 32 GB | Akışlı veri hattı olmadan 3 B token **sığmaz** (u16 + disk cache bunu çözer) |
| Termal | SFF kasa | 250 W'da 80 °C → **210 W güç limiti** uygulandı, 76 °C'ye indi |
| Süre | tek GPU | 236 M model + 450 M token ≈ **12,5 saat** |

Ölçülen eğitim hızları:

| Model | Adım süresi | Throughput |
|---|---|---|
| 842 K (GPU-native) | 4,2 ms | **244.698 tok/s** |
| 842 K (CPU+cuBLAS) | 120,8 ms | 8.476 tok/s |
| **236 M (GPU-native)** | ~3,27 s | **~10.000 tok/s** |

---

## 4. Mimari

| Dosya | Satır | Sorumluluk |
|---|---|---|
| [src/lib.rs](src/lib.rs) | 35 | Kütüphane kökü, `VERSION`, clippy gerekçeli allow'lar |
| [src/sys/mod.rs](src/sys/mod.rs) | 190 | Dinamik kütüphane yükleyici (LoadLibraryW / dlopen) |
| [src/sys/cuda.rs](src/sys/cuda.rs) | 200 | CUDA sürücü API'si (bellek, bağlam, modül, çekirdek) |
| [src/sys/cublas.rs](src/sys/cublas.rs) | 80 | cuBLAS SGEMM + strided-batched |
| [src/sys/nvrtc.rs](src/sys/nvrtc.rs) | 130 | NVRTC — çalışma zamanında CUDA C derleme |
| [src/sys/dev.rs](src/sys/dev.rs) | 520 | `DevBuf`/görünümler, modül, çekirdek başlatma |
| [src/sys/selftest.rs](src/sys/selftest.rs) | 250 | FFI öz-testi (`test-ffi` modu) |
| [src/tensor.rs](src/tensor.rs) | 142 | `Matrix` (row-major 2D): matmul, transpose, broadcast add/sub |
| [src/autograd.rs](src/autograd.rs) | 845 | `Tensor`, `Op` (17 varyant), topolojik backward, tüm op'lar |
| [src/nn.rs](src/nn.rs) | 99 | `Linear`, MNIST model (de)serializasyonu |
| [src/transformer.rs](src/transformer.rs) | 246 | Embedding, MultiHeadSelfAttention, TransformerBlock, GPT, GptConfig |
| [src/checkpoint.rs](src/checkpoint.rs) | 147 | GPT checkpoint v2 (magic/version/config/vocab/tensörler) |
| [src/optim.rs](src/optim.rs) | 167 | SGD, Adam (bias correction'lı), `clip_grad_norm` |
| [src/loss.rs](src/loss.rs) | 7 | Softmax + cross-entropy |
| [src/data.rs](src/data.rs) | 79 | MNIST idx okuyucu, metin → sequence |
| [src/tokenizer.rs](src/tokenizer.rs) | 128 | Karakter seviyesi tokenizer |
| [src/bpe.rs](src/bpe.rs) | 236 | Byte-level BPE: merge öğrenimi + encode/decode |
| [src/corpus.rs](src/corpus.rs) | 251 | u16 token cache, akışlı pencere okuyucu |
| [src/gpu.rs](src/gpu.rs) | 362 | cuBLAS GEMM (düz/transposed/device/strided-batched) + CPU yedeği |
| [src/gpu_kernels.rs](src/gpu_kernels.rs) | 405 | NVRTC derleme + PTX önbelleği + 15 çekirdek sarmalayıcısı |
| [src/gpu_train.rs](src/gpu_train.rs) | 1289 | Device-resident GPT eğiticisi (hızlı yol) |
| [src/en_train.rs](src/en_train.rs) | 270 | İngilizce hat: `prepare` + `train` |
| [src/profile.rs](src/profile.rs) | 119 | Hafif profil sayaçları |
| [src/main.rs](src/main.rs) | 1252 | CLI (20 mod) + eğitim döngüleri + `Trainer` soyutlaması |
| [kernels/stonefish.cu](kernels/stonefish.cu) | 389 | CUDA çekirdekleri |

Toplam: **~7.500 satır Rust** + 389 satır CUDA, **0 bağımlılık**.

---

## 5. Tasarım kararları

### 5.1 Batch: satır paketleme (Tensor3D yok)

`(batch, seq, d_model)` tensörü **`(batch·seq, d_model)` matrisi** olarak tutulur:

```
(B, S, D) ──paketle──▶ (B·S, D)
```

LayerNorm, Linear, GELU, bias-add ve softmax zaten **satır bazlı** olduğu için bu dönüşüm bedava. Tek istisna attention'dır ve o da `Op::CausalAttention` içinde `batch`/`seq`/`n_heads` bilgisiyle çözülür.

**Sonuç:** iki paralel API (2D autograd + 3D'siz tip) yerine **tek motor**, gerçek batch, tek `backward()`. Eski `Tensor3D` tipi tamamen kaldırıldı.

### 5.2 Fused causal multi-head attention

`Tensor::causal_attention(q, k, v, batch, seq, n_heads)` **tek bir düğümdür**:

- `d_model`'i `n_heads` dilime böler (`d_head = d_model/n_heads`)
- `1/√d_head` ölçeği
- nedensel maske
- head başına softmax
- **hepsinin backward'ı**

Attention ağırlıkları `(batch·n_heads·seq, seq)` olarak cache'lenir; dQ/dK/dV tek geçişte üretilir. Ölçek ve maske op'un **içinde** olduğu için dışarıda yanlış birleştirme yapılamaz.

### 5.3 Row-major GEMM → cuBLAS (transpose hilesi)

cuBLAS kolon-major çalışır. Row-major `C(m,n) = A·B` için:

```
C ᵀ = B ᵀ · A ᵀ   →  operandlar yer değişir, transa/transb ile ifade edilir
```

`dev_gemm_batched` bu eşlemeyi `Nn/Nt/Tn` modları için yapar; `lda` = **B'nin satır adımı**, `ldb` = **A'nın satır adımı**, `ldc` = C'nin satır adımı. Bu sayede transpose **malzemeleştirilmez** (eski kod her backward'da `(1024×128)` strided kopya alıyordu).

### 5.4 CUDA çekirdekleri: nvcc değil **NVRTC**

Bu makinede host C++ derleyicisi (`cl.exe`) yok; `nvcc` bu yüzden çalışmıyor. **NVRTC** yalnız cihaz kodu derlediği için kullanılabiliyor:

- Çalışma zamanında derleme (~100-300 ms, tek sefer)
- PTX `target/gpu-cache/` altında saklanır (anahtar: kaynak + arch + sürüm)
- Arch cihazdan okunur (`compute_120` → RTX 5070)

### 5.5 İki eğitici, tek arayüz

| | CPU + cuBLAS | GPU-native |
|---|---|---|
| Ne | autograd grafiği | elle yazılmış ileri/geri |
| Nerede | `main.rs` + `autograd.rs` | `gpu_train.rs` |
| Doğrulama | **referans** (gradyan testleri) | eşdeğerlik testleriyle CPU'ya bağlı |
| Hız | 8.476 tok/s | **244.698 tok/s** |

CUDA varsa `gpt-train` GPU yolunu otomatik seçer; yoksa CPU yoluna düşer.

### 5.6 Tokenizer: önce karakter, sonra BPE

Karakter seviyesinde model kapasitesinin büyük kısmı **heceleme** öğrenmeye gider. Byte-level BPE-16k ile aynı pencere **4-5× daha fazla metin** kapsar (ölçüldü: 4,7 karakter/token).

BPE öğrenimi **artımlı çift sayımı** ile yapılır: her merge'den sonra yalnız o çifti içeren kelimeler taranır → 16k merge **7 saniyede** (naif yaklaşım saatler sürüyordu).

### 5.7 Veri hattı: u16 + disk cache

`Vec<usize>` = token başına 8 bayt → 3 B token **24 GB RAM**. Bunun yerine:

- Tokenize edilmiş korpus diske **u16** olarak yazılır (2 bayt/token)
- Eğitim 64 MB'lık **bellek penceresiyle** okur, tüm dosya yüklenmez
- Tokenize: **148 MB/s** (kelime önbellekli)

---

## 6. İki model hattı

### 6.1 Türkçe (karakter seviyesi)

| | |
|---|---|
| Model | `d_model=128`, 4 katman, 4 head (`d_head=32`), ctx=128 |
| Vocab | 136 karakter |
| Parametre | **842.632** |
| Korpus | `data/turkce.txt` — 2.870.158 karakter, 22.423 sequence |
| Checkpoint | `stonefish_gpt.bin` (v2: magic + config + vocab + tensörler) |
| Eğitim | 6 epoch, 16.620 adım, batch 8 |

Eğitim eğrisi (GPU-native, 70 saniye):

| Epoch | train CE | val CE |
|---|---|---|
| 0 | 2,1881 | 1,8943 |
| 1 | 1,7973 | 1,7624 |
| 2 | 1,6868 | 1,6939 |
| 3 | 1,6145 | 1,6538 |
| 4 | 1,5591 | 1,6309 |
| 5 | **1,5215** | **1,6230** |

Tekrar üretilebilirlik: hiperparametreler + seed'ler + epoch metrikleri `runs/<ts>.json` manifestine yazılır.

### 6.2 İngilizce (BPE-16k) — **aktif geliştirme**

| | |
|---|---|
| Model | `d_model=1024`, 16 katman, 8 head (`d_head=128`), ctx=512 |
| Vocab | **16.384** (BPE, byte-level) |
| Parametre | **235.571.200** |
| Korpus | TinyStories v2: **450.873.533** train + 3.931.894 valid token |
| Sıkıştırma | 4,7 karakter/token |
| Batch | 4 × gradyan biriktirme 16 = efektif 64 dizi (32.768 token/adım) |
| Plan | 13.732 adım, max_lr=6e-4, 500 adım warmup + cosine, clip=1,0 |
| VRAM | 8,4 GB / 12 GB |
| Hız | ~10.000 tok/s → **~12,5 saat** |
| Checkpoint | `stonefish_en.state` (`SFEN001`, her 500 adım) |

**Eğitim tamamlandı (2026-11-17):** 13.732 adım, **745 dakika**, ortalama 10.061 tok/s.

**Ölçülen kalite (valid.bin, 3.928.064 token):**

| Metrik | Değer |
|---|---|
| val loss | **2,3700** nat/token |
| perplexity | **10,70** |
| **BPC** | **0,6913 bit/karakter** |
| train loss (son) | ~1,95 (val ile fark 0,42 → hafif ezber) |
| rastgele taban | 9,70 |

**Üretim örneği (sıcaklık 0, açgözlü):**

> Once upon a time-out lived a little girl named Lily. She loved to play with her toys and make up stories. One day, she found a big box in her room. She was very excited and wanted to see what was inside.
> Lily opened the box and found a lot of fun toys inside. She played with the toys and had a lot of fun. But then, she heard a loud noise outside. She looked out the window and saw a big, scary monster! The monster was very big and scary. Lily was scared and ran back to her room.

**Diyalog formatı (sıcaklık 0,8):**

> "Yes! I love to play with my car," Tommy said.
> "But I need your toy car," Jane said.

Model `<|endoftext|>` hikâye sınırını ve tırnaklı diyalog düzenini kendiliğinden öğrenmiş.

---

## 7. Ölçülen sonuçlar

### 7.1 MNIST MLP (784→128→10)

SGD lr=0,01, batch 64, 20 epoch → **%98,12** test doğruluğu (10.000 görüntü, 196 hata).

### 7.2 GPU vs CPU matmul (RTX 5070, ısınma sonrası)

| GEMM | CPU | GPU | Hızlanma |
|---|---|---|---|
| `(1024×128)@(128×128)` attn proj | 1,119 ms | 0,189 ms | **5,9×** |
| `(1024×128)@(128×512)` ff1 | 5,697 ms | 0,975 ms | **5,8×** |
| `(1024×512)@(512×128)` ff2 | 5,310 ms | 0,284 ms | **18,7×** |
| `(1024×128)@(128×136)` lm head | 1,150 ms | 0,210 ms | **5,5×** |
| `(128×128)@(128×128)` üretim (B=1) | 0,132 ms | 0,055 ms | **2,4×** |

1024³: CPU 2,16 s / GPU-sadece **67,5 µs** / transfer+matmul 2,23 ms.

### 7.3 Uçtan uca eğitim (6 epoch, aynı model)

| Yol | Adım süresi | Throughput | 6 epoch |
|---|---|---|---|
| CPU + cuBLAS (autograd) | 120,1 ms | 8.476 tok/s | **54,4 dk** |
| **GPU-native (device-resident)** | **4,2 ms** | **244.698 tok/s** | **70 s** |

**46,7× hızlanma.** Val CE: CPU 1,6337 / GPU 1,6230 (fast-math + toplama sırası).

### 7.4 İngilizce 236 M

~10.000 tok/s, 8,4 GB VRAM, 76 °C (210 W limitinde), GPU %99 — 11 saat boyunca hız düşmedi.
Kayıp eğrisi: 9,70 (rastgele) → **~1,97** (%90'da, hâlâ düşüyor).

---

## 8. Zaman bütçesi profili

`test-gpt-profile` (842 K model, gerçek konfig, 10 adım ortalaması):

| Kalem | ms/adım | pay |
|---|---|---|
| attention fwd+bwd (CPU) | 35,3 | 29,5 % |
| matmul **transfer** (CPU↔GPU) | 21,1 | 17,6 % |
| matmul **gerçek GEMM** | **1,0** | **0,8 %** |
| GELU fwd+bwd | 17,7 | 14,8 % |
| LayerNorm fwd+bwd | 7,4 | 6,1 % |
| ara tensör tahsisi | 6,0 | 5,0 % |
| adam.step / CE / zero_grad | 3,4 | 2,9 % |
| diğer (tahsis, paketleme, residual) | ~27 | ~23 % |
| **toplam** | **~120** | 100 % |

**Kritik çıkarım:** adımda 75 matmul çağrısı var ama gerçek GEMM toplamı 1 ms. Yani "GEMM'leri GPU'da tut" (transferi sıfırlamak) üst sınır %17 kazandırır — asıl kazanç **hesabı da GPU'ya taşımaktan** gelir. GPU-native yol tam bunu yapar ve 46,7× verir.

---

## 9. Testler

**62 test**, iki feature config'inde de yeşil:

| Grup | Adet | Ne doğrular |
|---|---|---|
| [tests/gradcheck.rs](tests/gradcheck.rs) | 29 | Her op için **merkezi fark** ile analitik ↔ numerik gradyan |
| [tests/gpu_kernels.rs](tests/gpu_kernels.rs) | 12 | 15 CUDA çekirdeği + device GEMM modları, CPU referansına karşı |
| [tests/gpu_train_equiv.rs](tests/gpu_train_equiv.rs) | 2 | GPU eğiticisi ↔ CPU autograd: **tüm gradyanlar** + 5 Adam adımının loss eğrisi |
| [tests/checkpoint.rs](tests/checkpoint.rs) | 5 | Roundtrip, bozuk magic, yarım dosya, vocab uyuşmazlığı, drift |
| Birim testleri | 14 | Tokenizer (4), optimizer/clip (4), BPE (4), corpus (2) |

Gradyan kapsamı: `add`/`sub` (broadcast dahil), `mul_elem`, `matmul` (kare dahil), `matmul+transpose`, `scale`, `relu`, `sigmoid`, `transpose`, `sum`, `softmax_ce`, `softmax_ce_ids` (seyrek), `softmax`, `mask_causal`, `gelu`, `layer_norm`, `embedding`, `causal_attention` (tek/çok head + batch), uçtan uca küçük GPT, nedensellik sözleşmesi, batch gradyanı = Σ tek-sequence gradyanı.

```bash
cargo test --release                       # CUDA yolu
cargo test --release --no-default-features  # CPU yolu
```

---

## 10. Geliştirme sırasında bulunan hatalar

Bu liste projenin en değerli kısmı: **her biri testlerle yakalandı.**

| # | Hata | Nasıl bulundu |
|---|---|---|
| 1 | `mask_causal` testi: `-1e9` maskesi loss'u 1e9'a çıkarıyor, `f32` ulp(1e9)≈64 `ΔL≈1e-3`'ü yutuyor | Gradyan testi (testin kendisi hatalıydı) |
| 2 | Gradyan testinde `eps` kalibrasyonu: embedding init 0,02 + LayerNorm → `eps=1e-3` satırın %10'u, doğrusallık çöküyor | `grad_probe` ile parametre bazlı ölçüm |
| 3 | Batch gradyan kimliğinde yanlış faktör (B³) | `L_batch = (1/B)·ΣL_b` türetilip düzeltildi |
| 4 | Attention backward'ında `dW` için `dOut` yerine `q` kullanılmış | Vektörleştirme sonrası 3 gradyan testi kırmızı |
| 5 | **Head adımı**: skor buffer'ında offset `h·S²` olmalı sanıldı, doğrusu `h·B·S²` | Skorlar 0 çıkıyordu → softmax 0,5/0,5 |
| 6 | **Residual gradyanı**: attention dalının yukarı akışı `dOut` değil `dOut + dx_ln2` (`res1 = x + proj`) | dAttn/dWo baştan yanlıştı |
| 7 | `dev_gemm`'de `beta` yanlışlıkla 1,0 → sonuç önceki içerikle toplanıyor | GEMM mod testi |
| 8 | BPE encode tüm 64 MB yığını işliyordu (her merge'de tüm vektör → O(n²)) | Tokenize "donuyordu"; kelime önbelleğine çevrildi |
| 9 | İki süreç aynı çıktı dosyasına yazıyordu ("Access denied") | Öldürülen işin çocuk süreci yaşamıştı |
| 10 | 13:33'te başlamış **yetim eğitim süreci**, bitince checkpoint'i eski (v1) formatla ezecekti | Süreç envanteri |
| 11 | `cargo clippy --fix` tüm `data.rs`'i satır sonu değişikliği yüzünden yeniden yazdı | Geri alındı, `.gitattributes` (LF) eklendi |

---

## 11. Bilinen sınırlar

- **KV-cache yok** — üretimde her token için tüm bağlam yeniden hesaplanır (O(n²)).
- **Üretimde konum kayması** — pencere kaydıkça `pos_emb` yeniden 0'dan uygulanır; eğitimdeki mutlak konumlarla tam örtüşmez (RoPE bunu çözer).
- **fp32 eğitim** — bf16 yok; VRAM tavanını ~250 M'da tutuyor.
- **Gradyan checkpointing yok** — aktivasyonlar saklanır, daha büyük model için gerekli.
- **Tek thread CPU yolu** — `rayon` yok; CPU yolu referans/yedektir.
- **Türkçe model 17 M token işledi** ama korpus yalnız 2,87 M token (3,4 benzersiz token/param) → çeşitlilik dar, epoch tekrarı yüksek.
- **İngilizce model 450 M token** görmüş (1,9 token/parametre) → Chinchilla'nın 1/10'u, yani **yetersiz eğitilmiş**.
- **SFT/DPO yok** → model asistan davranışı göstermez.
- **İngilizce hat için val/generation aracı henüz yok** (`en-eval`, `en-gen` yol haritasında).
- Ağırlık paylaşımı (weight tying), dropout, RoPE, SwiGLU, RMSNorm yok.

---

## 12. Yol haritası — yapılacaklar

### Faz A — İngilizce modeli doğrula *(bitişte, ~1 saat)*

- [ ] **A1 `en-eval`**: `valid.bin` (3,9 M token) üzerinde val loss, **perplexity**, **BPC**
  - BPC = loss[token] / (ln2 × karakter/token) → tokenizer'dan bağımsız tek karşılaştırma metriği
  - Trend: 450 M token'dan sonra `loss 1,7–2,0` bandında bekleniyor
- [ ] **A2 `en-gen`**: BPE encode → kayan pencere forward → sıcaklık/top-k → decode
- [ ] **A3 örnek rapor**: ham üretim çıktıları (temperature 0 ve 0,8)

### Faz B — Veri büyütme *(en yüksek getiri)*

- [ ] **B1** Project Gutenberg (~3-5 B token) indir + temizle (başlık/son söz ayıklama)
- [ ] **B2** Karışım: %60 TinyStories + %40 Gutenberg → 2-4 B token
- [ ] **B3** Yeniden tokenize → ~6-8 GB u16 cache
- [ ] **B4** Aynı mimariyle 2-4 B token eğitimi (~2-3 gün)
  - Hedef: 8-16 token/parametre → belirgin kalite sıçraması

### Faz C — Mimari iyileştirmeler

- [ ] **C1 KV-cache**: üretimde 10-50× hızlanma (O(n²) → O(n))
- [ ] **C2 RoPE**: öğrenilmiş mutlak konum yerine döner konum → uzun bağlam, konum kayması biter
- [ ] **C3 RMSNorm** (LayerNorm yerine), **SwiGLU** (FFN), **weight tying** (gömme ↔ çıkış), **dropout**
- [ ] **C4 bf16 karışık hassasiyet**: 2-3× hız + daha büyük model (çekirdeklerde `__half` desteği)
- [ ] **C5 Bellek-verimli/birleşik attention** (flash-attention tarzı)
- [ ] **C6 Gradyan checkpointing**: aktivasyonları yeniden hesapla → 300-500 M modele yer aç

### Faz D — Ölçek

- [ ] **D1** 300 M (d=1024, L=20) denemesi — VRAM ölçümüyle
- [ ] **D2** ctx 1024 (şu an 512)

### Faz E — Hizalama (asistan davranışı)

- [ ] **E1** SFT veri seti: 10-50 k `(talimat, cevap)` çifti *(asıl zor kısım)*
- [ ] **E2** Chat şablonu (`<|user|>… <|assistant|>…`) + SFT ince ayarı
- [ ] **E3** DPO / tercih eğitimi (üslup, yardımseverlik, reddetme)
- [ ] **E4** Talimat takibi değerlendirme seti

### Faz F — Mühendislik borcu

- [ ] **F1** Checkpoint **v3**: BPE merge tablosunu da checkpoint'e göm (şu an `SFEN001` ayrı)
- [ ] **F2** `en-eval`/`en-gen` için otomatik testler
- [ ] **F3** Periyodik checkpoint'i Türkçe yola da ekle
- [ ] **F4** README'leri Faz A-E ilerledikçe güncelle

### Beklenti tablosu

| Faz sonrası | Ne elde edilir |
|---|---|
| A | Ölçülmüş kalite + ilk gerçek İngilizce örnekler |
| B | Akıcı, tema tutarlı, 2-3 paragraf |
| C | Aynı donanımda ~2× hız, uzun bağlam, daha az VRAM |
| D | Kalite tavanı (donanım sınırı) |
| E | **Soru-cevap yapan** asistan (bilgi yok, format var) |

> **Dürüst uyarı:** Tek RTX 5070 ile ulaşılabilir son nokta ~250-300 M parametre ve ~4 B token'dır. Bu, "etkileyici bir sıfırdan-yazılmış motor + gerçek bir dil modeli" demektir; "kullanılabilir asistan" değil. Gerçek asistan için 1 B+ parametre ve ~20 B token gerekir → bulut GPU (8×A100 ile ~2 gün, ~$700) veya hazır bir temel modeli ince ayar.

---

## 13. Dizin yapısı

```
StonefishAI/
├── Cargo.toml            # sürüm 0.7.0, cuda feature (varsayılan açık)
├── README-Tr.md          # bu dosya
├── README-Eng.md         # İngilizce
├── kernels/
│   └── stonefish.cu      # 15 CUDA çekirdeği (NVRTC ile derlenir)
├── src/                  # 18 modül (~6.300 satır)
├── tests/                # 4 dosya, 62 test
├── examples/
│   ├── bpe_bench.rs      # BPE tokenizasyon hız ölçümü
│   └── grad_probe.rs     # parametre bazlı gradyan teşhisi
├── data/                 # (git dışı) MNIST + korpuslar
└── runs/                 # (git dışı) eğitim manifestleri
```

**Motor ilkeleri:** sıfırdan yazılmış · gradyanları sayısal doğrulanmış · ölçümle karar verilmiş · her hata testle kilitlenmiş · CUDA'sız da çalışır.

---

## 14. Sıfır bağımlılık nasıl çalışıyor

`[dependencies]` boş. GPU'ya erişim tamamen kendi FFI katmanımızla yapılır:

```
src/sys/mod.rs    LoadLibraryW (Windows) / dlopen (Linux) + GetProcAddress
                  → fonksiyon işaretçileri tembel (OnceLock) yüklenir
src/sys/cuda.rs   cuInit, cuCtxCreate, cuMemAlloc/HtoD/DtoH/Memset,
                  cuModuleLoadData, cuModuleGetFunction, cuLaunchKernel
src/sys/cublas.rs cublasCreate, cublasSetStream, cublasSgemm,
                  cublasSgemmStridedBatched
src/sys/nvrtc.rs  nvrtcCreateProgram, nvrtcCompileProgram, nvrtcGetPTX
```

**Faydaları:**

1. **Derleme zamanında CUDA SDK gerekmez** — import kütüphanesi (`.lib`) yok, `build.rs` yok
2. **CUDA yoksa zarif düşüş** — `LoadLibraryW` `None` döner, her şey CPU'da çalışır
3. **Denetlenebilir** — 1.100 satır FFI, 400 satırlık bir crate'in soyutlaması yerine doğrudan API
4. **Doğrulanmış** — `stonefish_ai test-ffi` alloc + kopyalama + PTX yükleme + çekirdek başlatma + cuBLAS SGEMM'i CPU referansına karşı sınar (ölçülen fark: **0.0**)

Çekirdek argümanları `Kargs` ile hazırlanır: her argüman bir `u64` slota yazılır (cihaz işaretçisi 8 bayt, i32/f32 ilk 4 bayt) ve `cuLaunchKernel`'e o slotların adresleri verilir — tahsis yok, sıfır kopya.
