# 🐟 StonefishAI

Sıfırdan yazılmış bir **autograd motoru + mini GPT** — Rust, harici ML kütüphanesi yok.
`Tensor`, topolojik sıralı backward, optimizer'lar, karakter tokenizer'ı ve transformer
bloklarının tamamı bu repoda elle yazılmıştır. Tek harici bağımlılık GPU yolunu açan
`cudarc`'tır (feature flag ile kapatılabilir).

İki şeyi yapar:

1. **MNIST MLP** — 784→128→10, SGD. Test doğruluğu **%98.12**.
2. **Karakter seviyesi GPT** — 842.632 parametre, Türkçe şiir korpusunda eğitilir, metin üretir.

---

## Hızlı başlangıç

```bash
# Derleme (CUDA'sız makinede: --no-default-features)
cargo build --release

# Modları listele
./target/release/stonefish_ai help

# MNIST: eğit → değerlendir
./target/release/stonefish_ai train
./target/release/stonefish_ai eval

# GPT: eğit → üret       (epoch, batch, max_steps, max_lr)
./target/release/stonefish_ai gpt-train 6 8
./target/release/stonefish_ai gpt-gen "Akşam erken iner" 300 0.8

# Zaman nereye gidiyor?
./target/release/stonefish_ai test-gpt-profile
```

### Veri

`data/` klasörü repoda tutulmaz (`.gitignore`). Beklenen dosyalar:

| Dosya | Ne | Nereden |
|---|---|---|
| `data/turkce.txt` | GPT eğitim korpusu (düz metin, UTF-8) | kendi metnin |
| `data/train-images-idx3-ubyte` vb. | MNIST (4 dosya) | [yann.lecun.com/exdb/mnist](http://yann.lecun.com/exdb/mnist/) |

`.gz` dosyaları açılmış halleriyle birlikte duruyorsa `.gz` olanları silebilirsin.

---

## Tasarım kararları

### Tek motor, satır paketleme ile batch

Batch için ayrı bir 3D tensor tipi **yok**. `(batch, seq, d_model)` tensörü
`(batch*seq, d_model)` matrisi olarak tutulur:

```
(B, S, D)  ──paketle──▶  (B*S, D)
```

LayerNorm, Linear, GELU, bias-add ve softmax zaten **satır bazlı** olduğu için bu
dönüşüm bedava. Tek istisna attention'dır: o da `Op::CausalAttention` içinde
`batch`/`seq`/`n_heads` bilgisiyle çözülür. Sonuç: iki paralel API yerine tek bir
autograd motoru, gerçek batch gradyanı, tek `backward()` çağrısı.

### Fused causal multi-head attention

`Tensor::causal_attention(q, k, v, batch, seq, n_heads)` tek bir düğümdür. İçinde:

* `d_model`'i `n_heads` dilime böler (her head `d_head = d_model/n_heads`),
* `1/sqrt(d_head)` ölçeği,
* nedensel maske,
* head başına softmax,
* ve **hepsinin backward'ı**.

Attention ağırlıkları `(batch*n_heads*seq, seq)` olarak cache'lenir; dQ/dK/dV tek
geçişte üretilir. Ölçek ve maske op'un içinde olduğu için dışarıda yanlış
birleştirme yapılamaz.

### Gradyanlar sayısal olarak doğrulanmış

Elle türetilmiş her backward, merkezi farkla (central difference) sınanır:
**36 test** — 27 gradyan/op testi, 5 checkpoint testi, 4 tokenizer birim testi.

```bash
cargo test --release                      # GPU yolu
cargo test --release --no-default-features # CPU yolu
```

Gradyan kapsamı: `add`/`sub` (broadcast dahil), `mul_elem`, `matmul`, `scale`,
`relu`, `sigmoid`, `transpose`, `sum`, `softmax_ce`, `softmax_ce_ids`, `softmax`,
`mask_causal`, `gelu`, `layer_norm`, `embedding`, `causal_attention`
(tek/çok head, batch'li), uçtan uca küçük GPT, GPU↔CPU matmul eşdeğerliği,
nedensellik sözleşmesi, ve batch gradyanı = Σ tek-sequence gradyanı.

Checkpoint kapsamı: roundtrip (config+vocab+ağırlıklar birebir), yanlış magic,
yarım dosya, vocab uyuşmazlığı, vocab drift tespiti.

> Not: `gradcheck`'te `eps`, parametrenin ölçeğine göre seçilmeli. Embedding init
> ölçeği 0.02 iken `eps=1e-3` LayerNorm'un std'sine göre satırın %10'u demektir ve
> merkezi farkın doğrusallık varsayımı çöker — testte bu yüzden embedding'ler
> ölçeklenir (bkz. `tiny_gpt_end_to_end` yorumu).

### Checkpoint v2 — kendini tanımlayan dosya

Eski format çıplak tensörlerden oluşuyordu: config yok, vocab yok, magic yok.
Korpus dosyasındaki tek karakter değişikliği ID'leri kaydırıp modeli sessizce
bozuyordu. Yeni format (`src/checkpoint.rs`):

```
magic "SFGPT002" | version | config (vocab, max_seq, d_model, layers, heads)
| vocab (karakter kodları) | tensörler (rows, cols, f32 veri)
```

Yüklemede her tensörün **şekli** ve sayısı doğrulanır; uyuşmazlıkta `InvalidData`
hatası döner (panik yok). Vocab gömülü olduğu için `gpt-gen` için korpus dosyası
gerekmez. Ayrıca `stonefish_gpt.vocab` yan dosyası yazılır — korpus drift'i
tespiti için.

---

## Performans

### GPU gerçekten kullanılıyor

CPU "gerçeğin kaynağı"dır ve her matmul için CPU→GPU→CPU turu yapılır. Bu ancak
GEMM büyüdükçe kârlı olur — batch'leme tam da bunu sağlar. Ölçüm
(`test-matmul-bench`, RTX 5070, ısınma sonrası):

| GEMM | CPU | GPU | Hızlanma |
|---|---|---|---|
| `(1024×128)@(128×128)` attn proj | 1.119 ms | 0.189 ms | **5.9×** |
| `(1024×128)@(128×512)` ff1 | 5.697 ms | 0.975 ms | **5.8×** |
| `(1024×512)@(512×128)` ff2 | 5.310 ms | 0.284 ms | **18.7×** |
| `(1024×128)@(128×136)` lm head | 1.150 ms | 0.210 ms | **5.5×** |
| `(128×128)@(128×128)` üretim (B=1) | 0.132 ms | 0.055 ms | **2.4×** |

Backward matmul'ları da GPU'ya gider (eskiden CPU'daydı: "GPU eğitimi"nin yarısı
CPU'da koşuyordu). GEMM çıktıları boyuta göre geri dönüştürülür (`gpu::POOL`),
böylece her çağrıda yeni device buffer ayrılmaz.

### Eğitim hızı

| | eski (efektif batch=1) | yeni (gerçek batch=8) |
|---|---|---|
| küçük model (d=64, 2 katman, S=64) | 5.714 tok/s | **24.138 tok/s** |
| gerçek model (d=128, 4 katman, S=128) | — | **~8.500 tok/s** |

`test-gpt-gpu` forward/backward dağılımını da basar.

### Zaman nereye gidiyor? (`test-gpt-profile`)

Gerçek konfigürasyonda bir eğitim adımının bütçesi (d=128, 4 katman, S=128, batch=8,
RTX 5070, 10 adım ortalaması, ısınma hariç):

| Kalem | ms/adım | pay |
|---|---|---|
| attention fwd + bwd (CPU) | 35.3 | 29.5 % |
| matmul — **transfer** (CPU↔GPU) | 21.1 | 17.6 % |
| matmul — **gerçek GEMM** | 1.0 | **0.8 %** |
| GELU fwd + bwd | 17.7 | 14.8 % |
| LayerNorm fwd + bwd | 7.4 | 6.1 % |
| ara tensör tahsisi (`from_op`) | 6.0 | 5.0 % |
| adam.step | 2.0 | 1.7 % |
| cross-entropy fwd + bwd | 1.0 | 0.9 % |
| `zero_grad` | 0.4 | 0.3 % |
| diğer (tahsis/serbest bırakma, paketleme, residual'lar) | ~27 | ~23 % |
| **toplam** | **~121** | 100 % |

Adımda **75 matmul çağrısı** var ama gerçek GEMM toplamı 1 ms. Bu ölçüm, "her şeyi
GPU'ya taşı" içgüdüsünü çürütüyor:

* Yalnız **device-resident** yapmak (transferi sıfırlamak) üst sınır **%17** kazandırır.
* Adımın **%82'si CPU'da** geçiyor (attention + elementwise + tahsis).
* Gerçek hızlanma için attention/softmax/LN/GELU/Adam'ın CUDA çekirdeklerine
  taşınması gerekir — bu, bu motora bir *device tensor katmanı* eklemek demek.

### Bu turda ölçülerek yapılan optimizasyonlar

| Değişiklik | Etki |
|---|---|
| `backward()`'ta düğüm başına `upstream` klonunu kaldır (ödünç al) | 177 → 161 ms |
| Matmul geri yayılımında transpose'u cuBLAS'a yaptır (`matmul_bt`/`matmul_ta`) | 161 → 148 ms |
| Attention backward'ını dilim+zip tabanlı vektörleştir | 148 → 120 ms |
| **Toplam** | **177 → 120 ms/adım (+%47 tok/s)** |

Attention backward'ı 47.9 ms'den 19.8 ms'ye indi. Her adım
`cargo test --release` (28 gradyan testi) ile doğrulandı; vektörleştirme sırasında
`dW`'de `dOut` yerine `q` kullanma hatası **gradyan testleri tarafından yakalandı**.

---

## Ölçülen sonuçlar

| Model | Metrik | Değer |
|---|---|---|
| MNIST MLP (784→128→10, SGD lr=0.01, 20 epoch) | test doğruluğu | **%98.12** |
| GPT (842.632 param, 6 epoch) | en iyi **val CE** | **1.6337** |
| GPT | son train CE | 1.5264 |

### Eğitim prosedürü

Koşu tekrar üretilebilir: hiperparametreler, seed'ler ve epoch metrikleri
`runs/<timestamp>.json` manifestine yazılır.

| Bileşen | Değer |
|---|---|
| Mimari | `d_model=128`, 4 katman, 4 head (`d_head=32`), `seq_len=128`, vocab 136 |
| Veri | 2.870.158 karakter → 22.423 sequence (22.167 train / 256 val) |
| Batch | 8 sequence (1.024 token/adım) |
| Optimizer | Adam β=(0.9, 0.999), `max_lr=1e-3` |
| LR programı | 150 adım doğrusal warmup → cosine ile 0.1× |
| Gradyan kırpma | global L2 norm, `clip_norm=1.0` |
| Checkpoint | her epoch sonunda **en iyi val** → `stonefish_gpt.bin`; nihai durum → `stonefish_gpt_final.bin` |
| Seed | model `0xDEADBEEF`, shuffle `0xA5A51234` |
| Toplam | 6 epoch × 2.771 adım = **16.626 adım**, 54.4 dk, ~5.218 tok/s (RTX 5070) |

### GPT eğitim eğrisi

| Epoch | train CE | val CE | süre | son LR | kırpılan adım |
|---|---|---|---|---|---|
| 0 | 2.1924 | 1.9060 | 521 s | 9.5e-4 | %80 |
| 1 | 1.8026 | 1.7729 | 525 s | 7.8e-4 | %16 |
| 2 | 1.6917 | 1.7069 | 548 s | 5.6e-4 | %3 |
| 3 | 1.6198 | 1.6632 | 552 s | 3.3e-4 | %5 |
| 4 | 1.5640 | 1.6388 | 556 s | 1.6e-4 | %21 |
| 5 | 1.5264 | **1.6337** | 550 s | 1.0e-4 | %63 |

Rastgele tahmin `ln(136) = 4.91`.

**Okunması gerekenler:**

* İlk epoch'ta adımların %80'i kırpılıyordu (warmup sırasında gradyan normu
  ~3.2). Kırpma oranı epoch 2'de %3'e indi — yani eşik doğru yerde: normal
  adımlara dokunmuyor, yalnızca sıçramaları kesiyor. Son epoch'ta oranın
  %63'e çıkması LR'in küçülmesinden, gradyanların değil.
* `train` ile `val` arasındaki fark epoch 3'ten sonra açılıyor
  (epoch 5'te 1.5264 vs 1.6337). Bu, **doyum bölgesine girildiğinin** işareti:
  7. epoch'tan itibaren val CE düşmeyebilir. Bir sonraki deney dropout veya
  daha fazla veri olmalı, daha fazla epoch değil.
* Üretim çıktısı Türkçe morfolojisine benziyor ama anlamsız — 842k parametre ve
  17M token bu ölçek için az; beklenen davranış.

```console
$ stonefish_ai gpt-gen "Akşam erken iner" 300 0.8
Akşam erken iner
Böcekler doğurmuş gibi yalan mi
Benden de bir kız bu gece bir anı ezilmiş ve ola
Sevgici güneş insan bir kaya sarı sakınmış yüreği gece
...
```

300 token üretim ≈ 2.8 s (KV-cache yok, her token tüm bağlamı yeniden hesaplar).

---

## Mimari

```
src/
  lib.rs          kütüphane kökü (modüller + VERSION)
  tensor.rs       Matrix (row-major 2D) — matmul, transpose, broadcast add/sub
  autograd.rs     Op enum, Tensor, topolojik backward, tüm op'lar
  nn.rs           Linear + MNIST model (de)serializasyonu
  transformer.rs  Embedding, MultiHeadSelfAttention, TransformerBlock, GPT, GptConfig
  checkpoint.rs   GPT checkpoint v2 (config + vocab + tensörler)
  optim.rs        SGD, Adam (bias correction'lı), gradyan kırpma
  loss.rs         softmax + cross-entropy
  data.rs         MNIST idx okuyucu, metin → sequence
  tokenizer.rs    karakter seviyesi tokenizer (deterministik ID)
  gpu.rs          cuBLAS matmul (+ ᵀ varyantları) + CPU yedeği (feature flag)
  profile.rs      hafif profil sayaçları (test-gpt-profile)
tests/
  gradcheck.rs    27 sayısal gradyan testi + nedensellik/batch tutarlılığı
examples/
  grad_probe.rs   parametre bazında analitik ↔ numerik gradyan teşhis aracı
```

### CUDA'sız derleme

```bash
cargo test --no-default-features      # her şey CPU'da
```

`gpu::matmul_gpu` bu durumda `Matrix::matmul`'a düşer; kod yolu aynıdır.

---

## Bilinen sınırlar

* **Üretimde KV-cache yok** — her token için tüm bağlam yeniden hesaplanır (O(n²)).
* **Konum indeksleri üretimde kayıyor** — pencere kaydıkça `pos_emb` yeniden 0'dan
  uygulanır; eğitimdeki mutlak konumlarla tam örtüşmez. Doğru çözüm RoPE veya
  eğitimde kayan pencere kullanmaktır.
* **Tek thread'li CPU** — `rayon` yok; CPU yolu referans/yedek yoldur.
* Ağırlık paylaşımı (weight tying), dropout, LR warmup/cosine, gradient clipping yok.
* CPU↔GPU transferi matmul başına yapılır; tam device-resident bir motor (tensörler
  GPU'da yaşar) bir sonraki büyük adım.
