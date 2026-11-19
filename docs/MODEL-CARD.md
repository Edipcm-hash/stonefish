# Model card — StonefishAI v2 (222M)

This card describes the model that ships with v1.0. It is written to be read
alongside the README's measurements, and it says as much about what the model
cannot do as about what it can.

## Model details

| | |
|---|---|
| Architecture | decoder-only transformer, pre-norm |
| Parameters | **222,331,904** |
| Layers | 16 |
| Width | d_model 1024 |
| Heads | 8 (d_head 128) |
| Context | 512 tokens |
| Vocabulary | 16,384 (byte-level BPE) |
| Normalisation | RMSNorm, no bias, eps 1e-6 |
| Activation | SwiGLU, hidden 2816 |
| Position | RoPE, theta 10,000, cached cos/sin table |
| Tying | output head reuses the token embedding |
| Dropout | 0.1 during training |
| Precision | fp32 |
| Checkpoint | `stonefish_en.final.state` (`SFEN001`) |

## Training

| | |
|---|---|
| Data | TinyStories v2 + Project Gutenberg (English) |
| Tokens | **2,047,457,313** train · 4,041,031 held out |
| Steps | 62,483 |
| Batch | 4 sequences × gradient accumulation 16 = 32,768 tokens/step |
| Optimiser | Adam (β 0.9 / 0.999, eps 1e-6) |
| LR | 4e-4, 1,000-step warmup, cosine decay, gradient clip 1.0 |
| Hardware | one RTX 5070 (12 GB), 250 W, ~5.5 days |
| Throughput | ~10,700 tokens/s |
| Data reuse | ~9.2 tokens per parameter |

## Intended use

- **Text continuation** in English, especially narrative prose.
- **A reference implementation**: reading, running and testing a language model
  that has no ML framework underneath.
- **Teaching**: every op's gradient is checked numerically, so the code can be
  used to show why a backward pass is correct.

## Out of scope

- **Instruction following, chat, Q&A.** The model was never fine-tuned for it;
  it will produce plausible-sounding prose that does not answer the question.
- **Factual reliability.** It will state invented facts fluently. Do not use it
  as a knowledge source.
- **Code generation, translation, summarisation** as products.
- **Anything with consequences**: medical, legal, financial, safety.
- **Languages other than English.**

## Evaluation

Measured on held-out data with `stonefish_ai en-eval`, reported as bits per
character (BPC) because it is comparable across tokenisers:

| Corpus | Loss (nat/token) | Perplexity | BPC |
|---|---|---|---|
| TinyStories (held out) | see README §7 | — | — |
| Gutenberg (held out) | see README §7 | — | — |
| Random baseline (BPE-16k) | 9.70 | 16,384 | 2.28 |

The README's §7 tables are the source of truth; this card intentionally does not
duplicate numbers that can drift out of sync. Run `en-eval` yourself — the exact
command is in §2.1.

## Bias, risks and limitations

- **Training data bias.** TinyStories is synthetic children's prose; Gutenberg is
  public-domain books from a mostly 19th-century Western canon. Both carry the
  demographics, values and blind spots of their sources.
- **Reproduction of harmful text.** As with any raw language model, a hostile
  prompt can elicit biased or unpleasant continuations. There is no safety
  tuning and no output filter.
- **Memorisation.** With ~9 tokens per parameter the model is not large enough to
  memorise the corpus the way much bigger models can, but rare passages could
  still be reproduced. Do not train on data you cannot redistribute.
- **Silent failure.** The model does not know when it is wrong, and its errors
  read exactly like its correct output.

## How to reproduce

```bash
cargo run --release -- demo 90        # a small model from scratch, in 90 s
cargo test --release                  # 75 tests
cargo run --release -- en-eval        # evaluate the shipped checkpoint
cargo run --release -- en-gen "Once upon a time" 150 0.8
# without CUDA: add --no-default-features; the CPU path is ~3 tokens/s
cargo run --release --no-default-features -- en-gen "Once upon a time" 10 0
```

## Licence

MIT for the code. The trained weights are released under the same terms; the
training corpora keep their own licences (TinyStories: CDLA-Sharing-1.0;
Gutenberg: public domain in the US), which is why the repository ships no data.
