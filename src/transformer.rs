// src/transformer.rs
// Transformer bileşenleri (v2 mimarisi): Embedding, MultiHeadSelfAttention,
// TransformerBlock, GPT.
//
// v2'de değişenler (v1 → v2):
//   LayerNorm  → RMSNorm  (ortalama ve beta yok, daha az iş, eşit kalite)
//   GELU FFN   → SwiGLU   (kapılı: silu(x·W1)·(x·W3), aynı parametre bütçesi)
//   mutlak pos → RoPE     (Q/K döndürülür; göreli konum, uzun bağlama genelleme)
//   ayrı head  → weight tying (giriş gömmesi = çıkış matrisi, -V·d parametre)
//
// Tüm bileşenler **satır paketli** çalışır: (batch, seq, d_model) tensörü
// (batch*seq, d_model) matrisi olarak temsil edilir. RMSNorm/SwiGLU/Linear
// satır bazlı olduğu için bu bedava; attention `Tensor::causal_attention`
// içinde batch/seq/head bilgisiyle çözülür.

use crate::autograd::Tensor;
use crate::tensor::Matrix;

/// RMSNorm epsilon'u.
pub const RMS_EPS: f32 = 1e-6;

/// SwiGLU gizli boyutu: LLaMA tarzı ~8/3·d.
///
/// Çıktı, GPU verimliliği için bir basamağa yuvarlanır. Basamak model
/// boyutuyla ölçeklenir — yoksa d_model=4 olan gradyan testi modeli 256
/// gizli nörona sahip olurdu (test için 64× fazla parametre).
/// Böylece toplam FFN parametresi v1'in (d→4d→d) ile aynı kalır.
pub fn swiglu_hidden(d_model: usize) -> usize {
    let raw = (8 * d_model).div_ceil(3);
    let step = if raw >= 1024 {
        256
    } else if raw >= 256 {
        64
    } else {
        8
    };
    raw.div_ceil(step) * step
}

// ============================================================
// Embedding (yalnız token — konum bilgisi RoPE'den gelir)
// ============================================================

pub struct Embedding {
    pub token_emb: Tensor, // (vocab, d_model)
    pub vocab_size: usize,
    pub d_model: usize,
}

impl Embedding {
    pub fn new(vocab_size: usize, d_model: usize, seed: &mut u64) -> Self {
        let scale = 0.02;
        Self {
            token_emb: Tensor::leaf(Matrix::random(vocab_size, d_model, scale, seed), true),
            vocab_size,
            d_model,
        }
    }

    /// `ids` satır paketli: uzunluk = batch*seq, pozisyon = satır % seq_len.
    pub fn forward(&self, ids: &[usize]) -> Tensor {
        let rows = ids.len();
        let d_model = self.d_model;
        let out = {
            let te = self.token_emb.data.borrow();
            let mut out = Matrix::zeros(rows, d_model);
            for (r, &id) in ids.iter().enumerate() {
                assert!(
                    id < self.vocab_size,
                    "token id {id} >= vocab {}",
                    self.vocab_size
                );
                let (dst, tok) = (
                    &mut out.data[r * d_model..(r + 1) * d_model],
                    &te.data[id * d_model..(id + 1) * d_model],
                );
                dst.copy_from_slice(tok);
            }
            out
        };
        let ids_mat = Matrix::from_vec(rows, 1, ids.iter().map(|&x| x as f32).collect());
        let ids_t = Tensor::leaf(ids_mat, false);
        Tensor::from_op(
            out,
            crate::autograd::Op::Embedding { seq_len: 1 },
            vec![self.token_emb.clone(), ids_t],
        )
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        vec![self.token_emb.clone()]
    }
}

// ============================================================
// Multi-Head Self-Attention (RoPE'lu)
// ============================================================

pub struct MultiHeadSelfAttention {
    pub w_q: Tensor, // (d_model, d_model)
    pub w_k: Tensor,
    pub w_v: Tensor,
    pub w_o: Tensor,
    pub n_heads: usize,
}

impl MultiHeadSelfAttention {
    pub fn new(d_model: usize, n_heads: usize, seed: &mut u64) -> Self {
        assert!(n_heads > 0, "n_heads > 0 olmalı");
        assert_eq!(d_model % n_heads, 0, "d_model, n_heads'e bölünmeli");
        let scale = (1.0 / d_model as f32).sqrt();
        let proj =
            |seed: &mut u64| Tensor::leaf(Matrix::random(d_model, d_model, scale, seed), true);
        Self {
            w_q: proj(seed),
            w_k: proj(seed),
            w_v: proj(seed),
            w_o: proj(seed),
            n_heads,
        }
    }

    /// x: (batch*seq, d_model) → (batch*seq, d_model)
    pub fn forward(&self, x: &Tensor, batch: usize, seq: usize) -> Tensor {
        let q = x.matmul(&self.w_q);
        let k = x.matmul(&self.w_k);
        let v = x.matmul(&self.w_v);
        // RoPE yalnız Q ve K'ye uygulanır (V'ye uygulanmaz).
        let q = q.rope(batch, seq, self.n_heads);
        let k = k.rope(batch, seq, self.n_heads);
        let attn = q.causal_attention(&k, &v, batch, seq, self.n_heads);
        attn.matmul(&self.w_o)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        vec![
            self.w_q.clone(),
            self.w_k.clone(),
            self.w_v.clone(),
            self.w_o.clone(),
        ]
    }
}

// ============================================================
// Transformer Block (pre-norm, RMSNorm + SwiGLU)
// ============================================================

pub struct TransformerBlock {
    pub norm1_gamma: Tensor, // (1, d_model)
    pub attn: MultiHeadSelfAttention,
    pub norm2_gamma: Tensor,
    pub ff_gate: Tensor, // (d_model, hidden)   W1
    pub ff_up: Tensor,   // (d_model, hidden)   W3
    pub ff_down: Tensor, // (hidden, d_model)   W2
    pub hidden: usize,
}

impl TransformerBlock {
    pub fn new(d_model: usize, n_heads: usize, seed: &mut u64) -> Self {
        let hidden = swiglu_hidden(d_model);
        let scale = (1.0 / d_model as f32).sqrt();
        let ff_scale = (2.0 / d_model as f32).sqrt();
        Self {
            norm1_gamma: Tensor::leaf(Matrix::filled(1, d_model, 1.0), true),
            attn: MultiHeadSelfAttention::new(d_model, n_heads, seed),
            norm2_gamma: Tensor::leaf(Matrix::filled(1, d_model, 1.0), true),
            ff_gate: Tensor::leaf(Matrix::random(d_model, hidden, ff_scale, seed), true),
            ff_up: Tensor::leaf(Matrix::random(d_model, hidden, ff_scale, seed), true),
            ff_down: Tensor::leaf(Matrix::random(hidden, d_model, scale, seed), true),
            hidden,
        }
    }

    pub fn forward(&self, x: &Tensor, batch: usize, seq: usize) -> Tensor {
        // pre-norm residual: x + attn(RMSNorm(x));  h + ffn(RMSNorm(h))
        let x_norm = x.rms_norm(&self.norm1_gamma, RMS_EPS);
        let h = x.add(&self.attn.forward(&x_norm, batch, seq));
        let h_norm = h.rms_norm(&self.norm2_gamma, RMS_EPS);
        // SwiGLU: silu(x·W1) ⊙ (x·W3) ·W2
        let gate = h_norm.matmul(&self.ff_gate);
        let up = h_norm.matmul(&self.ff_up);
        let silu = gate.mul_elem(&gate.sigmoid()); // silu(g) = g·σ(g)
        let act = silu.mul_elem(&up);
        h.add(&act.matmul(&self.ff_down))
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = vec![self.norm1_gamma.clone(), self.norm2_gamma.clone()];
        p.extend(self.attn.parameters());
        p.push(self.ff_gate.clone());
        p.push(self.ff_up.clone());
        p.push(self.ff_down.clone());
        p
    }
}

// ============================================================
// GPT
// ============================================================

/// Model hiperparametreleri — checkpoint'e gömülür, yüklemede doğrulanır.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GptConfig {
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
}

impl GptConfig {
    pub fn d_head(&self) -> usize {
        self.d_model / self.n_heads
    }

    /// Gerçek parametre sayısı (v2 mimarisi, weight tying dahil).
    pub fn n_params(&self) -> usize {
        let v = self.vocab_size;
        let d = self.d_model;
        let h = swiglu_hidden(d);
        let norm = 3 * d; // block içi 2 + final 1 (beta yok)
        let attn = 4 * d * d;
        let ff = 3 * d * h; // W1 + W3 + W2, bias yok
        // gömme yalnız bir kez sayılır (çıkış matrisi onunla paylaşılır)
        v * d + self.n_layers * (norm + attn + ff)
    }
}

pub struct GPT {
    pub config: GptConfig,
    pub embedding: Embedding,
    pub blocks: Vec<TransformerBlock>,
    pub ln_f_gamma: Tensor,
}

impl GPT {
    pub fn new(config: GptConfig, seed: &mut u64) -> Self {
        let d_model = config.d_model;
        let embedding = Embedding::new(config.vocab_size, d_model, seed);
        let mut blocks = Vec::with_capacity(config.n_layers);
        for _ in 0..config.n_layers {
            blocks.push(TransformerBlock::new(d_model, config.n_heads, seed));
        }
        Self {
            ln_f_gamma: Tensor::leaf(Matrix::filled(1, d_model, 1.0), true),
            config,
            embedding,
            blocks,
        }
    }

    /// `ids`: uzunluk batch*seq. Döner: logits (batch*seq, vocab_size).
    pub fn forward_batch(&self, ids: &[usize], batch: usize, seq: usize) -> Tensor {
        assert_eq!(ids.len(), batch * seq, "ids.len() = batch*seq olmalı");
        assert!(seq <= self.config.max_seq_len, "seq, max_seq_len'i aşıyor");
        let mut x = self.embedding.forward(ids);
        for block in &self.blocks {
            x = block.forward(&x, batch, seq);
        }
        let x = x.rms_norm(&self.ln_f_gamma, RMS_EPS);
        // weight tying: logits = x · tok_embᵀ  (ayrı çıkış matrisi yok)
        x.matmul(&self.embedding.token_emb.transpose())
    }

    /// Tek sequence (üretim yolu). Döner: (seq, vocab_size).
    pub fn forward(&self, ids: &[usize]) -> Tensor {
        self.forward_batch(ids, 1, ids.len())
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        p.extend(self.embedding.parameters());
        for b in &self.blocks {
            p.extend(b.parameters());
        }
        p.push(self.ln_f_gamma.clone());
        p
    }

    pub fn n_params(&self) -> usize {
        self.parameters()
            .iter()
            .map(|p| {
                let d = p.data.borrow();
                d.rows * d.cols
            })
            .sum()
    }
}
