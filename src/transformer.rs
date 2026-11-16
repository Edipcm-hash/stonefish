// src/transformer.rs
// Transformer bileşenleri: Embedding, MultiHeadSelfAttention, TransformerBlock, GPT.
//
// Tüm bileşenler **satır paketli** çalışır: (batch, seq, d_model) tensörü
// (batch*seq, d_model) matrisi olarak temsil edilir. LayerNorm/Linear/GELU satır
// bazlı olduğu için bu bedava; attention ise `Tensor::causal_attention` içinde
// batch/seq/head bilgisiyle çözülür.

use crate::autograd::{Op, Tensor};
use crate::nn::Linear;
use crate::tensor::Matrix;

// ============================================================
// Embedding (token + pozisyon)
// ============================================================

pub struct Embedding {
    pub token_emb: Tensor, // (vocab, d_model)
    pub pos_emb: Tensor,   // (max_seq_len, d_model)
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub d_model: usize,
}

impl Embedding {
    pub fn new(vocab_size: usize, max_seq_len: usize, d_model: usize, seed: &mut u64) -> Self {
        let scale = 0.02;
        Self {
            token_emb: Tensor::leaf(Matrix::random(vocab_size, d_model, scale, seed), true),
            pos_emb: Tensor::leaf(Matrix::random(max_seq_len, d_model, scale, seed), true),
            vocab_size,
            max_seq_len,
            d_model,
        }
    }

    /// `ids` satır paketli: uzunluk = batch*seq, pozisyon = satır % seq_len.
    pub fn forward(&self, ids: &[usize], seq_len: usize) -> Tensor {
        assert!(
            seq_len <= self.max_seq_len,
            "seq_len ({seq_len}) > max_seq_len ({})",
            self.max_seq_len
        );
        let rows = ids.len();
        let d_model = self.d_model;

        let out = {
            let te = self.token_emb.data.borrow();
            let pe = self.pos_emb.data.borrow();
            let mut out = Matrix::zeros(rows, d_model);
            for (r, &id) in ids.iter().enumerate() {
                assert!(id < self.vocab_size, "token id {id} >= vocab {}", self.vocab_size);
                let s = r % seq_len;
                let (dst, tok, pos) = (
                    &mut out.data[r * d_model..(r + 1) * d_model],
                    &te.data[id * d_model..(id + 1) * d_model],
                    &pe.data[s * d_model..(s + 1) * d_model],
                );
                for d in 0..d_model {
                    dst[d] = tok[d] + pos[d];
                }
            }
            out
        };

        let ids_mat = Matrix::from_vec(rows, 1, ids.iter().map(|&x| x as f32).collect());
        let ids_t = Tensor::leaf(ids_mat, false);

        Tensor::from_op(
            out,
            Op::Embedding { seq_len },
            vec![self.token_emb.clone(), self.pos_emb.clone(), ids_t],
        )
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        vec![self.token_emb.clone(), self.pos_emb.clone()]
    }
}

// ============================================================
// Multi-Head Self-Attention
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
        let proj = |seed: &mut u64| Tensor::leaf(Matrix::random(d_model, d_model, scale, seed), true);
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
// Transformer Block (pre-LN)
// ============================================================

pub struct TransformerBlock {
    pub ln1_gamma: Tensor,
    pub ln1_beta: Tensor,
    pub attn: MultiHeadSelfAttention,
    pub ln2_gamma: Tensor,
    pub ln2_beta: Tensor,
    pub ff1: Linear,
    pub ff2: Linear,
}

impl TransformerBlock {
    pub fn new(d_model: usize, n_heads: usize, seed: &mut u64) -> Self {
        Self {
            ln1_gamma: Tensor::leaf(Matrix::filled(1, d_model, 1.0), true),
            ln1_beta: Tensor::leaf(Matrix::zeros(1, d_model), true),
            ln2_gamma: Tensor::leaf(Matrix::filled(1, d_model, 1.0), true),
            ln2_beta: Tensor::leaf(Matrix::zeros(1, d_model), true),
            attn: MultiHeadSelfAttention::new(d_model, n_heads, seed),
            ff1: Linear::new(d_model, 4 * d_model, seed),
            ff2: Linear::new(4 * d_model, d_model, seed),
        }
    }

    pub fn forward(&self, x: &Tensor, batch: usize, seq: usize) -> Tensor {
        // pre-LN residual: x + attn(LN(x));  x + ff(LN(x))
        let x_norm = x.layer_norm(&self.ln1_gamma, &self.ln1_beta, 1e-5);
        let h = x.add(&self.attn.forward(&x_norm, batch, seq));
        let h_norm = h.layer_norm(&self.ln2_gamma, &self.ln2_beta, 1e-5);
        let ff = self.ff2.forward(&self.ff1.forward(&h_norm).gelu());
        h.add(&ff)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = vec![
            self.ln1_gamma.clone(),
            self.ln1_beta.clone(),
            self.ln2_gamma.clone(),
            self.ln2_beta.clone(),
        ];
        p.extend(self.attn.parameters());
        p.extend(self.ff1.parameters());
        p.extend(self.ff2.parameters());
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
    /// Gerçek parametre sayısı (her tensörün eleman sayısı toplamı).
    pub fn n_params(&self) -> usize {
        let v = self.vocab_size;
        let d = self.d_model;
        let s = self.max_seq_len;
        let ln = 4 * d; // ln1/ln2 gamma+beta
        let attn = 4 * d * d;
        let ff = (d * 4 * d + 4 * d) + (4 * d * d + d);
        v * d + s * d + self.n_layers * (ln + attn + ff) + 2 * d + (d * v + v)
    }
}

pub struct GPT {
    pub config: GptConfig,
    pub embedding: Embedding,
    pub blocks: Vec<TransformerBlock>,
    pub ln_f_gamma: Tensor,
    pub ln_f_beta: Tensor,
    pub head: Linear,
}

impl GPT {
    pub fn new(config: GptConfig, seed: &mut u64) -> Self {
        let d_model = config.d_model;
        let embedding = Embedding::new(
            config.vocab_size,
            config.max_seq_len,
            d_model,
            seed,
        );
        let mut blocks = Vec::with_capacity(config.n_layers);
        for _ in 0..config.n_layers {
            blocks.push(TransformerBlock::new(d_model, config.n_heads, seed));
        }
        Self {
            ln_f_gamma: Tensor::leaf(Matrix::filled(1, d_model, 1.0), true),
            ln_f_beta: Tensor::leaf(Matrix::zeros(1, d_model), true),
            head: Linear::new(d_model, config.vocab_size, seed),
            config,
            embedding,
            blocks,
        }
    }

    /// `ids`: uzunluk batch*seq. Döner: logits (batch*seq, vocab_size).
    pub fn forward_batch(&self, ids: &[usize], batch: usize, seq: usize) -> Tensor {
        assert_eq!(ids.len(), batch * seq, "ids.len() = batch*seq olmalı");
        assert!(seq <= self.config.max_seq_len, "seq, max_seq_len'i aşıyor");
        let mut x = self.embedding.forward(ids, seq);
        for block in &self.blocks {
            x = block.forward(&x, batch, seq);
        }
        let x = x.layer_norm(&self.ln_f_gamma, &self.ln_f_beta, 1e-5);
        self.head.forward(&x)
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
        p.push(self.ln_f_beta.clone());
        p.extend(self.head.parameters());
        p
    }

    pub fn n_params(&self) -> usize {
        self.parameters()
            .iter()
            .map(|p| p.data.borrow().data.len())
            .sum()
    }
}
