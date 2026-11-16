// src/transformer.rs
// Transformer bileşenleri: Embedding, MultiHeadAttention, TransformerBlock, GPT.

use crate::autograd::{Tensor, Op};
use crate::tensor::Matrix;
use crate::nn::Linear;

// ============================================================
// Embedding
// ============================================================

pub struct Embedding {
    pub token_emb: Tensor,
    pub pos_emb: Tensor,
}

impl Embedding {
    pub fn new(vocab_size: usize, max_seq_len: usize, d_model: usize, seed: &mut u64) -> Self {
        let scale = 0.02;
        let token_emb = Tensor::leaf(Matrix::random(vocab_size, d_model, scale, seed), true);
        let pos_emb = Tensor::leaf(Matrix::random(max_seq_len, d_model, scale, seed), true);
        Self { token_emb, pos_emb }
    }

    pub fn forward(&self, token_ids: &[usize]) -> Tensor {
        let seq_len = token_ids.len();
        let d_model = self.token_emb.data.borrow().cols;
        let te = self.token_emb.data.borrow();
        let pe = self.pos_emb.data.borrow();
        let mut out = Matrix::zeros(seq_len, d_model);
        for s in 0..seq_len {
            let id = token_ids[s];
            for d in 0..d_model {
                out.set(s, d, te.get(id, d) + pe.get(s, d));
            }
        }
        drop(te);
        drop(pe);

        let ids_mat = Matrix::from_vec(
            seq_len, 1,
            token_ids.iter().map(|&x| x as f32).collect(),
        );
        let ids_t = Tensor::leaf(ids_mat, false);

        Tensor::from_op(
            out,
            Op::Embedding,
            vec![self.token_emb.clone(), self.pos_emb.clone(), ids_t],
        )
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        vec![self.token_emb.clone(), self.pos_emb.clone()]
    }
}

// ============================================================
// Attention (tek head)
// ============================================================

pub struct AttentionHead {
    pub w_q: Tensor,
    pub w_k: Tensor,
    pub w_v: Tensor,
    pub w_o: Tensor,
}

impl AttentionHead {
    pub fn new(d_model: usize, seed: &mut u64) -> Self {
        let scale = (1.0 / d_model as f32).sqrt();
        let w_q = Tensor::leaf(Matrix::random(d_model, d_model, scale, seed), true);
        let w_k = Tensor::leaf(Matrix::random(d_model, d_model, scale, seed), true);
        let w_v = Tensor::leaf(Matrix::random(d_model, d_model, scale, seed), true);
        let w_o = Tensor::leaf(Matrix::random(d_model, d_model, scale, seed), true);
        Self { w_q, w_k, w_v, w_o }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let d_model = x.data.borrow().cols;
        let dk_sqrt = (d_model as f32).sqrt();
        let q = x.matmul(&self.w_q);
        let k = x.matmul(&self.w_k);
        let v = x.matmul(&self.w_v);
        let scores = q.matmul(&k.transpose()).scale(1.0 / dk_sqrt);
        let masked = scores.mask_causal();
        let weights = masked.softmax();
        let attn = weights.matmul(&v);
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
// Multi-Head Attention (şimdilik tek head kullanıyor)
// ============================================================

pub struct MultiHeadAttention {
    pub heads: Vec<AttentionHead>,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, n_heads: usize, seed: &mut u64) -> Self {
        let mut heads = Vec::with_capacity(n_heads);
        for _ in 0..n_heads {
            heads.push(AttentionHead::new(d_model, seed));
        }
        Self { heads }
    }

    /// Geçici: sadece ilk head kullanılıyor.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.heads[0].forward(x)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        for h in &self.heads {
            p.extend(h.parameters());
        }
        p
    }
}

// ============================================================
// Transformer Block
// ============================================================

pub struct TransformerBlock {
    pub ln1_gamma: Tensor,
    pub ln1_beta: Tensor,
    pub attn: MultiHeadAttention,
    pub ln2_gamma: Tensor,
    pub ln2_beta: Tensor,
    pub ff1: Linear,
    pub ff2: Linear,
}

impl TransformerBlock {
    pub fn new(d_model: usize, n_heads: usize, seed: &mut u64) -> Self {
        let ln1_gamma = Tensor::leaf(Matrix::filled(1, d_model, 1.0), true);
        let ln1_beta = Tensor::leaf(Matrix::zeros(1, d_model), true);
        let ln2_gamma = Tensor::leaf(Matrix::filled(1, d_model, 1.0), true);
        let ln2_beta = Tensor::leaf(Matrix::zeros(1, d_model), true);
        let attn = MultiHeadAttention::new(d_model, n_heads, seed);
        let ff1 = Linear::new(d_model, 4 * d_model, seed);
        let ff2 = Linear::new(4 * d_model, d_model, seed);
        Self { ln1_gamma, ln1_beta, attn, ln2_gamma, ln2_beta, ff1, ff2 }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x_norm = x.layer_norm(&self.ln1_gamma, &self.ln1_beta, 1e-5);
        let attn_out = self.attn.forward(&x_norm);
        let h = x.add(&attn_out);
        let h_norm = h.layer_norm(&self.ln2_gamma, &self.ln2_beta, 1e-5);
        let ff = self.ff1.forward(&h_norm).gelu();
        let ff = self.ff2.forward(&ff);
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
// GPT Model
// ============================================================

pub struct GPT {
    pub embedding: Embedding,
    pub blocks: Vec<TransformerBlock>,
    pub ln_f_gamma: Tensor,
    pub ln_f_beta: Tensor,
    pub head: Linear,
}

impl GPT {
    pub fn new(
        vocab_size: usize,
        max_seq_len: usize,
        d_model: usize,
        n_layers: usize,
        n_heads: usize,
        seed: &mut u64,
    ) -> Self {
        let embedding = Embedding::new(vocab_size, max_seq_len, d_model, seed);
        let mut blocks = Vec::new();
        for _ in 0..n_layers {
            blocks.push(TransformerBlock::new(d_model, n_heads, seed));
        }
        let ln_f_gamma = Tensor::leaf(Matrix::filled(1, d_model, 1.0), true);
        let ln_f_beta = Tensor::leaf(Matrix::zeros(1, d_model), true);
        let head = Linear::new(d_model, vocab_size, seed);
        Self { embedding, blocks, ln_f_gamma, ln_f_beta, head }
    }

    pub fn forward(&self, token_ids: &[usize]) -> Tensor {
        let mut x = self.embedding.forward(token_ids);
        for block in &self.blocks {
            x = block.forward(&x);
        }
        let x = x.layer_norm(&self.ln_f_gamma, &self.ln_f_beta, 1e-5);
        self.head.forward(&x)
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
}