// src/gpu_kernels.rs
// CUDA çekirdeklerinin Rust tarafı — HARİCİ CRATE YOK.
//
// NVRTC ile çalışma zamanında derleme + PTX disk önbelleği `gpu` modülünde
// (`kernel`/`launch`). Burada yalnızca tipli sarmalayıcılar var.

#![cfg(feature = "cuda")]

use crate::gpu::{self, cfg_1d, cfg_rows};
use crate::sys::dev::{DevPtr, Kargs};

/// Çekirdek kaynağı — derleme zamanında gömülür.
pub const SRC: &str = include_str!("../kernels/stonefish.cu");

// ============================================================
// Elementwise
// ============================================================

/// out = a + bias (satır bazlı broadcast)
pub fn add_bias(out: &impl DevPtr, a: &impl DevPtr, bias: &impl DevPtr, rows: usize, cols: usize) {
    let mut k = Kargs::new();
    k.buf(out)
        .buf(a)
        .buf(bias)
        .i32(rows as i32)
        .i32(cols as i32);
    gpu::launch("add_scalar_broadcast", cfg_1d(rows * cols), 0, &mut k);
}

/// x += bias (yerinde)
pub fn add_bias_inplace(x: &impl DevPtr, bias: &impl DevPtr, rows: usize, cols: usize) {
    let mut k = Kargs::new();
    k.buf(x).buf(bias).i32(rows as i32).i32(cols as i32);
    gpu::launch("add_bias_inplace", cfg_1d(rows * cols), 0, &mut k);
}

pub fn add_inplace(dst: &impl DevPtr, src: &impl DevPtr, n: usize) {
    let mut k = Kargs::new();
    k.buf(dst).buf(src).i32(n as i32);
    gpu::launch("add_inplace", cfg_1d(n), 0, &mut k);
}

pub fn zero_fill(dst: &impl DevPtr, n: usize) {
    let mut k = Kargs::new();
    k.buf(dst).i32(n as i32);
    gpu::launch("zero_fill", cfg_1d(n), 0, &mut k);
}

pub fn gelu_forward(out: &impl DevPtr, x: &impl DevPtr, n: usize) {
    let mut k = Kargs::new();
    k.buf(out).buf(x).i32(n as i32);
    gpu::launch("gelu_forward", cfg_1d(n), 0, &mut k);
}

pub fn gelu_backward(dx: &impl DevPtr, dy: &impl DevPtr, x: &impl DevPtr, n: usize) {
    let mut k = Kargs::new();
    k.buf(dx).buf(dy).buf(x).i32(n as i32);
    gpu::launch("gelu_backward", cfg_1d(n), 0, &mut k);
}

// ============================================================
// LayerNorm
// ============================================================

#[allow(clippy::too_many_arguments)]
pub fn layernorm_forward(
    out: &impl DevPtr,
    xhat: &impl DevPtr,
    inv: &impl DevPtr,
    x: &impl DevPtr,
    gamma: &impl DevPtr,
    beta: &impl DevPtr,
    rows: usize,
    cols: usize,
    eps: f32,
) {
    let threads = 128u32;
    let mut k = Kargs::new();
    k.buf(out)
        .buf(xhat)
        .buf(inv)
        .buf(x)
        .buf(gamma)
        .buf(beta)
        .i32(rows as i32)
        .i32(cols as i32)
        .f32(eps);
    gpu::launch(
        "layernorm_forward",
        cfg_rows(rows, threads),
        2 * threads * 4,
        &mut k,
    );
}

pub fn layernorm_backward(
    dx: &impl DevPtr,
    dy: &impl DevPtr,
    xhat: &impl DevPtr,
    inv: &impl DevPtr,
    gamma: &impl DevPtr,
    rows: usize,
    cols: usize,
) {
    let threads = 128u32;
    let mut k = Kargs::new();
    k.buf(dx)
        .buf(dy)
        .buf(xhat)
        .buf(inv)
        .buf(gamma)
        .i32(rows as i32)
        .i32(cols as i32);
    gpu::launch(
        "layernorm_backward",
        cfg_rows(rows, threads),
        2 * threads * 4,
        &mut k,
    );
}

pub fn layernorm_param_grad(
    dgamma: &impl DevPtr,
    dbeta: &impl DevPtr,
    dy: &impl DevPtr,
    xhat: &impl DevPtr,
    rows: usize,
    cols: usize,
) {
    let mut k = Kargs::new();
    k.buf(dgamma)
        .buf(dbeta)
        .buf(dy)
        .buf(xhat)
        .i32(rows as i32)
        .i32(cols as i32);
    gpu::launch(
        "layernorm_param_grad",
        ((cols as u32, 1, 1), (1, 1, 1)),
        0,
        &mut k,
    );
}

// ============================================================
// Attention softmax'ı
// ============================================================

pub fn masked_softmax_forward(
    w: &impl DevPtr,
    scores: &impl DevPtr,
    batch: usize,
    n_heads: usize,
    seq: usize,
    scale: f32,
) {
    let rows = batch * n_heads * seq;
    let mut k = Kargs::new();
    k.buf(w)
        .buf(scores)
        .i32(seq as i32)
        .i32(n_heads as i32)
        .i32(batch as i32)
        .f32(scale);
    gpu::launch("masked_softmax_forward", cfg_rows(rows, 128), 0, &mut k);
}

pub fn masked_softmax_backward(
    ds: &impl DevPtr,
    w: &impl DevPtr,
    dw: &impl DevPtr,
    batch: usize,
    n_heads: usize,
    seq: usize,
    scale: f32,
) {
    let rows = batch * n_heads * seq;
    let mut k = Kargs::new();
    k.buf(ds)
        .buf(w)
        .buf(dw)
        .i32(seq as i32)
        .i32(n_heads as i32)
        .i32(batch as i32)
        .f32(scale);
    gpu::launch("masked_softmax_backward", cfg_rows(rows, 128), 0, &mut k);
}

// ============================================================
// Embedding
// ============================================================

pub fn embedding_forward(
    out: &impl DevPtr,
    ids: &impl DevPtr,
    tok_emb: &impl DevPtr,
    pos_emb: &impl DevPtr,
    rows: usize,
    cols: usize,
    seq_len: usize,
) {
    let mut k = Kargs::new();
    k.buf(out)
        .buf(ids)
        .buf(tok_emb)
        .buf(pos_emb)
        .i32(rows as i32)
        .i32(cols as i32)
        .i32(seq_len as i32);
    gpu::launch("embedding_forward", cfg_1d(rows * cols), 0, &mut k);
}

pub fn embedding_backward(
    dtok: &impl DevPtr,
    dpos: &impl DevPtr,
    dout: &impl DevPtr,
    ids: &impl DevPtr,
    rows: usize,
    cols: usize,
    seq_len: usize,
) {
    let mut k = Kargs::new();
    k.buf(dtok)
        .buf(dpos)
        .buf(dout)
        .buf(ids)
        .i32(rows as i32)
        .i32(cols as i32)
        .i32(seq_len as i32);
    gpu::launch("embedding_backward", cfg_1d(rows * cols), 0, &mut k);
}

// ============================================================
// Cross-entropy
// ============================================================

pub fn ce_forward(
    loss: &impl DevPtr,
    probs: &impl DevPtr,
    logits: &impl DevPtr,
    targets: &impl DevPtr,
    rows: usize,
    classes: usize,
) {
    let mut k = Kargs::new();
    k.buf(loss)
        .buf(probs)
        .buf(logits)
        .buf(targets)
        .i32(rows as i32)
        .i32(classes as i32);
    gpu::launch("ce_forward", cfg_rows(rows, 128), 0, &mut k);
}

pub fn ce_backward(
    dlogits: &impl DevPtr,
    probs: &impl DevPtr,
    targets: &impl DevPtr,
    rows: usize,
    classes: usize,
    upstream: f32,
) {
    let mut k = Kargs::new();
    k.buf(dlogits)
        .buf(probs)
        .buf(targets)
        .i32(rows as i32)
        .i32(classes as i32)
        .f32(upstream);
    gpu::launch("ce_backward", cfg_1d(rows * classes), 0, &mut k);
}

// ============================================================
// Adam / gradyan kırpma
// ============================================================

#[allow(clippy::too_many_arguments)]
pub fn adam_step(
    param: &impl DevPtr,
    grad: &impl DevPtr,
    m: &impl DevPtr,
    v: &impl DevPtr,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    bc1: f32,
    bc2: f32,
    n: usize,
) {
    let mut k = Kargs::new();
    k.buf(param)
        .buf(grad)
        .buf(m)
        .buf(v)
        .f32(lr)
        .f32(beta1)
        .f32(beta2)
        .f32(eps)
        .f32(bc1)
        .f32(bc2)
        .i32(n as i32);
    gpu::launch("adam_step", cfg_1d(n), 0, &mut k);
}

pub fn sum_squares(acc: &impl DevPtr, x: &impl DevPtr, n: usize) {
    let mut k = Kargs::new();
    k.buf(acc).buf(x).i32(n as i32);
    let threads = 256u32;
    let blocks = ((n as u64).div_ceil(threads as u64)).max(1) as u32;
    gpu::launch(
        "sum_squares_atomic",
        ((blocks, 1, 1), (threads, 1, 1)),
        0,
        &mut k,
    );
}

pub fn scale_inplace(x: &impl DevPtr, n: usize, s: f32) {
    let mut k = Kargs::new();
    k.buf(x).i32(n as i32).f32(s);
    gpu::launch("scale_inplace", cfg_1d(n), 0, &mut k);
}

// ============================================================
// Yeni mimari için Rust sarmalayıcıları
// ============================================================

#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_forward(
    out: &impl DevPtr,
    xhat: &impl DevPtr,
    inv: &impl DevPtr,
    x: &impl DevPtr,
    gamma: &impl DevPtr,
    rows: usize,
    cols: usize,
    eps: f32,
) {
    let threads = 128u32;
    let mut k = Kargs::new();
    k.buf(out)
        .buf(xhat)
        .buf(inv)
        .buf(x)
        .buf(gamma)
        .i32(rows as i32)
        .i32(cols as i32)
        .f32(eps);
    gpu::launch(
        "rmsnorm_forward",
        cfg_rows(rows, threads),
        threads * 4,
        &mut k,
    );
}

pub fn rmsnorm_backward(
    dx: &impl DevPtr,
    dy: &impl DevPtr,
    xhat: &impl DevPtr,
    inv: &impl DevPtr,
    gamma: &impl DevPtr,
    rows: usize,
    cols: usize,
) {
    let threads = 128u32;
    let mut k = Kargs::new();
    k.buf(dx)
        .buf(dy)
        .buf(xhat)
        .buf(inv)
        .buf(gamma)
        .i32(rows as i32)
        .i32(cols as i32);
    gpu::launch(
        "rmsnorm_backward",
        cfg_rows(rows, threads),
        threads * 4,
        &mut k,
    );
}

pub fn rmsnorm_param_grad(
    dgamma: &impl DevPtr,
    dy: &impl DevPtr,
    xhat: &impl DevPtr,
    rows: usize,
    cols: usize,
) {
    let mut k = Kargs::new();
    k.buf(dgamma)
        .buf(dy)
        .buf(xhat)
        .i32(rows as i32)
        .i32(cols as i32);
    // Satır parçalama: yeterli paralellik için ~8192 blok hedefle.
    let chunks = (8192 / cols.max(1)).clamp(1, 32) as u32;
    gpu::launch(
        "rmsnorm_param_grad",
        ((cols as u32, chunks, 1), (1, 1, 1)),
        0,
        &mut k,
    );
}

pub fn silu_mul_forward(out: &impl DevPtr, gate: &impl DevPtr, up: &impl DevPtr, n: usize) {
    let mut k = Kargs::new();
    k.buf(out).buf(gate).buf(up).i32(n as i32);
    gpu::launch("silu_mul_forward", cfg_1d(n), 0, &mut k);
}

pub fn silu_mul_backward(
    dgate: &impl DevPtr,
    dup: &impl DevPtr,
    dy: &impl DevPtr,
    gate: &impl DevPtr,
    up: &impl DevPtr,
    n: usize,
) {
    let mut k = Kargs::new();
    k.buf(dgate)
        .buf(dup)
        .buf(dy)
        .buf(gate)
        .buf(up)
        .i32(n as i32);
    gpu::launch("silu_mul_backward", cfg_1d(n), 0, &mut k);
}

/// RoPE cos/sin tablosunu host'ta üret: (seq × d_head/2).
pub fn rope_tables(seq: usize, d_head: usize) -> (Vec<f32>, Vec<f32>) {
    let half = d_head / 2;
    let mut cos_t = Vec::with_capacity(seq * half);
    let mut sin_t = Vec::with_capacity(seq * half);
    for s in 0..seq {
        for j in 0..half {
            let freq = 10000f32.powf(-2.0 * j as f32 / d_head as f32);
            let ang = s as f32 * freq;
            cos_t.push(ang.cos());
            sin_t.push(ang.sin());
        }
    }
    (cos_t, sin_t)
}

#[allow(clippy::too_many_arguments)]
pub fn rope_forward(
    out: &impl DevPtr,
    x: &impl DevPtr,
    cos_t: &impl DevPtr,
    sin_t: &impl DevPtr,
    batch: usize,
    seq: usize,
    n_heads: usize,
    d_head: usize,
) {
    let half = d_head / 2;
    let total = batch * seq * n_heads * half;
    let mut k = Kargs::new();
    k.buf(out)
        .buf(x)
        .buf(cos_t)
        .buf(sin_t)
        .i32(batch as i32)
        .i32(seq as i32)
        .i32(n_heads as i32)
        .i32(d_head as i32);
    gpu::launch("rope_forward", cfg_1d(total), 0, &mut k);
}

#[allow(clippy::too_many_arguments)]
pub fn rope_backward(
    dx: &impl DevPtr,
    dy: &impl DevPtr,
    cos_t: &impl DevPtr,
    sin_t: &impl DevPtr,
    batch: usize,
    seq: usize,
    n_heads: usize,
    d_head: usize,
) {
    let half = d_head / 2;
    let total = batch * seq * n_heads * half;
    let mut k = Kargs::new();
    k.buf(dx)
        .buf(dy)
        .buf(cos_t)
        .buf(sin_t)
        .i32(batch as i32)
        .i32(seq as i32)
        .i32(n_heads as i32)
        .i32(d_head as i32);
    gpu::launch("rope_backward", cfg_1d(total), 0, &mut k);
}

pub fn dropout_inplace(x: &impl DevPtr, n: usize, p: f32, seed: u32) {
    let mut k = Kargs::new();
    k.buf(x).i32(n as i32).f32(p).i32(seed as i32);
    gpu::launch("dropout_inplace", cfg_1d(n), 0, &mut k);
}

pub fn dropout_backward(dx: &impl DevPtr, n: usize, p: f32, seed: u32) {
    let mut k = Kargs::new();
    k.buf(dx).i32(n as i32).f32(p).i32(seed as i32);
    gpu::launch("dropout_backward", cfg_1d(n), 0, &mut k);
}

/// RoPE'lu mimari: yalnız token gömmesi (pozisyon gömme yok).
pub fn embedding_forward_tok(
    out: &impl DevPtr,
    ids: &impl DevPtr,
    tok_emb: &impl DevPtr,
    rows: usize,
    cols: usize,
) {
    let mut k = Kargs::new();
    k.buf(out)
        .buf(ids)
        .buf(tok_emb)
        .i32(rows as i32)
        .i32(cols as i32);
    gpu::launch("embedding_forward_tok", cfg_1d(rows * cols), 0, &mut k);
}

pub fn embedding_backward_tok(
    dtok: &impl DevPtr,
    dout: &impl DevPtr,
    ids: &impl DevPtr,
    rows: usize,
    cols: usize,
) {
    let mut k = Kargs::new();
    k.buf(dtok)
        .buf(dout)
        .buf(ids)
        .i32(rows as i32)
        .i32(cols as i32);
    gpu::launch("embedding_backward_tok", cfg_1d(rows * cols), 0, &mut k);
}

// ============================================================
// KV-cache üretimi
// ============================================================

/// Tek satır softmax (maske yok).
pub fn softmax_vec(out: &impl DevPtr, x: &impl DevPtr, n: usize) {
    let threads = 128u32;
    let mut k = Kargs::new();
    k.buf(out).buf(x).i32(n as i32);
    gpu::launch(
        "softmax_vec",
        ((1, 1, 1), (threads, 1, 1)),
        threads * 4,
        &mut k,
    );
}

/// Belirli bir konum için RoPE (tek token).
pub fn rope_forward_pos(
    out: &impl DevPtr,
    x: &impl DevPtr,
    cos_t: &impl DevPtr,
    sin_t: &impl DevPtr,
    n_heads: usize,
    d_head: usize,
    pos: usize,
) {
    let total = n_heads * (d_head / 2);
    let mut k = Kargs::new();
    k.buf(out)
        .buf(x)
        .buf(cos_t)
        .buf(sin_t)
        .i32(n_heads as i32)
        .i32(d_head as i32)
        .i32(pos as i32);
    gpu::launch("rope_forward_pos", cfg_1d(total), 0, &mut k);
}

/// k/v vektörünü cache'e yaz (yerleşim: [head][capacity][d_head]).
#[allow(clippy::too_many_arguments)]
pub fn kv_store(
    cache_k: &impl DevPtr,
    cache_v: &impl DevPtr,
    k: &impl DevPtr,
    v: &impl DevPtr,
    n_heads: usize,
    d_head: usize,
    pos: usize,
    capacity: usize,
) {
    let mut a = Kargs::new();
    a.buf(cache_k)
        .buf(cache_v)
        .buf(k)
        .buf(v)
        .i32(n_heads as i32)
        .i32(d_head as i32)
        .i32(pos as i32)
        .i32(capacity as i32);
    gpu::launch("kv_store", cfg_1d(n_heads * d_head), 0, &mut a);
}
