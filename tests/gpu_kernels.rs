// tests/gpu_kernels.rs
//
// CUDA çekirdeklerini CPU referansına karşı doğrular. Referans olarak
// *doğrulanmış* autograd yolları kullanılır (aynı formülü iki kez yazıp
// iki kez yanlış yapma riskini ortadan kaldırır).

#![cfg(feature = "cuda")]

use stonefish_ai::sys::dev::{DevBuf, DevBufI32};
use stonefish_ai::autograd::Tensor;
use stonefish_ai::sys::dev::device_info;
use stonefish_ai::gpu_kernels as k;
use stonefish_ai::tensor::Matrix;

const TOL: f32 = 2e-4;

fn up(v: &[f32]) -> DevBuf {
    DevBuf::from_host(v)
}

fn up_i32(v: &[i32]) -> DevBufI32 {
    DevBufI32::from_host(v)
}

fn zeros(n: usize) -> DevBuf {
    DevBuf::zeros(n)
}

fn down(s: &DevBuf) -> Vec<f32> {
    s.to_host()
}

#[allow(dead_code)]
fn _touch_device_info() {
    let _ = device_info();
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "uzunluk farkı");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn rand_vec(n: usize, scale: f32, seed: &mut u64) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            (((*seed >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5) * scale
        })
        .collect()
}

// ------------------------------------------------------------
// GELU
// ------------------------------------------------------------

#[test]
fn gelu_kernels_match_cpu() {
    let mut s = 1u64;
    let n = 1000;
    let x_host = rand_vec(n, 6.0, &mut s);

    // Referans: autograd
    let x = Tensor::leaf(Matrix::from_vec(1, n, x_host.clone()), true);
    let y = x.gelu();
    let y_cpu = y.data.borrow().data.clone();
    y.sum().backward();
    let dx_cpu = x.grad.borrow().data.clone();

    // GPU ileri
    let x_dev = up(&x_host);
    let y_dev = zeros(n);
    k::gelu_forward(&y_dev, &x_dev, n);
    let d_fwd = max_diff(&down(&y_dev), &y_cpu);

    // GPU geri
    let dy_dev = up(&vec![1.0f32; n]);
    let dx_dev = zeros(n);
    k::gelu_backward(&dx_dev, &dy_dev, &x_dev, n);
    let d_bwd = max_diff(&down(&dx_dev), &dx_cpu);

    assert!(d_fwd < TOL, "gelu ileri farkı {d_fwd}");
    assert!(d_bwd < TOL, "gelu geri farkı {d_bwd}");
    println!("  ✅ gelu: ileri {d_fwd:.2e}, geri {d_bwd:.2e}");
}

// ------------------------------------------------------------
// LayerNorm
// ------------------------------------------------------------

#[test]
fn layernorm_kernels_match_cpu() {
    let mut s = 2u64;
    let (rows, cols) = (64, 32);
    let n = rows * cols;
    let x_host = rand_vec(n, 4.0, &mut s);
    let g_host = rand_vec(cols, 1.5, &mut s);
    let b_host = rand_vec(cols, 1.0, &mut s);
    let dy_host = rand_vec(n, 2.0, &mut s);

    let x = Tensor::leaf(Matrix::from_vec(rows, cols, x_host.clone()), true);
    let gamma = Tensor::leaf(Matrix::from_vec(1, cols, g_host.clone()), true);
    let beta = Tensor::leaf(Matrix::from_vec(1, cols, b_host.clone()), true);
    let ln = x.layer_norm(&gamma, &beta, 1e-5);
    let y_cpu = ln.data.borrow().data.clone();

    // Referans geri: dL/dy = dy_host olduğu için loss = sum(ln * dy)
    let dy_t = Tensor::leaf(Matrix::from_vec(rows, cols, dy_host.clone()), false);
    ln.mul_elem(&dy_t).sum().backward();
    let dx_cpu = x.grad.borrow().data.clone();
    let dg_cpu = gamma.grad.borrow().data.clone();
    let db_cpu = beta.grad.borrow().data.clone();

    // GPU
    let x_dev = up(&x_host);
    let g_dev = up(&g_host);
    let b_dev = up(&b_host);
    let dy_dev = up(&dy_host);
    let y_dev = zeros(n);
    let xhat_dev = zeros(n);
    let inv_dev = zeros(rows);
    k::layernorm_forward(
        &y_dev,
        &xhat_dev,
        &inv_dev,
        &x_dev,
        &g_dev,
        &b_dev,
        rows,
        cols,
        1e-5,
    );
    let d_fwd = max_diff(&down(&y_dev), &y_cpu);

    let dx_dev = zeros(n);
    k::layernorm_backward(&dx_dev, &dy_dev, &xhat_dev, &inv_dev, &g_dev, rows, cols);
    let d_bwd = max_diff(&down(&dx_dev), &dx_cpu);

    let dg_dev = zeros(cols);
    let db_dev = zeros(cols);
    k::layernorm_param_grad(&dg_dev, &db_dev, &dy_dev, &xhat_dev, rows, cols);
    let d_dg = max_diff(&down(&dg_dev), &dg_cpu);
    let d_db = max_diff(&down(&db_dev), &db_cpu);

    assert!(d_fwd < TOL, "LN ileri farkı {d_fwd}");
    assert!(d_bwd < TOL, "LN dx farkı {d_bwd}");
    assert!(d_dg < TOL, "LN dgamma farkı {d_dg}");
    assert!(d_db < TOL, "LN dbeta farkı {d_db}");
    println!("  ✅ layernorm: fwd {d_fwd:.2e}, dx {d_bwd:.2e}, dγ {d_dg:.2e}, dβ {d_db:.2e}");
}

// ------------------------------------------------------------
// Maskeli softmax (attention çekirdeği)
// ------------------------------------------------------------

#[test]
fn masked_softmax_kernels_match_cpu() {
    let mut s = 3u64;
    let (batch, heads, seq) = (2usize, 3usize, 8usize);
    let bh = batch * heads;
    let scale = 1.0 / (4.0f32).sqrt();
    let scores_host = rand_vec(bh * seq * seq, 4.0, &mut s);

    // CPU referansı: nedensel softmax
    let mut w_cpu = vec![0.0f32; bh * seq * seq];
    for b in 0..bh {
        for i in 0..seq {
            let base = b * seq * seq + i * seq;
            let mut mx = f32::NEG_INFINITY;
            for j in 0..=i {
                mx = mx.max(scores_host[base + j] * scale);
            }
            let mut sum = 0.0f32;
            for j in 0..=i {
                sum += (scores_host[base + j] * scale - mx).exp();
            }
            for j in 0..seq {
                w_cpu[base + j] = if j <= i {
                    (scores_host[base + j] * scale - mx).exp() / sum
                } else {
                    0.0
                };
            }
        }
    }

    let scores_dev = up(&scores_host);
    let w_dev = zeros(bh * seq * seq);
    k::masked_softmax_forward(&w_dev, &scores_dev, batch, heads, seq, scale);
    let d_fwd = max_diff(&down(&w_dev), &w_cpu);
    assert!(d_fwd < TOL, "masked softmax ileri farkı {d_fwd}");

    // Geri: ds = w*(dw - Σ w·dw)*scale
    let dw_host = rand_vec(bh * seq * seq, 2.0, &mut s);
    let mut ds_cpu = vec![0.0f32; bh * seq * seq];
    for b in 0..bh {
        for i in 0..seq {
            let base = b * seq * seq + i * seq;
            let mut dot = 0.0f32;
            for j in 0..=i {
                dot += w_cpu[base + j] * dw_host[base + j];
            }
            for j in 0..seq {
                ds_cpu[base + j] = if j <= i {
                    w_cpu[base + j] * (dw_host[base + j] - dot) * scale
                } else {
                    0.0
                };
            }
        }
    }

    let dw_dev = up(&dw_host);
    let ds_dev = zeros(bh * seq * seq);
    k::masked_softmax_backward(&ds_dev, &w_dev, &dw_dev, batch, heads, seq, scale);
    let d_bwd = max_diff(&down(&ds_dev), &ds_cpu);
    assert!(d_bwd < TOL, "masked softmax geri farkı {d_bwd}");
    println!("  ✅ masked softmax: fwd {d_fwd:.2e}, bwd {d_bwd:.2e}");
}

// ------------------------------------------------------------
// Cross-entropy
// ------------------------------------------------------------

#[test]
fn ce_kernels_match_cpu() {
    let mut s = 4u64;
    let (rows, classes) = (32usize, 17usize);
    let logits_host = rand_vec(rows * classes, 4.0, &mut s);
    let targets: Vec<usize> = (0..rows).map(|i| (i * 7 + 3) % classes).collect();
    let targets_i32: Vec<i32> = targets.iter().map(|&t| t as i32).collect();

    // Referans: autograd
    let logits = Tensor::leaf(Matrix::from_vec(rows, classes, logits_host.clone()), true);
    let loss_t = logits.softmax_cross_entropy_ids(&targets);
    let loss_cpu = loss_t.item();
    let probs_cpu = logits.softmax().data.borrow().data.clone();
    loss_t.backward();
    let dlogits_cpu = logits.grad.borrow().data.clone();

    // GPU
    let logits_dev = up(&logits_host);
    let targets_dev = up_i32(&targets_i32);
    let loss_dev = zeros(1);
    let probs_dev = zeros(rows * classes);
    k::ce_forward(
        &loss_dev,
        &probs_dev,
        &logits_dev,
        &targets_dev,
        rows,
        classes,
    );
    let loss_gpu = down(&loss_dev)[0];
    let d_loss = (loss_gpu - loss_cpu).abs();
    let d_probs = max_diff(&down(&probs_dev), &probs_cpu);

    let dlogits_dev = zeros(rows * classes);
    k::ce_backward(
        &dlogits_dev,
        &probs_dev,
        &targets_dev,
        rows,
        classes,
        1.0,
    );
    let d_grad = max_diff(&down(&dlogits_dev), &dlogits_cpu);

    assert!(d_loss < 1e-4, "CE loss farkı {d_loss}");
    assert!(d_probs < TOL, "CE probs farkı {d_probs}");
    assert!(d_grad < TOL, "CE gradyan farkı {d_grad}");
    println!("  ✅ cross-entropy: loss {d_loss:.2e}, probs {d_probs:.2e}, grad {d_grad:.2e}");
}

// ------------------------------------------------------------
// Embedding
// ------------------------------------------------------------

#[test]
fn embedding_kernels_match_cpu() {
    use stonefish_ai::transformer::Embedding;

    let mut s = 5u64;
    let (vocab, max_seq, d) = (11usize, 6usize, 5usize);
    let (batch, seq_len) = (2usize, 3usize);
    let rows = batch * seq_len;
    let ids: Vec<usize> = vec![1, 3, 0, 5, 2, 3];
    let ids_i32: Vec<i32> = ids.iter().map(|&i| i as i32).collect();

    let emb = Embedding::new(vocab, max_seq, d, &mut s);
    let tok_host = emb.token_emb.data.borrow().data.clone();
    let pos_host = emb.pos_emb.data.borrow().data.clone();
    let dy_host = rand_vec(rows * d, 2.0, &mut s);

    // Referans: autograd
    let dy_t = Tensor::leaf(Matrix::from_vec(rows, d, dy_host.clone()), false);
    let out = emb.forward(&ids, seq_len);
    let y_cpu = out.data.borrow().data.clone();
    out.mul_elem(&dy_t).sum().backward();
    let dtok_cpu = emb.token_emb.grad.borrow().data.clone();
    let dpos_cpu = emb.pos_emb.grad.borrow().data.clone();

    // GPU
    let tok_dev = up(&tok_host);
    let pos_dev = up(&pos_host);
    let ids_dev = up_i32(&ids_i32);
    let y_dev = zeros(rows * d);
    k::embedding_forward(&y_dev, &ids_dev, &tok_dev, &pos_dev, rows, d, seq_len);
    let d_fwd = max_diff(&down(&y_dev), &y_cpu);

    let dy_dev = up(&dy_host);
    let dtok_dev = zeros(vocab * d);
    let dpos_dev = zeros(max_seq * d);
    k::embedding_backward(
        &dtok_dev,
        &dpos_dev,
        &dy_dev,
        &ids_dev,
        rows,
        d,
        seq_len,
    );
    let d_tok = max_diff(&down(&dtok_dev), &dtok_cpu);
    let d_pos = max_diff(&down(&dpos_dev), &dpos_cpu);

    assert!(d_fwd < TOL, "embedding ileri farkı {d_fwd}");
    assert!(d_tok < TOL, "embedding dtok farkı {d_tok}");
    assert!(d_pos < TOL, "embedding dpos farkı {d_pos}");
    println!("  ✅ embedding: fwd {d_fwd:.2e}, dtok {d_tok:.2e}, dpos {d_pos:.2e}");
}

// ------------------------------------------------------------
// Adam
// ------------------------------------------------------------

#[test]
fn adam_kernel_matches_cpu() {
    use stonefish_ai::optim::Adam;

    let mut s = 6u64;
    let n = 257; // tek blok sınırında değil
    let p_host = rand_vec(n, 2.0, &mut s);
    let g_host = rand_vec(n, 0.5, &mut s);
    let lr = 1e-3f32;

    // Referans: optim::Adam ile 3 adım (aynı gradyan)
    let p = Tensor::leaf(Matrix::from_vec(1, n, p_host.clone()), true);
    let mut opt = Adam::new(vec![p.clone()], lr);
    for _ in 0..3 {
        *p.grad.borrow_mut() = Matrix::from_vec(1, n, g_host.clone());
        opt.step();
    }
    let p_cpu = p.data.borrow().data.clone();

    // GPU
    let p_dev = up(&p_host);
    let g_dev = up(&g_host);
    let m_dev = zeros(n);
    let v_dev = zeros(n);
    let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-6f32);
    for t in 1..=3usize {
        let bc1 = 1.0 - b1.powi(t as i32);
        let bc2 = 1.0 - b2.powi(t as i32);
        k::adam_step(
            &p_dev, &g_dev, &m_dev, &v_dev, lr, b1, b2, eps, bc1, bc2, n,
        );
    }
    let d = max_diff(&down(&p_dev), &p_cpu);
    assert!(d < 1e-6, "adam farkı {d}");
    println!("  ✅ adam: max fark {d:.2e}");
}

// ------------------------------------------------------------
// Cihaz-üstü (batched) GEMM'ler
// ------------------------------------------------------------

#[test]
fn dev_gemm_modes_match_cpu() {
    use stonefish_ai::gpu::GemmMode;

    let mut s = 8u64;
    // Farklı şekiller + "geniş satır adımı" (view benzetimi: row > cols)
    let (m, n, k) = (6usize, 5usize, 4usize);

    // A: m×k, satır adımı 8 (k'dan büyük); B: k×n satır adımı 7
    let (ar, br) = (8usize, 7usize);
    let a_buf = rand_vec(m * ar, 2.0, &mut s);
    let b_buf = rand_vec(k * br, 2.0, &mut s);
    let mut c = vec![0.0f32; m * n];

    let a_dev = up(&a_buf);
    let b_dev = up(&b_buf);
    let mut c_dev = zeros(m * n);

    // NN: C = A·B
    stonefish_ai::gpu::dev_gemm(&mut c_dev, &a_dev, &b_dev, m, n, k, ar, br, n, GemmMode::Nn);
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a_buf[i * ar + kk] * b_buf[kk * br + j];
            }
            c[i * n + j] = acc;
        }
    }
    let d1 = max_diff(&down(&c_dev), &c);

    // NT: C = A·Bᵀ  (B'yi n×k gibi kullan)
    let b_nt = rand_vec(n * br, 2.0, &mut s);
    let b_nt_dev = up(&b_nt);
    stonefish_ai::gpu::dev_gemm(&mut c_dev, &a_dev, &b_nt_dev, m, n, k, ar, br, n, GemmMode::Nt);
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a_buf[i * ar + kk] * b_nt[j * br + kk];
            }
            c[i * n + j] = acc;
        }
    }
    let d2 = max_diff(&down(&c_dev), &c);

    // TN: C = Aᵀ·B  (A'yı k×m gibi kullan)
    let a_tn = rand_vec(k * ar, 2.0, &mut s);
    let a_tn_dev = up(&a_tn);
    stonefish_ai::gpu::dev_gemm(&mut c_dev, &a_tn_dev, &b_dev, m, n, k, ar, br, n, GemmMode::Tn);
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a_tn[kk * ar + i] * b_buf[kk * br + j];
            }
            c[i * n + j] = acc;
        }
    }
    let d3 = max_diff(&down(&c_dev), &c);

    assert!(d1 < TOL, "GEMM NN farkı {d1}");
    assert!(d2 < TOL, "GEMM NT farkı {d2}");
    assert!(d3 < TOL, "GEMM TN farkı {d3}");
    println!("  ✅ dev_gemm: NN {d1:.2e}, NT {d2:.2e}, TN {d3:.2e}");
}

#[test]
fn dev_batched_gemm_matches_cpu() {
    use stonefish_ai::gpu::GemmMode;

    let mut s = 9u64;
    let (batch, m, n, k) = (3usize, 5usize, 4usize, 3usize);
    let (ar, br, cr) = (6usize, 5usize, 4usize); // row > cols (view benzetimi)
    let (as_, _bs_, cs_) = (m * ar, k * br, m * cr);

    let a_buf = rand_vec(batch * as_, 2.0, &mut s);
    let b_nt = rand_vec(batch * n * br, 2.0, &mut s); // B: n×k
    let a_nt = up(&a_buf);
    let b_nt_dev = up(&b_nt);
    let mut c_dev = zeros(batch * cs_);

    // NT batched: C_i = A_i·B_iᵀ
    stonefish_ai::gpu::dev_gemm_batched(
        &mut c_dev,
        &a_nt,
        &b_nt_dev,
        m,
        n,
        k,
        ar,
        br,
        cr,
        as_,
        n * br,
        cs_,
        batch,
        0.0,
        GemmMode::Nt,
    );
    let got = down(&c_dev);
    let mut d = 0.0f32;
    for bi in 0..batch {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += a_buf[bi * as_ + i * ar + kk] * b_nt[bi * n * br + j * br + kk];
                }
                let g = got[bi * cs_ + i * cr + j];
                d = d.max((g - acc).abs());
            }
        }
    }
    assert!(d < TOL, "batched NT farkı {d}");
    println!("  ✅ dev_batched_gemm: NT {d:.2e}");
}

#[test]
fn batched_attention_scores_with_head_views() {
    // Gerçek kullanım: q/k (B*S, D) matrisleri, head dilimleri view offset'iyle
    // (h*d_head) alınır, batch üzerinde strided. Skorlar [head][batch][S][S].
    use stonefish_ai::gpu::GemmMode;

    let mut s = 11u64;
    let (b, seq, d, nh) = (2usize, 4usize, 8usize, 2usize);
    let dh = d / nh;
    let bs = b * seq;

    let q_host = rand_vec(bs * d, 2.0, &mut s);
    let k_host = rand_vec(bs * d, 2.0, &mut s);
    let q_dev = up(&q_host);
    let k_dev = up(&k_host);
    let mut scores = zeros(nh * b * seq * seq);

    for h in 0..nh {
        let off = h * dh;
        let q_view = q_dev.slice(off..);
        let k_view = k_dev.slice(off..);
        let mut sc = scores.slice_mut(h * b * seq * seq..);
        stonefish_ai::gpu::dev_gemm_batched(
            &mut sc,
            &q_view,
            &k_view,
            seq,
            seq,
            dh,
            d,
            d,
            seq,
            seq * d,
            seq * d,
            seq * seq,
            b,
            0.0,
            GemmMode::Nt,
        );
    }

    let got = down(&scores);
    let mut worst = 0.0f32;
    let mut detail = String::new();
    for h in 0..nh {
        for bi in 0..b {
            for i in 0..seq {
                for j in 0..seq {
                    let mut acc = 0.0f32;
                    for t in 0..dh {
                        acc += q_host[(bi * seq + i) * d + h * dh + t]
                            * k_host[(bi * seq + j) * d + h * dh + t];
                    }
                    let idx = ((h * b + bi) * seq + i) * seq + j;
                    let diff = (got[idx] - acc).abs();
                    if diff > worst {
                        worst = diff;
                        detail = format!("h={h} b={bi} i={i} j={j}: gpu={} cpu={acc}", got[idx]);
                    }
                }
            }
        }
    }
    assert!(worst < TOL, "attention skor farkı {worst} ({detail})");
    println!("  ✅ attention skorları (head view + strided batch): {worst:.2e}");
}

#[test]
fn attention_backward_composition_matches_cpu() {
    // GPU eğiticisinin attention geri yayılımını (dV, dW, softmax, dQ, dK)
    // birebir kurar ve CPU autograd'ın causal_attention geri yayılımıyla
    // karşılaştırır. View/strided çağrıların bileşimi burada sınanıyor.
    use stonefish_ai::gpu::GemmMode;

    let mut s = 12u64;
    let (b, seq, d, nh) = (2usize, 4usize, 8usize, 2usize);
    let dh = d / nh;
    let bs = b * seq;
    let scale = 1.0 / (dh as f32).sqrt();

    let q_host = rand_vec(bs * d, 1.0, &mut s);
    let k_host = rand_vec(bs * d, 1.0, &mut s);
    let v_host = rand_vec(bs * d, 1.0, &mut s);
    let da_host = rand_vec(bs * d, 1.0, &mut s); // dAttn

    // --- CPU referansı ---
    let q = Tensor::leaf(Matrix::from_vec(bs, d, q_host.clone()), true);
    let k = Tensor::leaf(Matrix::from_vec(bs, d, k_host.clone()), true);
    let v = Tensor::leaf(Matrix::from_vec(bs, d, v_host.clone()), true);
    let da = Tensor::leaf(Matrix::from_vec(bs, d, da_host.clone()), false);
    let attn = q.causal_attention(&k, &v, b, seq, nh);
    attn.mul_elem(&da).sum().backward();
    let dq_cpu = q.grad.borrow().data.clone();
    let dk_cpu = k.grad.borrow().data.clone();
    let dv_cpu = v.grad.borrow().data.clone();
    let attn_cpu = attn.data.borrow().data.clone(); // attention ÇIKTISI (W·V)

    // --- GPU: aynı bileşim ---
    let q_dev = up(&q_host);
    let k_dev = up(&k_host);
    let v_dev = up(&v_host);
    let da_dev = up(&da_host);
    let mut scores = zeros(nh * b * seq * seq);
    let w = zeros(nh * b * seq * seq);
    let mut dscores = zeros(nh * b * seq * seq);
    let mut dq = zeros(bs * d);
    let mut dk = zeros(bs * d);
    let mut dv = zeros(bs * d);

    // ileri: skorlar + softmax
    for h in 0..nh {
        let off = h * dh;
        let q_view = q_dev.slice(off..);
        let k_view = k_dev.slice(off..);
        let mut sc = scores.slice_mut(h * b * seq * seq..);
        stonefish_ai::gpu::dev_gemm_batched(
            &mut sc, &q_view, &k_view, seq, seq, dh, d, d, seq, seq * d, seq * d,
            seq * seq, b, 0.0, GemmMode::Nt,
        );
    }
    k::masked_softmax_forward(&w, &scores, b, nh, seq, scale);

    // attention çıktısı: attn = W·V
    let mut attn = zeros(bs * d);
    for h in 0..nh {
        let off = h * dh;
        let w_view = w.slice(h * b * seq * seq..);
        let v_view = v_dev.slice(off..);
        let mut av = attn.slice_mut(off..);
        stonefish_ai::gpu::dev_gemm_batched(
            &mut av, &w_view, &v_view, seq, dh, seq, seq, d, d, seq * seq,
            seq * d, seq * d, b, 0.0, GemmMode::Nn,
        );
    }

    // geri: dV
    for h in 0..nh {
        let off = h * dh;
        let w_view = w.slice(h * b * seq * seq..);
        let da_view = da_dev.slice(off..);
        let mut dv_view = dv.slice_mut(off..);
        stonefish_ai::gpu::dev_gemm_batched(
            &mut dv_view, &w_view, &da_view, seq, dh, seq, seq, d, d, seq * seq,
            seq * d, seq * d, b, 0.0, GemmMode::Tn,
        );
    }
    // dW
    for h in 0..nh {
        let off = h * dh;
        let da_view = da_dev.slice(off..);
        let v_view = v_dev.slice(off..);
        let mut dw_view = dscores.slice_mut(h * b * seq * seq..);
        stonefish_ai::gpu::dev_gemm_batched(
            &mut dw_view, &da_view, &v_view, seq, seq, dh, d, d, seq, seq * d,
            seq * d, seq * seq, b, 0.0, GemmMode::Nt,
        );
    }
    // softmax geri (dscores yerinde değil: ayrı tampon)
    let ds = zeros(nh * b * seq * seq);
    k::masked_softmax_backward(&ds, &w, &dscores, b, nh, seq, scale);

    // dQ, dK
    for h in 0..nh {
        let off = h * dh;
        let ds_view = ds.slice(h * b * seq * seq..);
        let k_view = k_dev.slice(off..);
        let q_view = q_dev.slice(off..);
        {
            let mut dq_view = dq.slice_mut(off..);
            stonefish_ai::gpu::dev_gemm_batched(
                &mut dq_view, &ds_view, &k_view, seq, dh, seq, seq, d, d,
                seq * seq, seq * d, seq * d, b, 0.0, GemmMode::Nn,
            );
        }
        {
            let mut dk_view = dk.slice_mut(off..);
            stonefish_ai::gpu::dev_gemm_batched(
                &mut dk_view, &ds_view, &q_view, seq, dh, seq, seq, d, d,
                seq * seq, seq * d, seq * d, b, 0.0, GemmMode::Tn,
            );
        }
    }

    let attn_gpu = down(&attn);
    let dq_got = down(&dq);
    let dk_got = down(&dk);
    let dv_got = down(&dv);

    let d_w = max_diff(&attn_gpu, &attn_cpu);
    let d_q = max_diff(&dq_got, &dq_cpu);
    let d_k = max_diff(&dk_got, &dk_cpu);
    let d_v = max_diff(&dv_got, &dv_cpu);
    println!("  attn {d_w:.2e}  dQ {d_q:.2e}  dK {d_k:.2e}  dV {d_v:.2e}");
    assert!(d_w < TOL, "attention çıktı farkı {d_w}");
    assert!(d_q < TOL, "dQ farkı {d_q}");
    assert!(d_k < TOL, "dK farkı {d_k}");
    assert!(d_v < TOL, "dV farkı {d_v}");
}

#[test]
fn gemm_tn_and_beta_accumulation_real_shapes() {
    // block_backward'ın iki kritik yolu:
    //   dwo = attnᵀ·dOut   (Tn, (BS,D)ᵀ·(BS,D) → (D,D))
    //   dln1 = dQ·wqᵀ + dK·wkᵀ + dV·wvᵀ  (beta=1 ile biriktirme)
    use stonefish_ai::gpu::GemmMode;

    let mut s = 13u64;
    let (bs, d) = (8usize, 8usize);
    let attn = rand_vec(bs * d, 1.0, &mut s);
    let dout = rand_vec(bs * d, 1.0, &mut s);
    let a_dev = up(&attn);
    let b_dev = up(&dout);

    let mut c = zeros(d * d);
    stonefish_ai::gpu::dev_gemm(&mut c, &a_dev, &b_dev, d, d, bs, d, d, d, GemmMode::Tn);
    let got = down(&c);
    let mut worst = 0.0f32;
    for i in 0..d {
        for j in 0..d {
            let mut acc = 0.0f32;
            for kk in 0..bs {
                acc += attn[kk * d + i] * dout[kk * d + j];
            }
            worst = worst.max((got[i * d + j] - acc).abs());
        }
    }
    assert!(worst < TOL, "Tn (dwo) farkı {worst}");

    // beta biriktirme: C = A1·B1ᵀ + A2·B2ᵀ + A3·B3ᵀ
    let cols = 8usize;
    let x1 = rand_vec(bs * cols, 1.0, &mut s);
    let x2 = rand_vec(bs * cols, 1.0, &mut s);
    let x3 = rand_vec(bs * cols, 1.0, &mut s);
    let w1 = rand_vec(cols * cols, 1.0, &mut s);
    let w2 = rand_vec(cols * cols, 1.0, &mut s);
    let w3 = rand_vec(cols * cols, 1.0, &mut s);
    let (x1d, x2d, x3d) = (up(&x1), up(&x2), up(&x3));
    let (w1d, w2d, w3d) = (up(&w1), up(&w2), up(&w3));
    let mut acc = zeros(bs * cols);
    for (i, (xd, wd)) in [(x1d, w1d), (x2d, w2d), (x3d, w3d)].iter().enumerate() {
        let beta = if i == 0 { 0.0 } else { 1.0 };
        stonefish_ai::gpu::dev_gemm_batched(
            &mut acc, xd, wd, bs, cols, cols, cols, cols, cols, 0, 0, 0, 1, beta,
            GemmMode::Nt,
        );
    }
    let got = down(&acc);
    let mut worst2 = 0.0f32;
    for r in 0..bs {
        for j in 0..cols {
            let dot = |x: &[f32], w: &[f32]| -> f32 {
                (0..cols).map(|kk| x[r * cols + kk] * w[j * cols + kk]).sum()
            };
            let want = dot(&x1, &w1) + dot(&x2, &w2) + dot(&x3, &w3);
            worst2 = worst2.max((got[r * cols + j] - want).abs());
        }
    }
    assert!(worst2 < TOL, "beta biriktirme farkı {worst2}");
    println!("  ✅ Tn dwo {worst:.2e}, beta-accumulate {worst2:.2e}");
}

// ------------------------------------------------------------
// Yardımcı çekirdekler
// ------------------------------------------------------------

#[test]
fn elementwise_and_reduction_kernels() {
    let mut s = 7u64;
    let (rows, cols) = (10usize, 6usize);
    let n = rows * cols;
    let a = rand_vec(n, 2.0, &mut s);
    let bias = rand_vec(cols, 1.0, &mut s);

    // add_bias
    let a_dev = up(&a);
    let bias_dev = up(&bias);
    let out_dev = zeros(n);
    k::add_bias(&out_dev, &a_dev, &bias_dev, rows, cols);
    let mut ref_out = vec![0.0f32; n];
    for r in 0..rows {
        for c in 0..cols {
            ref_out[r * cols + c] = a[r * cols + c] + bias[c];
        }
    }
    let d1 = max_diff(&down(&out_dev), &ref_out);
    assert!(d1 < 1e-6, "add_bias farkı {d1}");

    // add_inplace
    let acc_dev = up(&a);
    k::add_inplace(&acc_dev, &a_dev, n);
    let ref2: Vec<f32> = a.iter().map(|v| v * 2.0).collect();
    let d2 = max_diff(&down(&acc_dev), &ref2);
    assert!(d2 < 1e-6, "add_inplace farkı {d2}");

    // zero_fill
    let z_dev = up(&a);
    k::zero_fill(&z_dev, n);
    let d3 = down(&z_dev).iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(d3 == 0.0, "zero_fill farkı {d3}");

    // sum_squares + scale_inplace
    let acc = zeros(1);
    k::sum_squares(&acc, &a_dev, n);
    let ref_sq: f32 = a.iter().map(|v| v * v).sum();
    let d4 = (down(&acc)[0] - ref_sq).abs();
    assert!(d4 < 1e-2, "sum_squares farkı {d4} (ref {ref_sq})");

    let sc_dev = up(&a);
    k::scale_inplace(&sc_dev, n, 0.5);
    let ref3: Vec<f32> = a.iter().map(|v| v * 0.5).collect();
    let d5 = max_diff(&down(&sc_dev), &ref3);
    assert!(d5 < 1e-6, "scale_inplace farkı {d5}");

    println!("  ✅ elementwise/reduction: {d1:.1e} {d2:.1e} {d3:.1e} {d4:.1e} {d5:.1e}");
}

// ============================================================
// Yeni mimari çekirdekleri (RMSNorm, SwiGLU, RoPE, dropout)
// ============================================================

/// Basit xorshift — test verisi üretimi için.
fn rnd(n: usize, seed: &mut u64, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            (((*seed >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5) * 2.0 * scale
        })
        .collect()
}

fn rmsnorm_cpu(x: &[f32], g: &[f32], rows: usize, cols: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        let mut ss = 0.0f32;
        for i in 0..cols {
            ss += x[r * cols + i] * x[r * cols + i];
        }
        let inv = 1.0 / (ss / cols as f32 + eps).sqrt();
        for i in 0..cols {
            out[r * cols + i] = x[r * cols + i] * inv * g[i];
        }
    }
    out
}

#[test]
fn rmsnorm_forward_matches_cpu() {
    let (rows, cols) = (5usize, 16usize);
    let mut s = 11u64;
    let x = rnd(rows * cols, &mut s, 2.0);
    let g = rnd(cols, &mut s, 1.0).iter().map(|v| 1.0 + v * 0.3).collect::<Vec<_>>();
    let eps = 1e-6;

    let (dx, dg, dxi) = (up(&x), up(&g), zeros(rows));
    let out = zeros(rows * cols);
    let xhat = zeros(rows * cols);
    k::rmsnorm_forward(&out, &xhat, &dxi, &dx, &dg, rows, cols, eps);
    assert!(max_diff(&down(&out), &rmsnorm_cpu(&x, &g, rows, cols, eps)) < TOL);

    // xhat'in karesel ortalaması ~1 olmalı (ölçek doğru mu)
    let xh = down(&xhat);
    for r in 0..rows {
        let ms: f32 = (0..cols).map(|i| xh[r * cols + i] * xh[r * cols + i]).sum::<f32>() / cols as f32;
        assert!((ms - 1.0).abs() < 1e-3, "xhat rms {ms}");
    }

    // dx: sayısal türevle karşılaştır (loss = sum(out * w))
    let w = rnd(rows * cols, &mut s, 1.0);
    let f = |xv: &[f32]| -> f32 {
        rmsnorm_cpu(xv, &g, rows, cols, eps)
            .iter()
            .zip(w.iter())
            .map(|(a, b)| a * b)
            .sum()
    };
    let eps_num = 1e-3;
    let mut dx_num = vec![0f32; rows * cols];
    for i in 0..rows * cols {
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[i] += eps_num;
        xm[i] -= eps_num;
        dx_num[i] = (f(&xp) - f(&xm)) / (2.0 * eps_num);
    }
    // dgamma = sum_r dy*xhat ; dx çekirdeği dy = w ile
    let dy = up(&w);
    let dx_out = zeros(rows * cols);
    let dgam = zeros(cols);
    k::rmsnorm_backward(&dx_out, &dy, &xhat, &dxi, &dg, rows, cols);
    k::rmsnorm_param_grad(&dgam, &dy, &xhat, rows, cols);
    let got = down(&dx_out);
    let diff = max_diff(&got, &dx_num);
    assert!(diff < 2e-3, "rmsnorm dx farkı {diff}");

    let mut dg_num = vec![0f32; cols];
    let fg = |gv: &[f32]| -> f32 {
        rmsnorm_cpu(&x, gv, rows, cols, eps)
            .iter()
            .zip(w.iter())
            .map(|(a, b)| a * b)
            .sum()
    };
    for i in 0..cols {
        let mut gp = g.clone();
        let mut gm = g.clone();
        gp[i] += eps_num;
        gm[i] -= eps_num;
        dg_num[i] = (fg(&gp) - fg(&gm)) / (2.0 * eps_num);
    }
    let dgd = max_diff(&down(&dgam), &dg_num);
    assert!(dgd < 2e-3, "rmsnorm dgamma farkı {dgd}");
}

#[test]
fn silu_mul_matches_cpu() {
    let n = 64usize;
    let mut s = 21u64;
    let gate = rnd(n, &mut s, 3.0);
    let up_v = rnd(n, &mut s, 2.0);
    let g = up(&gate);
    let u = up(&up_v);
    let out = zeros(n);
    k::silu_mul_forward(&out, &g, &u, n);

    let cpu: Vec<f32> = gate
        .iter()
        .zip(up_v.iter())
        .map(|(a, b)| (a / (1.0 + (-a).exp())) * b)
        .collect();
    assert!(max_diff(&down(&out), &cpu) < TOL);

    // geri yayılım: dy = 1 → dgate = up*silu'(gate), dup = silu(gate)
    let dy = up(&vec![1.0f32; n]);
    let dgate = zeros(n);
    let dup = zeros(n);
    k::silu_mul_backward(&dgate, &dup, &dy, &g, &u, n);
    let dg_cpu: Vec<f32> = gate
        .iter()
        .zip(up_v.iter())
        .map(|(a, b)| {
            let sig = 1.0 / (1.0 + (-a).exp());
            b * sig * (1.0 + a * (1.0 - sig))
        })
        .collect();
    let du_cpu: Vec<f32> = gate.iter().map(|a| a / (1.0 + (-a).exp())).collect();
    assert!(max_diff(&down(&dgate), &dg_cpu) < TOL, "dgate");
    assert!(max_diff(&down(&dup), &du_cpu) < TOL, "dup");
}

#[test]
fn rope_forward_and_inverse() {
    let (batch, seq, n_heads, d_head) = (2usize, 6usize, 2usize, 8usize);
    let n = batch * seq * n_heads * d_head;
    let mut s = 33u64;
    let x = rnd(n, &mut s, 1.0);

    let xd = up(&x);
    let rot = zeros(n);
    k::rope_forward(&rot, &xd, batch, seq, n_heads, d_head);
    let fwd = down(&rot);

    // pozitif norm korunmalı (dönme izometridir) ve 0. konumda değişim olmamalı
    for r in 0..(batch * seq) {
        if r % seq == 0 {
            for i in 0..(n_heads * d_head) {
                assert!((fwd[r * n_heads * d_head + i] - x[r * n_heads * d_head + i]).abs() < 1e-6);
            }
        }
    }
    let norm = |v: &[f32]| v.iter().map(|a| a * a).sum::<f32>();
    assert!((norm(&fwd) - norm(&x)).abs() / norm(&x) < 1e-4, "norm korunmadı");

    // geri yayılım dönmeyi tersine çevirir → orijinali geri vermeli
    let back = zeros(n);
    k::rope_backward(&back, &rot, batch, seq, n_heads, d_head);
    let b = down(&back);
    let d = max_diff(&b, &x);
    assert!(d < 1e-4, "rope ters dönüş farkı {d}");
}

#[test]
fn dropout_scales_and_masks() {
    let n = 10000usize;
    let x = vec![1.0f32; n];
    let xd = up(&x);
    let p = 0.25f32;
    k::dropout_inplace(&xd, n, p, 12345);
    let out = down(&xd);
    let zeros_n = out.iter().filter(|v| **v == 0.0).count();
    let kept = out.iter().filter(|v| **v > 0.0).count();
    let frac = zeros_n as f32 / n as f32;
    assert!((frac - p).abs() < 0.03, "dropout oranı {frac} (beklenen {p})");
    let expected = 1.0 / (1.0 - p);
    for v in out.iter().filter(|v| **v > 0.0).take(100) {
        assert!((v - expected).abs() < 1e-5, "ölçek {v} != {expected}");
    }
    assert!(kept > 0);

    // geri yayılım aynı maskeyi uygulamalı
    let dx = up(&vec![1.0f32; n]);
    k::dropout_backward(&dx, n, p, 12345);
    let d = down(&dx);
    for i in 0..n {
        if out[i] == 0.0 {
            assert_eq!(d[i], 0.0, "maske tutarsız at {i}");
        } else {
            assert!((d[i] - expected).abs() < 1e-5);
        }
    }
}
