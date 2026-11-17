// tests/gradcheck.rs
//
// Numerical gradient verification by central difference.
//
// For every op: the analytic gradient (backward) is compared against the
// numerical gradient ((L(t+e) - L(t-e)) / 2e).
// Goal: lock down the correctness of the hand-derived backward formulas.

// In this file inner loops index several arrays at once (see src/lib.rs).
#![allow(clippy::needless_range_loop)]

use stonefish_ai::autograd::Tensor;
use stonefish_ai::tensor::Matrix;

const EPS: f32 = 1e-3;
const TOL: f32 = 2e-2;

/// For single-parameter ops: a clone-free slice instead of `&[t.clone()]`.
fn one(t: &Tensor) -> &[Tensor] {
    std::slice::from_ref(t)
}

fn scalar(t: &Tensor) -> f32 {
    let d = t.data.borrow();
    assert_eq!(
        (d.rows, d.cols),
        (1, 1),
        "gradcheck: the loss must be a 1x1 scalar, got {}x{}",
        d.rows,
        d.cols
    );
    d.data[0]
}

/// `build` must produce a scalar loss from the given leaves.
/// Every element of every parameter is compared analytically/numerically.
fn gradient_check<F>(name: &str, params: &[Tensor], build: F)
where
    F: Fn() -> Tensor,
{
    // --- analitik ---
    for p in params {
        p.zero_grad();
    }
    let loss = build();
    let _ = scalar(&loss);
    loss.backward();
    let analytic: Vec<Vec<f32>> = params
        .iter()
        .map(|p| p.grad.borrow().data.clone())
        .collect();

    // --- numerik ---
    let mut worst_rel = 0.0f32;
    let mut worst_msg = String::new();
    for (pi, p) in params.iter().enumerate() {
        let n = p.data.borrow().data.len();
        for k in 0..n {
            let orig = p.data.borrow().data[k];

            p.data.borrow_mut().data[k] = orig + EPS;
            let l_plus = scalar(&build());

            p.data.borrow_mut().data[k] = orig - EPS;
            let l_minus = scalar(&build());

            p.data.borrow_mut().data[k] = orig;

            let numeric = (l_plus - l_minus) / (2.0 * EPS);
            let a = analytic[pi][k];
            let denom = a.abs().max(numeric.abs()).max(1.0);
            let rel = (a - numeric).abs() / denom;
            if rel > worst_rel {
                worst_rel = rel;
                worst_msg = format!(
                    "param[{pi}] element {k}: analytic={a:.6} numeric={numeric:.6} (shape {}x{})",
                    p.data.borrow().rows,
                    p.data.borrow().cols
                );
            }
        }
    }

    assert!(
        worst_rel <= TOL,
        "{name}: gradient mismatch - worst rel diff {worst_rel:.3e} > {TOL:e}\n  {worst_msg}"
    );
    println!("  ✅ {name:<28} worst rel diff {worst_rel:.2e}");
}

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------

fn leaf(rows: usize, cols: usize, seed: &mut u64) -> Tensor {
    Tensor::leaf(Matrix::random(rows, cols, 1.0, seed), true)
}

fn frozen(rows: usize, cols: usize, seed: &mut u64) -> Tensor {
    Tensor::leaf(Matrix::random(rows, cols, 1.0, seed), false)
}

/// For ReLU tests: |x| >= 0.6 (eps = 1e-3 cannot cross zero).
fn leaf_away_from_zero(rows: usize, cols: usize, seed: &mut u64) -> Tensor {
    Tensor::leaf(
        Matrix::random(rows, cols, 1.0, seed).map(|x| if x >= 0.0 { x + 0.6 } else { x - 0.6 }),
        true,
    )
}

fn frozen_positive(rows: usize, cols: usize, seed: &mut u64) -> Tensor {
    Tensor::leaf(
        Matrix::random(rows, cols, 1.0, seed).map(|x| x.abs() + 0.2),
        false,
    )
}

// ------------------------------------------------------------
// Element-wise op'lar
// ------------------------------------------------------------

#[test]
fn add_same_shape() {
    let mut s = 11u64;
    let a = leaf(3, 4, &mut s);
    let b = leaf(3, 4, &mut s);
    gradient_check("add (same shape)", &[a.clone(), b.clone()], || {
        a.add(&b).sum()
    });
}

#[test]
fn add_broadcast_bias() {
    let mut s = 12u64;
    let a = leaf(5, 4, &mut s);
    let bias = leaf(1, 4, &mut s); // the path Linear::forward uses
    gradient_check("add (bias broadcast)", &[a.clone(), bias.clone()], || {
        a.add(&bias).sum()
    });
}

#[test]
fn sub_same_shape() {
    let mut s = 13u64;
    let a = leaf(3, 4, &mut s);
    let b = leaf(3, 4, &mut s);
    gradient_check("sub (same shape)", &[a.clone(), b.clone()], || {
        a.sub(&b).sum()
    });
}

#[test]
fn sub_broadcast() {
    let mut s = 14u64;
    let a = leaf(5, 4, &mut s);
    let b = leaf(1, 4, &mut s);
    gradient_check("sub (bias broadcast)", &[a.clone(), b.clone()], || {
        a.sub(&b).sum()
    });
}

#[test]
fn mul_elem() {
    let mut s = 15u64;
    let a = leaf(3, 4, &mut s);
    let b = leaf(3, 4, &mut s);
    gradient_check("mul_elem", &[a.clone(), b.clone()], || a.mul_elem(&b).sum());
}

#[test]
fn scale_op() {
    let mut s = 16u64;
    let a = leaf(3, 4, &mut s);
    gradient_check("scale", one(&a), || a.scale(2.75).sum());
}

#[test]
fn relu() {
    let mut s = 17u64;
    let a = leaf_away_from_zero(3, 5, &mut s);
    gradient_check("relu", one(&a), || a.relu().sum());
}

#[test]
fn sigmoid() {
    let mut s = 18u64;
    let a = leaf(3, 5, &mut s);
    gradient_check("sigmoid", one(&a), || a.sigmoid().sum());
}

// ------------------------------------------------------------
// Lineer cebir
// ------------------------------------------------------------

#[test]
fn matmul() {
    let mut s = 19u64;
    let a = leaf(4, 3, &mut s);
    let b = leaf(3, 5, &mut s);
    gradient_check("matmul", &[a.clone(), b.clone()], || a.matmul(&b).sum());
}

#[test]
fn matmul_square() {
    // Square matmul: the transposed backward paths (matmul_ta/bt) are the
    // hardest place to catch an lda/ldb mix-up - checked separately too.
    let mut s = 19_000u64;
    let n = 8;
    let a = leaf(n, n, &mut s);
    let b = leaf(n, n, &mut s);
    let w = frozen(n, n, &mut s);
    gradient_check("matmul (kare 8x8)", &[a.clone(), b.clone()], || {
        a.matmul(&b).mul_elem(&w).sum()
    });
}

#[test]
fn matmul_then_transpose() {
    let mut s = 20u64;
    let a = leaf(4, 3, &mut s); // (4,3)
    let b = leaf(6, 3, &mut s); // (6,3) -> a @ b^T = (4,6)  [attention scores]
    let w = frozen(4, 6, &mut s);
    gradient_check("matmul + transpose", &[a.clone(), b.clone()], || {
        a.matmul(&b.transpose()).mul_elem(&w).sum()
    });
}

#[test]
fn transpose_op() {
    let mut s = 21u64;
    let a = leaf(3, 4, &mut s);
    let w = frozen(4, 3, &mut s);
    gradient_check("transpose", one(&a), || a.transpose().mul_elem(&w).sum());
}

#[test]
fn sum_op() {
    let mut s = 22u64;
    let a = leaf(3, 4, &mut s);
    gradient_check("sum", one(&a), || a.sum().scale(2.0));
}

// ------------------------------------------------------------
// Loss / probability
// ------------------------------------------------------------

#[test]
fn softmax_cross_entropy() {
    let mut s = 23u64;
    let logits = leaf(4, 6, &mut s);
    let mut t = vec![0.0f32; 4 * 6];
    for (i, &cls) in [2usize, 0, 5, 3].iter().enumerate() {
        t[i * 6 + cls] = 1.0;
    }
    let target = Tensor::leaf(Matrix::from_vec(4, 6, t), false);
    gradient_check("softmax_cross_entropy", one(&logits), || {
        logits.softmax_cross_entropy(&target)
    });
}

#[test]
fn softmax_op() {
    let mut s = 24u64;
    let a = leaf(3, 5, &mut s);
    // A plain sum is constant (1) so the gradient is zero; use a weighted sum.
    let w = frozen(3, 5, &mut s);
    gradient_check("softmax", one(&a), || a.softmax().mul_elem(&w).sum());
}

#[test]
fn mask_causal() {
    let mut s = 25u64;
    let a = leaf(4, 4, &mut s);

    // WARNING: mask_causal sets the upper triangle to -1e9. If the weight matrix
    // is not zero there the loss jumps to ~1e9 and the central-difference dL
    // (~1e-3) falls below f32 resolution (ulp(1e9) ~ 64) - the test becomes meaningless.
    // In real use a softmax always follows the mask.
    let mut w_mat = Matrix::random(4, 4, 1.0, &mut s);
    for i in 0..4 {
        for j in 0..4 {
            if j > i {
                w_mat.set(i, j, 0.0);
            }
        }
    }
    let w = Tensor::leaf(w_mat, false);

    gradient_check("mask_causal", one(&a), || {
        a.mask_causal().mul_elem(&w).sum()
    });
}

#[test]
fn softmax_after_mask() {
    // The real attention path: scores -> causal mask -> softmax.
    let mut s = 26u64;
    let a = leaf(4, 4, &mut s);
    let w = frozen(4, 4, &mut s);
    gradient_check("mask_causal + softmax", one(&a), || {
        a.mask_causal().softmax().mul_elem(&w).sum()
    });
}

// ------------------------------------------------------------
// Normalizasyon / aktivasyon
// ------------------------------------------------------------

#[test]
fn gelu() {
    let mut s = 27u64;
    let a = leaf(3, 5, &mut s);
    gradient_check("gelu", one(&a), || a.gelu().sum());
}

#[test]
fn layer_norm() {
    let mut s = 28u64;
    let x = leaf(3, 4, &mut s);
    let gamma = leaf(1, 4, &mut s);
    let beta = leaf(1, 4, &mut s);
    // A plain sum of the LN output is constant; use a weighted sum.
    let w = frozen(3, 4, &mut s);
    gradient_check(
        "layer_norm (x, gamma, beta)",
        &[x.clone(), gamma.clone(), beta.clone()],
        || x.layer_norm(&gamma, &beta, 1e-5).mul_elem(&w).sum(),
    );
}

// ------------------------------------------------------------
// Embedding
// ------------------------------------------------------------

#[test]
fn embedding() {
    // Through the real `transformer::Embedding` - batched (B=2, S=4) row packing.
    use stonefish_ai::transformer::Embedding;

    let mut s = 29u64;
    let (vocab, d) = (7usize, 3usize);
    let (batch, seq_len) = (2usize, 4usize);
    let emb = Embedding::new(vocab, d, &mut s);
    let ids: Vec<usize> = vec![1, 3, 0, 6, 2, 2, 5, 1];
    assert_eq!(ids.len(), batch * seq_len);

    let w = {
        let mut s2 = 999u64;
        frozen_positive(ids.len(), d, &mut s2)
    };
    // v2: token embedding only (position comes from RoPE)
    assert_eq!(ids.len(), batch * seq_len);
    gradient_check(
        "embedding (tok, B=2 S=4)",
        std::slice::from_ref(&emb.token_emb),
        || emb.forward(&ids).mul_elem(&w).sum(),
    );
}

// ------------------------------------------------------------
// GPU / CPU matmul equivalence
// ------------------------------------------------------------

#[test]
fn gpu_matmul_matches_cpu() {
    use stonefish_ai::gpu;

    let mut s = 30u64;
    let m = 37;
    let k = 53;
    let n = 29;
    let a = Matrix::random(m, k, 1.0, &mut s);
    let b = Matrix::random(k, n, 1.0, &mut s);

    let cpu = a.matmul(&b);
    let gpu = gpu::matmul_gpu(&a, &b);

    assert_eq!((cpu.rows, cpu.cols), (gpu.rows, gpu.cols));
    let mut max_diff = 0.0f32;
    for i in 0..cpu.data.len() {
        max_diff = max_diff.max((cpu.data[i] - gpu.data[i]).abs());
    }
    assert!(
        max_diff < 1e-4,
        "GPU matmul does not match the CPU: max diff {max_diff}"
    );
    println!("  ✅ gpu matmul == cpu matmul (max fark {max_diff:.2e})");
}

// ------------------------------------------------------------
// Fused causal multi-head attention
// ------------------------------------------------------------

#[test]
fn causal_attention_single_head() {
    let mut s = 40u64;
    let (batch, seq, d) = (1usize, 4usize, 3usize);
    let rows = batch * seq;
    let q = leaf(rows, d, &mut s);
    let k = leaf(rows, d, &mut s);
    let v = leaf(rows, d, &mut s);
    let w = frozen_positive(rows, d, &mut s);
    gradient_check(
        "causal_attention (h=1)",
        &[q.clone(), k.clone(), v.clone()],
        || q.causal_attention(&k, &v, batch, seq, 1).mul_elem(&w).sum(),
    );
}

#[test]
fn causal_attention_multi_head_batched() {
    // B=2, S=3, d_model=6, n_heads=3 → d_head=2
    let mut s = 41u64;
    let (batch, seq, d, n_heads) = (2usize, 3usize, 6usize, 3usize);
    let rows = batch * seq;
    let q = leaf(rows, d, &mut s);
    let k = leaf(rows, d, &mut s);
    let v = leaf(rows, d, &mut s);
    let w = frozen_positive(rows, d, &mut s);
    gradient_check(
        "causal_attention (B=2,h=3)",
        &[q.clone(), k.clone(), v.clone()],
        || {
            q.causal_attention(&k, &v, batch, seq, n_heads)
                .mul_elem(&w)
                .sum()
        },
    );
}

#[test]
fn causal_attention_is_causal() {
    // Changing a future token must NOT affect the output of past positions
    // (the causality contract).
    let mut s = 42u64;
    let (batch, seq, d) = (1usize, 4usize, 3usize);
    let q = leaf(batch * seq, d, &mut s);
    let k = leaf(batch * seq, d, &mut s);
    let v = leaf(batch * seq, d, &mut s);

    let before = q
        .causal_attention(&k, &v, batch, seq, 1)
        .data
        .borrow()
        .clone();

    // Son pozisyonun V'sini boz
    {
        let mut vd = v.data.borrow_mut();
        for j in 0..d {
            let cur = vd.get(seq - 1, j);
            vd.set(seq - 1, j, cur + 5.0);
        }
    }
    let after = q
        .causal_attention(&k, &v, batch, seq, 1)
        .data
        .borrow()
        .clone();

    for i in 0..seq - 1 {
        for j in 0..d {
            assert!(
                (before.get(i, j) - after.get(i, j)).abs() < 1e-6,
                "causality violation: position {i} changed when the last token changed"
            );
        }
    }
    println!("  ✅ causal_attention preserves the causality contract");
}

// ------------------------------------------------------------
// Seyrek cross-entropy
// ------------------------------------------------------------

#[test]
fn sparse_ce_matches_dense() {
    let mut s = 43u64;
    let (rows, classes) = (5usize, 4usize);
    let logits = leaf(rows, classes, &mut s);
    let targets: Vec<usize> = vec![0, 3, 1, 2, 0];

    // dense one-hot yol
    let mut t = vec![0.0f32; rows * classes];
    for (i, &c) in targets.iter().enumerate() {
        t[i * classes + c] = 1.0;
    }
    let onehot = Tensor::leaf(Matrix::from_vec(rows, classes, t), false);

    // (a) dense
    logits.zero_grad();
    let l_dense = logits.softmax_cross_entropy(&onehot);
    l_dense.backward();
    let g_dense = logits.grad.borrow().data.clone();
    let loss_dense = l_dense.item();

    // (b) sparse
    logits.zero_grad();
    let l_sparse = logits.softmax_cross_entropy_ids(&targets);
    l_sparse.backward();
    let g_sparse = logits.grad.borrow().data.clone();
    let loss_sparse = l_sparse.item();

    assert!(
        (loss_dense - loss_sparse).abs() < 1e-5,
        "loss diff: dense={loss_dense} sparse={loss_sparse}"
    );
    let mut max_diff = 0.0f32;
    for i in 0..g_dense.len() {
        max_diff = max_diff.max((g_dense[i] - g_sparse[i]).abs());
    }
    assert!(max_diff < 1e-6, "gradient diff: {max_diff}");
    println!("  ✅ sparse CE == dense CE (loss and gradient)");
}

#[test]
fn sparse_ce_gradient() {
    let mut s = 44u64;
    let (rows, classes) = (4usize, 6usize);
    let logits = leaf(rows, classes, &mut s);
    let targets: Vec<usize> = vec![5, 0, 3, 1];
    gradient_check("softmax_cross_entropy_ids", one(&logits), || {
        logits.softmax_cross_entropy_ids(&targets)
    });
}

#[test]
fn gpu_transposed_matmul_variants_match_cpu() {
    // matmul_bt (A*B^T) and matmul_ta (A^T*B) - cuBLAS's transa/transb paths.
    // These two variants remove the transpose materialization from the
    // autograd matmul backward; if set up wrongly they corrupt silently.
    // matmul_bt (A*B^T) and matmul_ta (A^T*B) - cuBLAS's transa/transb paths.
    // These two variants remove the transpose materialization from the
    // autograd matmul backward; if set up wrongly they corrupt silently.
    #[cfg(feature = "cuda")]
    {
        let mut s = 55u64;
        let (m, k, n) = (7usize, 5usize, 3usize);
        let a = Matrix::random(m, k, 1.0, &mut s);
        let b = Matrix::random(n, k, 1.0, &mut s);

        // (a) A·Bᵀ
        let bt = stonefish_ai::gpu::matmul_bt_gpu(&a, &b);
        let bt_ref = a.matmul(&b.transpose());
        assert_eq!((bt.rows, bt.cols), (m, n));
        let mut d = 0.0f32;
        for i in 0..bt.data.len() {
            d = d.max((bt.data[i] - bt_ref.data[i]).abs());
        }
        assert!(d < 1e-4, "matmul_bt wrong: max diff {d}");

        // (b) Aᵀ·B
        let at = a.transpose(); // (k, m)
        let b2 = Matrix::random(k, n, 1.0, &mut s);
        let ta = stonefish_ai::gpu::matmul_ta_gpu(&at, &b2);
        let ta_ref = at.transpose().matmul(&b2); // (m,k)·(k,n)
        assert_eq!((ta.rows, ta.cols), (m, n));
        let mut d2 = 0.0f32;
        for i in 0..ta.data.len() {
            d2 = d2.max((ta.data[i] - ta_ref.data[i]).abs());
        }
        assert!(d2 < 1e-4, "matmul_ta wrong: max diff {d2}");

        println!("  ✅ matmul_bt / matmul_ta match the CPU reference ({d:.1e}, {d2:.1e})");
    }
    #[cfg(not(feature = "cuda"))]
    println!("  (CUDA off - transposed matmul test skipped)");
}

// ------------------------------------------------------------
// End to end: a small GPT - verifies the composition
// ------------------------------------------------------------

#[test]
fn tiny_gpt_end_to_end() {
    use stonefish_ai::transformer::{GptConfig, GPT};

    let mut s = 31u64;
    let config = GptConfig {
        vocab_size: 5,
        max_seq_len: 4,
        d_model: 4,
        n_layers: 1,
        n_heads: 2,
    };
    let model = GPT::new(config, &mut s);

    // EPS CALIBRATION: embedding init scale 0.02, d_model=4 -> row std ~0.01.
    // RMSNorm divides by exactly that std, so eps=1e-3 is ~10% of a row and the
    // linearity assumption of the central difference collapses (the second-order
    // Measured: untouched it gives a spurious 1.4e-1 deviation on the embedding
    // gradients; scaling by x50 drops it to 8e-5, level with every other parameter.
    {
        // Embedding init scale 0.02 -> x50 for RMSNorm eps calibration
        let p = &model.embedding.token_emb;
        let scaled = p.data.borrow().scale(50.0);
        *p.data.borrow_mut() = scaled;
    }

    let params = model.parameters();

    // Batch'li: B=2, S=3
    let (batch, seq_len) = (2usize, 3usize);
    let ids: Vec<usize> = vec![1, 3, 0, 2, 4, 1];
    let targets: Vec<usize> = vec![2, 4, 1, 0, 3, 2];

    let total: usize = params.iter().map(|p| p.data.borrow().data.len()).sum();
    println!("  (small GPT: {total} parameters, B=2 S=3, all checked numerically)");

    assert!(
        total < 2000,
        "the end-to-end gradient test needs a small model"
    );

    gradient_check("GPT end to end (B=2,S=3)", &params, || {
        model
            .forward_batch(&ids, batch, seq_len)
            .softmax_cross_entropy_ids(&targets)
    });
}

/// Verifies that batched training gives *mathematically the same* gradient as
/// single-sequence training: summing the gradients of the split batch must
/// equal the gradient computed in one pack (linearity).
#[test]
fn batch_grad_equals_sum_of_single_sequence_grads() {
    use stonefish_ai::transformer::{GptConfig, GPT};

    let mut s = 77u64;
    let config = GptConfig {
        vocab_size: 6,
        max_seq_len: 4,
        d_model: 4,
        n_layers: 1,
        n_heads: 2,
    };
    let model = GPT::new(config, &mut s);
    {
        // Embedding init scale 0.02 -> x50 for RMSNorm eps calibration
        let p = &model.embedding.token_emb;
        let scaled = p.data.borrow().scale(50.0);
        *p.data.borrow_mut() = scaled;
    }
    let params = model.parameters();

    let (batch, seq_len) = (3usize, 3usize);
    let ids: Vec<usize> = vec![1, 3, 0, 2, 4, 1, 5, 2, 3];
    let targets: Vec<usize> = vec![2, 4, 1, 0, 3, 2, 1, 5, 0];

    // (a) batch gradient in a single pack
    for p in &params {
        p.zero_grad();
    }
    model
        .forward_batch(&ids, batch, seq_len)
        .softmax_cross_entropy_ids(&targets)
        .backward();
    let g_batch: Vec<Vec<f32>> = params
        .iter()
        .map(|p| p.grad.borrow().data.clone())
        .collect();

    // (b) forward each sequence separately and sum the gradients
    for p in &params {
        p.zero_grad();
    }
    for b in 0..batch {
        let lo = b * seq_len;
        let hi = lo + seq_len;
        let l = model
            .forward_batch(&ids[lo..hi], 1, seq_len)
            .softmax_cross_entropy_ids(&targets[lo..hi]);
        // CE divides the mean by the number of rows (B*S). A single-sequence loss
        // divides by S, so L_batch = (1/B)*sum_b L_b holds.
        l.scale(1.0 / batch as f32).backward();
    }
    let g_sum: Vec<Vec<f32>> = params
        .iter()
        .map(|p| p.grad.borrow().data.clone())
        .collect();

    let mut worst = 0.0f32;
    for (a, b) in g_batch.iter().zip(g_sum.iter()) {
        for i in 0..a.len() {
            worst = worst.max((a[i] - b[i]).abs());
        }
    }
    assert!(
        worst < 1e-5,
        "the batch gradient does not equal the sum of the individual gradients (max diff {worst})"
    );
    println!("  ✅ batch gradient == sum of single-sequence gradients (max diff {worst:.2e})");
}

// ============================================================
// v2 architecture: RMSNorm and RoPE
// ============================================================

#[test]
fn rms_norm_forward_and_grad() {
    let mut s = 77u64;
    let (rows, cols) = (4usize, 6usize);
    let x = leaf(rows, cols, &mut s);
    let gamma = Tensor::leaf(Matrix::filled(1, cols, 1.0), true);
    let eps = 1e-6;

    // the mean square of xhat must be 1
    let out = x.rms_norm(&gamma, eps);
    {
        let od = out.data.borrow();
        for r in 0..rows {
            let ms: f32 = (0..cols).map(|j| od.get(r, j) * od.get(r, j)).sum::<f32>() / cols as f32;
            assert!((ms - 1.0).abs() < 1e-3, "rms_norm output rms {ms}");
        }
    }

    // by hand: x / sqrt(mean(x^2)+eps)
    {
        let xd = x.data.borrow();
        let od = out.data.borrow();
        for r in 0..rows {
            let ss: f32 = (0..cols).map(|j| xd.get(r, j) * xd.get(r, j)).sum();
            let inv = 1.0 / (ss / cols as f32 + eps).sqrt();
            for j in 0..cols {
                assert!((od.get(r, j) - xd.get(r, j) * inv).abs() < 1e-6);
            }
        }
    }

    gradient_check("rms_norm (x)", &[x.clone(), gamma.clone()], || {
        x.rms_norm(&gamma, eps)
            .mul_elem(&x.rms_norm(&gamma, eps))
            .scale(0.5)
            .sum()
    });
}

#[test]
fn rms_norm_gamma_grad() {
    let mut s = 78u64;
    let (rows, cols) = (5usize, 8usize);
    let x = frozen(rows, cols, &mut s);
    // keep gamma away from zero (eps calibration)
    let gamma = Tensor::leaf(
        Matrix::random(1, cols, 1.0, &mut s).map(|v| 1.0 + v * 0.5),
        true,
    );
    let eps = 1e-6;
    // weight the output by gamma to make the gamma gradient visible
    let w = frozen(rows, cols, &mut s);
    let f = || x.rms_norm(&gamma, eps).mul_elem(&w).sum();
    // Only gamma is checked: `x` is a frozen constant and `gradient_check`
    // would see its analytic gradient as 0 and compare it numerically.
    gradient_check("rms_norm (gamma)", std::slice::from_ref(&gamma), f);
}

#[test]
fn rope_forward_and_grad() {
    let mut s = 79u64;
    let (batch, seq, n_heads, d_head) = (2usize, 5usize, 2usize, 4usize);
    let cols = n_heads * d_head;
    let x = leaf(batch * seq, cols, &mut s);

    let rot = x.rope(batch, seq, n_heads);
    {
        let rd = rot.data.borrow();
        let xd = x.data.borrow();
        // position 0 must not change
        for j in 0..cols {
            assert!(
                (rd.get(0, j) - xd.get(0, j)).abs() < 1e-6,
                "position 0 changed"
            );
        }
        // the norm must be preserved (rotation is an isometry)
        let n_in: f32 = xd.data.iter().map(|v| v * v).sum();
        let n_out: f32 = rd.data.iter().map(|v| v * v).sum();
        assert!(
            (n_in - n_out).abs() / n_in < 1e-4,
            "the norm was not preserved"
        );
    }

    gradient_check("rope", std::slice::from_ref(&x), || {
        x.rope(batch, seq, n_heads)
            .mul_elem(&x.rope(batch, seq, n_heads))
            .sum()
    });
}
