// tests/gradcheck.rs
//
// Merkezi fark (central difference) ile sayısal gradyan doğrulaması.
//
// Her op için: analitik gradyan (backward) ile numerik gradyan
// ((L(θ+ε) − L(θ−ε)) / 2ε) karşılaştırılır.
// Amaç: elle türetilmiş backward formüllerinin tek tek ve *birlikte* doğruluğunu kilitlemek.

use stonefish_ai::autograd::Tensor;
use stonefish_ai::tensor::Matrix;

const EPS: f32 = 1e-3;
const TOL: f32 = 2e-2;

fn scalar(t: &Tensor) -> f32 {
    let d = t.data.borrow();
    assert_eq!(
        (d.rows, d.cols),
        (1, 1),
        "gradcheck: loss 1x1 skaler olmalı, {}x{} geldi",
        d.rows,
        d.cols
    );
    d.data[0]
}

/// `build` verilen leaf'lerden skaler bir loss üretmeli.
/// Tüm parametrelerin her elemanı için analitik/numerik gradyan karşılaştırılır.
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
    let analytic: Vec<Vec<f32>> = params.iter().map(|p| p.grad.borrow().data.clone()).collect();

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
                    "param[{pi}] eleman {k}: analitik={a:.6} numerik={numeric:.6} (şekil {}x{})",
                    p.data.borrow().rows,
                    p.data.borrow().cols
                );
            }
        }
    }

    assert!(
        worst_rel <= TOL,
        "{name}: gradyan uyuşmuyor — en kötü rel fark {worst_rel:.3e} > {TOL:e}\n  {worst_msg}"
    );
    println!("  ✅ {name:<28} en kötü rel fark {worst_rel:.2e}");
}

// ------------------------------------------------------------
// Yardımcılar
// ------------------------------------------------------------

fn leaf(rows: usize, cols: usize, seed: &mut u64) -> Tensor {
    Tensor::leaf(Matrix::random(rows, cols, 1.0, seed), true)
}

fn frozen(rows: usize, cols: usize, seed: &mut u64) -> Tensor {
    Tensor::leaf(Matrix::random(rows, cols, 1.0, seed), false)
}

/// ReLU testleri için: |x| ≥ 0.6 (ε = 1e-3 sıfırı geçemez).
fn leaf_away_from_zero(rows: usize, cols: usize, seed: &mut u64) -> Tensor {
    Tensor::leaf(
        Matrix::random(rows, cols, 1.0, seed)
            .map(|x| if x >= 0.0 { x + 0.6 } else { x - 0.6 }),
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
    gradient_check("add (aynı şekil)", &[a.clone(), b.clone()], || a.add(&b).sum());
}

#[test]
fn add_broadcast_bias() {
    let mut s = 12u64;
    let a = leaf(5, 4, &mut s);
    let bias = leaf(1, 4, &mut s); // Linear::forward'ın kullandığı yol
    gradient_check("add (bias broadcast)", &[a.clone(), bias.clone()], || {
        a.add(&bias).sum()
    });
}

#[test]
fn sub_same_shape() {
    let mut s = 13u64;
    let a = leaf(3, 4, &mut s);
    let b = leaf(3, 4, &mut s);
    gradient_check("sub (aynı şekil)", &[a.clone(), b.clone()], || a.sub(&b).sum());
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
    gradient_check("mul_elem", &[a.clone(), b.clone()], || {
        a.mul_elem(&b).sum()
    });
}

#[test]
fn scale_op() {
    let mut s = 16u64;
    let a = leaf(3, 4, &mut s);
    gradient_check("scale", &[a.clone()], || a.scale(2.75).sum());
}

#[test]
fn relu() {
    let mut s = 17u64;
    let a = leaf_away_from_zero(3, 5, &mut s);
    gradient_check("relu", &[a.clone()], || a.relu().sum());
}

#[test]
fn sigmoid() {
    let mut s = 18u64;
    let a = leaf(3, 5, &mut s);
    gradient_check("sigmoid", &[a.clone()], || a.sigmoid().sum());
}

// ------------------------------------------------------------
// Lineer cebir
// ------------------------------------------------------------

#[test]
fn matmul() {
    let mut s = 19u64;
    let a = leaf(4, 3, &mut s);
    let b = leaf(3, 5, &mut s);
    gradient_check("matmul", &[a.clone(), b.clone()], || {
        a.matmul(&b).sum()
    });
}

#[test]
fn matmul_then_transpose() {
    let mut s = 20u64;
    let a = leaf(4, 3, &mut s); // (4,3)
    let b = leaf(6, 3, &mut s); // (6,3) → a @ b^T = (4,6)  [attention skorları]
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
    gradient_check("transpose", &[a.clone()], || {
        a.transpose().mul_elem(&w).sum()
    });
}

#[test]
fn sum_op() {
    let mut s = 22u64;
    let a = leaf(3, 4, &mut s);
    gradient_check("sum", &[a.clone()], || a.sum().scale(2.0));
}

// ------------------------------------------------------------
// Kayıp / olasılık
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
    gradient_check("softmax_cross_entropy", &[logits.clone()], || {
        logits.softmax_cross_entropy(&target)
    });
}

#[test]
fn softmax_op() {
    let mut s = 24u64;
    let a = leaf(3, 5, &mut s);
    // Düz toplam sabit (1) olduğu için gradyan sıfır çıkar; ağırlıklı toplam kullan.
    let w = frozen(3, 5, &mut s);
    gradient_check("softmax", &[a.clone()], || {
        a.softmax().mul_elem(&w).sum()
    });
}

#[test]
fn mask_causal() {
    let mut s = 25u64;
    let a = leaf(4, 4, &mut s);

    // DİKKAT: mask_causal üst üçgeni -1e9 yapar. Ağırlık matrisi orada sıfır
    // olmazsa loss ~1e9 mertebesine çıkar ve merkezi farkın ΔL'si (~1e-3) f32
    // çözünürlüğünün (ulp(1e9) ≈ 64) altında kalır — test anlamsızlaşır.
    // Gerçek kullanımda da maskeden sonra daima softmax gelir.
    let mut w_mat = Matrix::random(4, 4, 1.0, &mut s);
    for i in 0..4 {
        for j in 0..4 {
            if j > i {
                w_mat.set(i, j, 0.0);
            }
        }
    }
    let w = Tensor::leaf(w_mat, false);

    gradient_check("mask_causal", &[a.clone()], || {
        a.mask_causal().mul_elem(&w).sum()
    });
}

#[test]
fn softmax_after_mask() {
    // Attention'ın gerçek yolu: skor → causal mask → softmax.
    let mut s = 26u64;
    let a = leaf(4, 4, &mut s);
    let w = frozen(4, 4, &mut s);
    gradient_check("mask_causal + softmax", &[a.clone()], || {
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
    gradient_check("gelu", &[a.clone()], || a.gelu().sum());
}

#[test]
fn layer_norm() {
    let mut s = 28u64;
    let x = leaf(3, 4, &mut s);
    let gamma = leaf(1, 4, &mut s);
    let beta = leaf(1, 4, &mut s);
    // LN çıktısının düz toplamı sabittir; ağırlıklı toplam kullan.
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
    let mut s = 29u64;
    let vocab = 7;
    let max_seq = 5;
    let d = 3;
    let tok = leaf(vocab, d, &mut s);
    let pos = leaf(max_seq, d, &mut s);
    let ids: Vec<usize> = vec![1, 3, 0, 6, 2];

    // Embedding::forward ile aynı grafiği kur (tok + pos toplamı).
    let build = |tok: &Tensor, pos: &Tensor| -> Tensor {
        let d_model = tok.data.borrow().cols;
        let mut out = Matrix::zeros(ids.len(), d_model);
        {
            let te = tok.data.borrow();
            let pe = pos.data.borrow();
            for (s, &id) in ids.iter().enumerate() {
                for k in 0..d_model {
                    out.set(s, k, te.get(id, k) + pe.get(s, k));
                }
            }
        }
        let ids_mat = Matrix::from_vec(
            ids.len(),
            1,
            ids.iter().map(|&x| x as f32).collect(),
        );
        let ids_t = Tensor::leaf(ids_mat, false);
        Tensor::from_op(
            out,
            stonefish_ai::autograd::Op::Embedding,
            vec![tok.clone(), pos.clone(), ids_t],
        )
    };

    let w = { let mut s2 = 999u64; frozen_positive(ids.len(), d, &mut s2) };
    gradient_check("embedding (tok+pos)", &[tok.clone(), pos.clone()], || {
        build(&tok, &pos).mul_elem(&w).sum()
    });
}

// ------------------------------------------------------------
// GPU / CPU matmul eşdeğerliği
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
        "GPU matmul CPU ile uyuşmuyor: max fark {max_diff}"
    );
    println!("  ✅ gpu matmul == cpu matmul (max fark {max_diff:.2e})");
}

// ------------------------------------------------------------
// Uçtan uca: küçük GPT — kompozisyonu doğrular
// ------------------------------------------------------------

#[test]
fn tiny_gpt_end_to_end() {
    use stonefish_ai::transformer::GPT;

    let mut s = 31u64;
    let vocab = 5;
    let seq_len = 3;
    let d_model = 4;
    let model = GPT::new(vocab, seq_len, d_model, 1, 1, &mut s);
    let params = model.parameters();

    let ids: Vec<usize> = vec![1, 3, 0];
    let mut t = vec![0.0f32; seq_len * vocab];
    for (i, &cls) in [2usize, 4, 1].iter().enumerate() {
        t[i * vocab + cls] = 1.0;
    }
    let target = Tensor::leaf(Matrix::from_vec(seq_len, vocab, t), false);

    let total: usize = params.iter().map(|p| p.data.borrow().data.len()).sum();
    println!("  (küçük GPT: {total} parametre, tamamı numerik kontrol ediliyor)");

    assert!(
        total < 1000,
        "uçtan uca gradyan testi küçük model ister"
    );

    gradient_check("GPT uçtan uca", &params, || {
        model.forward(&ids).softmax_cross_entropy(&target)
    });
}
