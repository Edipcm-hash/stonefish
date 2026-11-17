// tests/gradcheck.rs
//
// Merkezi fark (central difference) ile sayısal gradyan doğrulaması.
//
// Her op için: analitik gradyan (backward) ile numerik gradyan
// ((L(θ+ε) − L(θ−ε)) / 2ε) karşılaştırılır.
// Amaç: elle türetilmiş backward formüllerinin tek tek ve *birlikte* doğruluğunu kilitlemek.

// Bu dosyada iç döngüler aynı anda birden çok diziyi indeksliyor (bkz. src/lib.rs).
#![allow(clippy::needless_range_loop)]

use stonefish_ai::autograd::Tensor;
use stonefish_ai::tensor::Matrix;

const EPS: f32 = 1e-3;
const TOL: f32 = 2e-2;

/// Tek parametreli op'lar için: `&[t.clone()]` yerine klonsuz dilim.
fn one(t: &Tensor) -> &[Tensor] {
    std::slice::from_ref(t)
}

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
    gradient_check("matmul", &[a.clone(), b.clone()], || {
        a.matmul(&b).sum()
    });
}

#[test]
fn matmul_square() {
    // Kare matmul: transpose'lu geri yayılım yolları (matmul_ta/bt) burada
    // lda/ldb karışması en zor yakalanır — ayrıca sınanıyor.
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
    gradient_check("transpose", one(&a), || {
        a.transpose().mul_elem(&w).sum()
    });
}

#[test]
fn sum_op() {
    let mut s = 22u64;
    let a = leaf(3, 4, &mut s);
    gradient_check("sum", one(&a), || a.sum().scale(2.0));
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
    gradient_check("softmax_cross_entropy", one(&logits), || {
        logits.softmax_cross_entropy(&target)
    });
}

#[test]
fn softmax_op() {
    let mut s = 24u64;
    let a = leaf(3, 5, &mut s);
    // Düz toplam sabit (1) olduğu için gradyan sıfır çıkar; ağırlıklı toplam kullan.
    let w = frozen(3, 5, &mut s);
    gradient_check("softmax", one(&a), || {
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

    gradient_check("mask_causal", one(&a), || {
        a.mask_causal().mul_elem(&w).sum()
    });
}

#[test]
fn softmax_after_mask() {
    // Attention'ın gerçek yolu: skor → causal mask → softmax.
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
    // Gerçek `transformer::Embedding` üzerinden — batch'li (B=2, S=4) satır paketleme.
    use stonefish_ai::transformer::Embedding;

    let mut s = 29u64;
    let (vocab, max_seq, d) = (7usize, 6usize, 3usize);
    let (batch, seq_len) = (2usize, 4usize);
    let emb = Embedding::new(vocab, max_seq, d, &mut s);
    let ids: Vec<usize> = vec![1, 3, 0, 6, 2, 2, 5, 1];
    assert_eq!(ids.len(), batch * seq_len);

    let w = {
        let mut s2 = 999u64;
        frozen_positive(ids.len(), d, &mut s2)
    };
    gradient_check(
        "embedding (tok+pos, B=2 S=4)",
        &[emb.token_emb.clone(), emb.pos_emb.clone()],
        || emb.forward(&ids, seq_len).mul_elem(&w).sum(),
    );
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
    // Gelecekteki bir token'ı değiştirmek, geçmiş pozisyonların çıktısını
    // ETKİLEMEMELİ. (nedensellik sözleşmesi)
    let mut s = 42u64;
    let (batch, seq, d) = (1usize, 4usize, 3usize);
    let q = leaf(batch * seq, d, &mut s);
    let k = leaf(batch * seq, d, &mut s);
    let v = leaf(batch * seq, d, &mut s);

    let before = q.causal_attention(&k, &v, batch, seq, 1).data.borrow().clone();

    // Son pozisyonun V'sini boz
    {
        let mut vd = v.data.borrow_mut();
        for j in 0..d {
            let cur = vd.get(seq - 1, j);
            vd.set(seq - 1, j, cur + 5.0);
        }
    }
    let after = q.causal_attention(&k, &v, batch, seq, 1).data.borrow().clone();

    for i in 0..seq - 1 {
        for j in 0..d {
            assert!(
                (before.get(i, j) - after.get(i, j)).abs() < 1e-6,
                "nedensellik ihlali: pozisyon {i} son token değişince değişti"
            );
        }
    }
    println!("  ✅ causal_attention nedensellik sözleşmesini koruyor");
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
        "loss farkı: dense={loss_dense} sparse={loss_sparse}"
    );
    let mut max_diff = 0.0f32;
    for i in 0..g_dense.len() {
        max_diff = max_diff.max((g_dense[i] - g_sparse[i]).abs());
    }
    assert!(max_diff < 1e-6, "gradyan farkı: {max_diff}");
    println!("  ✅ sparse CE == dense CE (loss ve gradyan)");
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
    // matmul_bt (A·Bᵀ) ve matmul_ta (Aᵀ·B) — cuBLAS'ın transa/transb yolları.
    // Bu iki varyant autograd'ın matmul geri yayılımında transpose
    // malzemeleştirmesini ortadan kaldırır; yanlış kurulursa sessizce bozar.
    #[cfg(feature = "cuda")]
    {
        use stonefish_ai::gpu::GpuMatrix;

        let mut s = 55u64;
        let (m, k, n) = (7usize, 5usize, 3usize);
        let a = Matrix::random(m, k, 1.0, &mut s);
        let b = Matrix::random(n, k, 1.0, &mut s);

        let ctx = stonefish_ai::gpu::get_ctx();
        let a_gpu = GpuMatrix::from_cpu(ctx, &a);
        let b_gpu = GpuMatrix::from_cpu(ctx, &b);

        // (a) A·Bᵀ
        let bt = GpuMatrix::matmul_bt(&a_gpu, &b_gpu).to_cpu();
        let bt_ref = a.matmul(&b.transpose());
        assert_eq!((bt.rows, bt.cols), (m, n));
        let mut d = 0.0f32;
        for i in 0..bt.data.len() {
            d = d.max((bt.data[i] - bt_ref.data[i]).abs());
        }
        assert!(d < 1e-4, "matmul_bt yanlış: max fark {d}");

        // (b) Aᵀ·B  (A'yı (k,m) gibi kullanmak için transpose edip veriyoruz)
        let at = a.transpose(); // (k, m)
        let b2 = Matrix::random(k, n, 1.0, &mut s);
        let at_gpu = GpuMatrix::from_cpu(ctx, &at);
        let b2_gpu = GpuMatrix::from_cpu(ctx, &b2);
        let ta = GpuMatrix::matmul_ta(&at_gpu, &b2_gpu).to_cpu();
        let ta_ref = at.transpose().matmul(&b2); // (m,k)·(k,n)
        assert_eq!((ta.rows, ta.cols), (m, n));
        let mut d2 = 0.0f32;
        for i in 0..ta.data.len() {
            d2 = d2.max((ta.data[i] - ta_ref.data[i]).abs());
        }
        assert!(d2 < 1e-4, "matmul_ta yanlış: max fark {d2}");

        println!("  ✅ matmul_bt / matmul_ta CPU referansıyla uyuşuyor ({d:.1e}, {d2:.1e})");
    }
    #[cfg(not(feature = "cuda"))]
    println!("  (CUDA kapalı — transposed matmul testi atlandı)");
}

// ------------------------------------------------------------
// Uçtan uca: küçük GPT — kompozisyonu doğrular
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

    // EPS KALİBRASYONU: Embedding init ölçeği 0.02, d_model=4 → satır std'si ~0.01.
    // LayerNorm tam bu std'ye böler, yani eps=1e-3 satırın ~%10'u demektir ve
    // merkezi farkın doğrusallık varsayımı çöker (ikinci türev terimi baskın çıkar).
    // Ölçüldü: dokunulmazsa gömme gradyanlarında 1.4e-1 sahte sapma; ×50 ölçekle
    // 8e-5'e iner ve diğer tüm parametrelerle aynı seviyeye gelir.
    for p in [&model.embedding.token_emb, &model.embedding.pos_emb] {
        let scaled = p.data.borrow().scale(50.0);
        *p.data.borrow_mut() = scaled;
    }

    let params = model.parameters();

    // Batch'li: B=2, S=3
    let (batch, seq_len) = (2usize, 3usize);
    let ids: Vec<usize> = vec![1, 3, 0, 2, 4, 1];
    let targets: Vec<usize> = vec![2, 4, 1, 0, 3, 2];

    let total: usize = params.iter().map(|p| p.data.borrow().data.len()).sum();
    println!("  (küçük GPT: {total} parametre, B=2 S=3, tamamı numerik kontrol ediliyor)");

    assert!(total < 2000, "uçtan uca gradyan testi küçük model ister");

    gradient_check("GPT uçtan uca (B=2,S=3)", &params, || {
        model
            .forward_batch(&ids, batch, seq_len)
            .softmax_cross_entropy_ids(&targets)
    });
}

/// Batch'li eğitimin tek-sequence eğitimle *matematiksel olarak aynı* gradyanı
/// verdiğini doğrular: batch'i parçalayıp gradyanları toplamak, tek pakette
/// hesaplanan gradyana eşit olmalı (linearite).
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
    for p in [&model.embedding.token_emb, &model.embedding.pos_emb] {
        let scaled = p.data.borrow().scale(50.0);
        *p.data.borrow_mut() = scaled;
    }
    let params = model.parameters();

    let (batch, seq_len) = (3usize, 3usize);
    let ids: Vec<usize> = vec![1, 3, 0, 2, 4, 1, 5, 2, 3];
    let targets: Vec<usize> = vec![2, 4, 1, 0, 3, 2, 1, 5, 0];

    // (a) tek pakette batch gradyanı
    for p in &params {
        p.zero_grad();
    }
    model
        .forward_batch(&ids, batch, seq_len)
        .softmax_cross_entropy_ids(&targets)
        .backward();
    let g_batch: Vec<Vec<f32>> = params.iter().map(|p| p.grad.borrow().data.clone()).collect();

    // (b) sequence'leri tek tek forward edip gradyanları topla
    for p in &params {
        p.zero_grad();
    }
    for b in 0..batch {
        let lo = b * seq_len;
        let hi = lo + seq_len;
        let l = model
            .forward_batch(&ids[lo..hi], 1, seq_len)
            .softmax_cross_entropy_ids(&targets[lo..hi]);
        // CE ortalamayı satır sayısına (B*S) böler. Tek sequence loss'u ise S'e
        // böler; dolayısıyla L_batch = (1/B)·Σ_b L_b ilişkisi geçerlidir.
        l.scale(1.0 / batch as f32).backward();
    }
    let g_sum: Vec<Vec<f32>> = params.iter().map(|p| p.grad.borrow().data.clone()).collect();

    let mut worst = 0.0f32;
    for (a, b) in g_batch.iter().zip(g_sum.iter()) {
        for i in 0..a.len() {
            worst = worst.max((a[i] - b[i]).abs());
        }
    }
    assert!(
        worst < 1e-5,
        "batch gradyanı, tek tek gradyanların toplamına eşit değil (max fark {worst})"
    );
    println!("  ✅ batch gradyanı == Σ tek-sequence gradyanı (max fark {worst:.2e})");
}
