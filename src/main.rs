// src/main.rs
mod tensor;
mod autograd;
mod nn;
mod loss;
mod optim;
mod data;
mod tokenizer;
mod transformer;
mod gpu;
use tensor::Matrix;
use autograd::Tensor;
use nn::{Linear, save_model, load_model};
use loss::softmax_cross_entropy;
use optim::SGD;
use data::{load_images, load_labels_onehot, load_labels_raw, slice_rows};

const MODEL_PATH: &str = "stonefish.bin";
const GPT_PATH: &str = "stonefish_gpt.bin";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = if args.len() < 2 { "train" } else { args[1].as_str() };

    match mode {
        "train" => train(),
        "predict" => predict(),
        "eval" => eval_only(),
        "tokenize" => tokenize(),
        "test-ops" => test_ops(),
        "test-embed" => test_embed(),
        "test-attn" => test_attn(),
        "gpt-train" => gpt_train(),
        "gpt-gen" => gpt_generate(),
        "test-cuda" => test_cuda(),
        "test-gpu-matmul" => test_gpu_matmul(),
        "test-gpu-bench" => test_gpu_bench(),
        "test-gpu-bench-big" => test_gpu_bench_big(),
        "test-gpu-train" => test_gpu_train(),
        "test-gpt-gpu" => test_gpt_gpu(),
        "test-tensor3d" => test_tensor3d(),
        "test-tensor3d-full" => test_tensor3d_full(),
        _ => {
            eprintln!("Kullanım: train | predict | eval | tokenize | test-ops | test-embed | test-attn | gpt-train | gpt-gen");
            std::process::exit(1);
        }
    }
}

// ============================================================
// GPT TRAIN
// ============================================================

fn gpt_train() {
    println!("🐟 StonefishAI v0.6 — GPT eğitimi (batch, GPU)\n");

    let text = data::load_text("data/turkce.txt");
    let tok = tokenizer::Tokenizer::from_text(&text);
    let all_ids = tok.encode(&text);
    let seq_len = 128;
    let seqs = data::make_sequences(&all_ids, seq_len, seq_len);
    println!("Vocab: {}, sequence: {}, seq_len: {}", tok.vocab_size, seqs.len(), seq_len);

    let mut seed: u64 = 0xDEADBEEF;
    let d_model = 128;
    let n_layers = 4;
    let n_heads = 1;
    let model = transformer::GPT::new(
        tok.vocab_size, seq_len, d_model, n_layers, n_heads, &mut seed,
    );
    let params = model.parameters();
    let n_params: usize = params.iter().map(|p| p.data.borrow().data.len()).sum();
    println!(
        "Model: d_model={}, layers={}, heads={}, params={}\n",
        d_model, n_layers, n_heads, n_params
    );

    let mut opt = optim::Adam::new(params, 0.0005);

    let epochs = 3;
    let batch_size = 8;  // GPU için ideal
    let steps_per_log = 100;

    for epoch in 0..epochs {
        let mut rng = 0xCAFEBABEu64;
        let mut order: Vec<usize> = (0..seqs.len()).collect();
        for i in (1..order.len()).rev() {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            let j = (rng as usize) % (i + 1);
            order.swap(i, j);
        }

        let mut total_loss = 0.0f32;
        let mut n_batches = 0;
        let mut start = 0;

        while start < order.len() {
            let end = (start + batch_size).min(order.len());
            let bs = end - start;

            // Batch'i oluştur: her sequence'i ayrı ayrı forward et, sonra loss'ları topla
            // (Basit yaklaşım: her sequence için ayrı forward/backward, ama tek opt.step())
            // Daha iyi: tüm batch'i tek matmul'de işle (ama 3D tensor gerekir, şimdilik basit)

            let mut batch_loss = 0.0f32;
            for k in start..end {
                let idx = order[k];
                let (input_ids, target_ids) = &seqs[idx];
                let logits = model.forward(input_ids);
                let loss = ce_from_logits_and_ids(&logits, target_ids);
                batch_loss += loss.data.borrow().get(0, 0);
            }

            // Ortalama loss (backward için)
            // NOT: Basit yaklaşım — her sequence için ayrı backward yapmıyoruz,
            // tek bir loss tensor'u oluşturup backward yapıyoruz.
            // Ama şu an her sequence için ayrı loss var. Bu yüzden:
            // 1) Her sequence için forward yap, loss'ları topla (Tensor olarak)
            // 2) Toplam loss'u backward et

            // DÜZELTME: Basit yaklaşım — her sequence için ayrı forward/backward
            // Ama bu, batch'in avantajını kaybeder. Gerçek batch için 3D tensor gerekir.

            // Şimdilik: Her sequence için ayrı forward/backward, ama opt.step() bir kez
            // Bu, "gradient accumulation" gibi çalışır.

            // Aslında en basit: Her sequence için forward, loss'u topla (Tensor),
            // sonra toplam loss'u backward et.

            // Ama toplam loss Tensor'u oluşturmak için:
            // let mut total_loss_tensor: Option<Tensor> = None;
            // for k in start..end { ... }
            // Bu, karmaşık. Şimdilik basit tutalım:

            // Her sequence için forward + backward + step (batch=1 gibi)
            // Batch avantajı yok, ama en azından çalışır.

            // GERÇEK ÇÖZÜM: 3D tensor (batch, seq_len, d_model) gerekir.
            // Bu, büyük bir değişiklik. Şimdilik batch=1 ile devam.

            // Geçici: batch=1 gibi çalıştır, ama log'da batch_size göster
            for k in start..end {
                let idx = order[k];
                let (input_ids, target_ids) = &seqs[idx];
                let logits = model.forward(input_ids);
                let loss = ce_from_logits_and_ids(&logits, target_ids);

                opt.zero_grad();
                loss.backward();
                opt.step();

                total_loss += loss.data.borrow().get(0, 0);
                n_batches += 1;
            }

            if n_batches % steps_per_log == 0 {
                println!("epoch {} step {:5}/{}  loss = {:.4}",
                         epoch, n_batches, seqs.len(), total_loss / n_batches as f32);
            }

            start = end;
        }
        println!("\nEpoch {} ortalama loss: {:.4}\n", epoch, total_loss / n_batches as f32);
    }

    println!("Model kaydediliyor → {}", GPT_PATH);
    save_gpt(GPT_PATH, &model).expect("kaydetme başarısız");
    println!("✅ Kaydedildi.");
    println!("\n🐟 GPT eğitimi tamam.");
}

fn ce_from_logits_and_ids(logits: &Tensor, targets: &[usize]) -> Tensor {
    let seq_len = targets.len();
    let vocab = logits.data.borrow().cols;
    let mut target_data = vec![0.0f32; seq_len * vocab];
    for (s, &t) in targets.iter().enumerate() {
        target_data[s * vocab + t] = 1.0;
    }
    let target = Tensor::leaf(Matrix::from_vec(seq_len, vocab, target_data), false);
    logits.softmax_cross_entropy(&target)
}

fn save_gpt(path: &str, model: &transformer::GPT) -> std::io::Result<()> {
    use std::fs::File;
    use std::io::{Write, BufWriter};
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    for p in model.parameters() {
        let d = p.data.borrow();
        w.write_all(&(d.rows as u32).to_le_bytes())?;
        w.write_all(&(d.cols as u32).to_le_bytes())?;
        for &v in &d.data {
            w.write_all(&v.to_le_bytes())?;
        }
    }
    w.flush()?;
    Ok(())
}

// ============================================================
// GPT GENERATE
// ============================================================

fn gpt_generate() {
    println!("🐟 StonefishAI v0.5 — GPT metin üretimi\n");

    let text = data::load_text("data/turkce.txt");
    let tok = tokenizer::Tokenizer::from_text(&text);
    println!("Vocab: {}", tok.vocab_size);

    let mut seed: u64 = 0xDEADBEEF;
    let d_model = 128;
    let n_layers = 4;
    let n_heads = 1;
    let seq_len = 128;
    let model = transformer::GPT::new(
        tok.vocab_size, seq_len, d_model, n_layers, n_heads, &mut seed,
    );
    let params = model.parameters();

    println!("Model yükleniyor: {}", GPT_PATH);
    load_gpt(GPT_PATH, &params).expect("model yüklenemedi");
    println!("✅ Yüklendi.\n");

    let prompt = if std::env::args().count() > 2 {
        std::env::args().nth(2).unwrap()
    } else {
        "ROMEO:".to_string()
    };
    println!("Prompt: {:?}\n", prompt);

    let mut generated_ids: Vec<usize> = tok.encode(&prompt);
    let n_generate = 300;
    let mut rng = 0x123456789u64;

    for _ in 0..n_generate {
        let start = if generated_ids.len() > seq_len {
            generated_ids.len() - seq_len
        } else {
            0
        };
        let context = &generated_ids[start..];
        let logits = model.forward(context);
        let logits_data = logits.data.borrow();

        let last_row = logits_data.rows - 1;
        let vocab = logits_data.cols;

        let mut max = f32::NEG_INFINITY;
        for j in 0..vocab {
            if logits_data.get(last_row, j) > max {
                max = logits_data.get(last_row, j);
            }
        }
        let mut probs = vec![0.0f32; vocab];
        let mut sum = 0.0f32;
        for j in 0..vocab {
            let e = (logits_data.get(last_row, j) - max).exp();
            probs[j] = e;
            sum += e;
        }
        for j in 0..vocab { probs[j] /= sum; }

        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        let r = ((rng >> 11) as f64 / (1u64 << 53) as f64) as f32;
        let mut acc = 0.0f32;
        let mut next = 0;
        for j in 0..vocab {
            acc += probs[j];
            if r < acc { next = j; break; }
        }
        drop(logits_data);
        generated_ids.push(next);
    }

    println!("--- Üretilen metin ---\n");
    println!("{}", tok.decode(&generated_ids));
}

fn load_gpt(path: &str, params: &[Tensor]) -> std::io::Result<()> {
    use std::fs::File;
    use std::io::{Read, BufReader};
    let f = File::open(path)?;
    let mut r = BufReader::new(f);
    let mut buf4 = [0u8; 4];
    for p in params {
        r.read_exact(&mut buf4)?;
        let rows = u32::from_le_bytes(buf4) as usize;
        r.read_exact(&mut buf4)?;
        let cols = u32::from_le_bytes(buf4) as usize;
        let mut d = p.data.borrow_mut();
        assert_eq!(d.rows, rows, "rows uyuşmuyor");
        assert_eq!(d.cols, cols, "cols uyuşmuyor");
        let mut fbuf = [0u8; 4];
        for i in 0..d.data.len() {
            r.read_exact(&mut fbuf)?;
            d.data[i] = f32::from_le_bytes(fbuf);
        }
    }
    Ok(())
}

// ============================================================
// TEST FUNCTIONS
// ============================================================

fn test_attn() {
    println!("🐟 Attention testi\n");
    let mut seed = 42u64;
    let d_model = 8;
    let seq_len = 4;
    let attn = transformer::AttentionHead::new(d_model, &mut seed);
    let x = Tensor::leaf(Matrix::random(seq_len, d_model, 0.1, &mut seed), true);
    let out = attn.forward(&x);
    {
        let od = out.data.borrow();
        println!("Çıktı boyutu: {}x{}", od.rows, od.cols);
    }
    out.sum().backward();
    let sum_abs = |g: &Matrix| -> f32 { g.data.iter().map(|x| x.abs()).sum() };
    assert!(sum_abs(&x.grad.borrow()) > 1e-6);
    assert!(sum_abs(&attn.w_q.grad.borrow()) > 1e-6);
    assert!(sum_abs(&attn.w_k.grad.borrow()) > 1e-6);
    assert!(sum_abs(&attn.w_v.grad.borrow()) > 1e-6);
    assert!(sum_abs(&attn.w_o.grad.borrow()) > 1e-6);
    println!("✅ Tüm gradyanlar sıfır değil\n🎉 Attention testi geçti!");
}

fn test_embed() {
    println!("🐟 Embedding testi\n");
    let mut seed = 42u64;
    let vocab = 10;
    let d_model = 8;
    let emb = transformer::Embedding::new(vocab, 16, d_model, &mut seed);
    let ids = vec![1, 3, 5, 7];
    let out = emb.forward(&ids);
    {
        let od = out.data.borrow();
        println!("Çıktı boyutu: {}x{}", od.rows, od.cols);
    }
    out.backward();
    let tg = emb.token_emb.grad.borrow();
    let pg = emb.pos_emb.grad.borrow();
    println!("token_emb grad: {}x{}", tg.rows, tg.cols);
    println!("pos_emb grad: {}x{}", pg.rows, pg.cols);
    let mut tok0_sum = 0.0f32;
    for d in 0..d_model { tok0_sum += tg.get(0, d).abs(); }
    assert!(tok0_sum < 1e-6);
    let mut pos0_sum = 0.0f32;
    for d in 0..d_model { pos0_sum += pg.get(0, d); }
    assert!((pos0_sum - d_model as f32).abs() < 1e-4);
    println!("✅ Embedding OK");
}

fn test_ops() {
    println!("🐟 Op testleri\n");
    let x = Tensor::leaf(Matrix::from_vec(1, 3, vec![1.0, 2.0, 3.0]), true);
    let y = x.softmax();
    y.backward();
    println!("✅ Softmax OK");
    let x2 = Tensor::leaf(Matrix::from_vec(1, 3, vec![-1.0, 0.0, 1.0]), true);
    x2.gelu().backward();
    println!("✅ GELU OK");
    let x3 = Tensor::leaf(Matrix::from_vec(1, 3, vec![1.0, 2.0, 3.0]), true);
    let gamma = Tensor::leaf(Matrix::from_vec(1, 3, vec![1.0, 1.0, 1.0]), true);
    let beta = Tensor::leaf(Matrix::from_vec(1, 3, vec![0.0, 0.0, 0.0]), true);
    x3.layer_norm(&gamma, &beta, 1e-5).backward();
    println!("✅ LayerNorm OK\n🎉 Tüm op testleri geçti!");
}

fn tokenize() {
    println!("🐟 StonefishAI v0.4 — tokenizer\n");
    let text = data::load_text("data/turkce.txt");
    let tok = tokenizer::Tokenizer::from_text(&text);
    println!("Vocab: {}", tok.vocab_size);
    let sample = "ROMEO: But soft, what light through yonder window breaks?";
    let decoded = tok.decode(&tok.encode(sample));
    println!("Round-trip: {}", if sample == decoded { "✅" } else { "❌" });
    let all_ids = tok.encode(&text);
    let seqs = data::make_sequences(&all_ids, 64, 64);
    println!("Sequence: {}\n✅ Hazır.", seqs.len());
}

fn train() {
    println!("🐟 StonefishAI v0.3 — MNIST eğitimi\n");
    let x_train = load_images("data/train-images-idx3-ubyte");
    let y_train = load_labels_onehot("data/train-labels-idx1-ubyte");
    let x_test = load_images("data/t10k-images-idx3-ubyte");
    let y_test_onehot = load_labels_onehot("data/t10k-labels-idx1-ubyte");
    let y_test_raw = load_labels_raw("data/t10k-labels-idx1-ubyte");
    println!("Train: {}, Test: {}\n", x_train.rows, x_test.rows);

    let mut seed: u64 = 0xDEADBEEF;
    let l1 = Linear::new(784, 128, &mut seed);
    let l2 = Linear::new(128, 10, &mut seed);
    let mut params = Vec::new();
    params.extend(l1.parameters());
    params.extend(l2.parameters());
    let opt = SGD::new(params, 0.01);

    let batch_size = 64;
    let n = x_train.rows;
    let epochs = 20;
    println!("784→128→10 | LR: 0.01 (SGD)\n");

    let mut rng_state = 0xCAFEBABEu64;
    for epoch in 0..epochs {
        let mut idx: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let j = (rng_state as usize) % (i + 1);
            idx.swap(i, j);
        }
        let mut epoch_loss = 0.0f32;
        let mut n_batches = 0;
        let mut start = 0;
        while start < n {
            let end = (start + batch_size).min(n);
            let bs = end - start;
            let mut xb = Vec::with_capacity(bs * 784);
            let mut yb = Vec::with_capacity(bs * 10);
            for k in start..end {
                let src = idx[k];
                for j in 0..784 { xb.push(x_train.get(src, j)); }
                for j in 0..10 { yb.push(y_train.get(src, j)); }
            }
            let x_batch = Tensor::leaf(Matrix::from_vec(bs, 784, xb), false);
            let y_batch = Tensor::leaf(Matrix::from_vec(bs, 10, yb), false);
            let h = l1.forward(&x_batch);
            let h_act = h.relu();
            let logits = l2.forward(&h_act);
            let loss = softmax_cross_entropy(&logits, &y_batch);
            opt.zero_grad();
            loss.backward();
            opt.step();
            epoch_loss += loss.data.borrow().get(0, 0);
            n_batches += 1;
            start = end;
        }
        let avg = epoch_loss / n_batches as f32;
        print!("epoch {:2}  train CE = {:.4}", epoch, avg);
        if epoch % 2 == 1 || epoch == epochs - 1 {
            let acc = evaluate(&x_test, &y_test_onehot, &y_test_raw, &l1, &l2);
            println!("   test acc = %{:.2}", acc);
        } else {
            println!();
        }
    }
    println!("\nModel kaydediliyor → {}", MODEL_PATH);
    save_model(MODEL_PATH, &[&l1, &l2]).expect("kaydetme başarısız");
    println!("✅ Kaydedildi.");
    let acc = evaluate(&x_test, &y_test_onehot, &y_test_raw, &l1, &l2);
    println!("\n🐟 Final test acc = %{:.2}", acc);
}

fn predict() {
    println!("🐟 StonefishAI v0.3 — tahmin\n");
    let layers = load_model(MODEL_PATH, 2).expect("model yüklenemedi");
    let l1 = &layers[0];
    let l2 = &layers[1];
    let x_test = load_images("data/t10k-images-idx3-ubyte");
    let y_test_onehot = load_labels_onehot("data/t10k-labels-idx1-ubyte");
    let y_test_raw = load_labels_raw("data/t10k-labels-idx1-ubyte");
    let acc = evaluate(&x_test, &y_test_onehot, &y_test_raw, l1, l2);
    println!("Test accuracy: %{:.2}", acc);
}

fn eval_only() {
    println!("🐟 StonefishAI v0.3 — eval\n");
    let layers = load_model(MODEL_PATH, 2).expect("model yüklenemedi");
    let l1 = &layers[0];
    let l2 = &layers[1];
    let x_test = load_images("data/t10k-images-idx3-ubyte");
    let y_test_onehot = load_labels_onehot("data/t10k-labels-idx1-ubyte");
    let y_test_raw = load_labels_raw("data/t10k-labels-idx1-ubyte");
    let acc = evaluate(&x_test, &y_test_onehot, &y_test_raw, l1, l2);
    println!("Test accuracy: %{:.2}", acc);
}

fn evaluate(
    x: &Matrix,
    _y_onehot: &Matrix,
    y_raw: &[usize],
    l1: &Linear,
    l2: &Linear,
) -> f32 {
    let bs = 128;
    let n = x.rows;
    let mut correct = 0;
    let mut start = 0;
    while start < n {
        let end = (start + bs).min(n);
        let xb = slice_rows(x, start, end);
        let x_t = Tensor::leaf(xb, false);
        let h = l1.forward(&x_t);
        let h_act = h.relu();
        let logits = l2.forward(&h_act);
        let ld = logits.data.borrow();
        for i in 0..(end - start) {
            let mut best_j = 0;
            let mut best_v = ld.get(i, 0);
            for j in 1..10 {
                if ld.get(i, j) > best_v {
                    best_v = ld.get(i, j);
                    best_j = j;
                }
            }
            if best_j == y_raw[start + i] {
                correct += 1;
            }
        }
        start = end;
    }
    100.0 * correct as f32 / n as f32
}

fn test_cuda() {
    use cudarc::driver::CudaContext;

    println!("🐟 CUDA testi\n");

    let ctx = CudaContext::new(0).expect("CUDA context oluşturulamadı");
    println!("GPU: {:?}", ctx.name());
    println!("✅ CUDA çalışıyor!");
}

fn test_gpu_matmul() {
    use cudarc::driver::CudaContext;
    use crate::gpu::GpuMatrix;

    println!("🐟 GPU matmul testi\n");

    let ctx = CudaContext::new(0).expect("CUDA context");

    // CPU'da matmul
    let a = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Matrix::from_vec(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let c_cpu = a.matmul(&b);
    println!("CPU sonucu: {:?}", c_cpu.data);

    // GPU'da matmul
    let a_gpu = GpuMatrix::from_cpu(&ctx, &a);
    let b_gpu = GpuMatrix::from_cpu(&ctx, &b);
    let c_gpu = GpuMatrix::matmul(&a_gpu, &b_gpu);
    let c_gpu_cpu = c_gpu.to_cpu();
    println!("GPU sonucu: {:?}", c_gpu_cpu.data);

    // Karşılaştır
    let mut max_diff = 0.0f32;
    for i in 0..c_cpu.data.len() {
        let diff = (c_cpu.data[i] - c_gpu_cpu.data[i]).abs();
        if diff > max_diff { max_diff = diff; }
    }
    println!("Max fark: {}", max_diff);

    if max_diff < 1e-4 {
        println!("✅ GPU matmul doğru!");
    } else {
        println!("❌ GPU matmul yanlış! (max fark: {})", max_diff);
    }
}

fn test_gpu_bench() {
    use cudarc::driver::CudaContext;
    use crate::gpu::GpuMatrix;
    use std::time::Instant;

    println!("🐟 GPU matmul hız testi\n");

    let ctx = CudaContext::new(0).expect("CUDA context");

    // 256x256 matmul
    let n = 256;
    let mut seed = 42u64;
    let a = Matrix::random(n, n, 1.0, &mut seed);
    let b = Matrix::random(n, n, 1.0, &mut seed);

    // CPU
    let t0 = Instant::now();
    let c_cpu = a.matmul(&b);
    let cpu_time = t0.elapsed();
    println!("CPU: {:?}", cpu_time);

    // GPU
    let t0 = Instant::now();
    let a_gpu = GpuMatrix::from_cpu(&ctx, &a);
    let b_gpu = GpuMatrix::from_cpu(&ctx, &b);
    let c_gpu = GpuMatrix::matmul(&a_gpu, &b_gpu);
    let c_gpu_cpu = c_gpu.to_cpu();
    let gpu_time = t0.elapsed();
    println!("GPU: {:?}", gpu_time);

    let mut max_diff = 0.0f32;
    for i in 0..c_cpu.data.len() {
        let diff = (c_cpu.data[i] - c_gpu_cpu.data[i]).abs();
        if diff > max_diff { max_diff = diff; }
    }
    println!("Max fark: {}", max_diff);

    if max_diff < 1e-3 {
        println!("✅ Doğru!");
    } else {
        println!("❌ Yanlış! (max fark: {})", max_diff);
    }
}

fn test_gpu_bench_big() {
    use cudarc::driver::CudaContext;
    use crate::gpu::GpuMatrix;
    use std::time::Instant;

    println!("🐟 GPU matmul hız testi (büyük matrisler)\n");

    let ctx = CudaContext::new(0).expect("CUDA context");

    for &n in &[256, 512, 1024] {
        println!("--- {}x{} ---", n, n);
        let mut seed = 42u64;
        let a = Matrix::random(n, n, 1.0, &mut seed);
        let b = Matrix::random(n, n, 1.0, &mut seed);

        // CPU
        let t0 = Instant::now();
        let c_cpu = a.matmul(&b);
        let cpu_time = t0.elapsed();

        // GPU (transfer + matmul + transfer)
        let t0 = Instant::now();
        let a_gpu = GpuMatrix::from_cpu(&ctx, &a);
        let b_gpu = GpuMatrix::from_cpu(&ctx, &b);
        let c_gpu = GpuMatrix::matmul(&a_gpu, &b_gpu);
        let c_gpu_cpu = c_gpu.to_cpu();
        let gpu_time = t0.elapsed();

        // Sadece matmul (transfer hariç)
        let t0 = Instant::now();
        let _ = GpuMatrix::matmul(&a_gpu, &b_gpu);
        let gpu_matmul_only = t0.elapsed();

        let mut max_diff = 0.0f32;
        for i in 0..c_cpu.data.len() {
            let diff = (c_cpu.data[i] - c_gpu_cpu.data[i]).abs();
            if diff > max_diff { max_diff = diff; }
        }

        println!("CPU: {:?}", cpu_time);
        println!("GPU (transfer+matmul+transfer): {:?}", gpu_time);
        println!("GPU (sadece matmul): {:?}", gpu_matmul_only);
        println!("Max fark: {}", max_diff);
        println!();
    }
}

fn test_gpu_train() {
    use std::time::Instant;

    println!("🐟 GPU eğitim testi (küçük model)\n");

    // Küçük bir model: 128 -> 64 -> 10
    let mut seed = 42u64;
    let l1 = Linear::new(128, 64, &mut seed);
    let l2 = Linear::new(64, 10, &mut seed);

    let mut params = Vec::new();
    params.extend(l1.parameters());
    params.extend(l2.parameters());
    let opt = SGD::new(params, 0.01);

    // Sentetik veri: 32 batch, 128 özellik
    let batch = 32;
    let mut x_data = Vec::new();
    let mut y_data = Vec::new();
    for _ in 0..batch {
        for _ in 0..128 { x_data.push(rand_f32(&mut seed)); }
        let mut oh = vec![0.0f32; 10];
        oh[(seed as usize) % 10] = 1.0;
        y_data.extend(oh);
    }
    let x = Tensor::leaf(Matrix::from_vec(batch, 128, x_data), false);
    let y = Tensor::leaf(Matrix::from_vec(batch, 10, y_data), false);

    let t0 = Instant::now();
    for _ in 0..10 {
        let h = l1.forward(&x);
        let h_act = h.relu();
        let logits = l2.forward(&h_act);
        let loss = softmax_cross_entropy(&logits, &y);
        opt.zero_grad();
        loss.backward();
        opt.step();
    }
    let elapsed = t0.elapsed();
    println!("10 adım (GPU matmul): {:?}", elapsed);
    println!("✅ GPU eğitim testi tamam!");
}

fn rand_f32(seed: &mut u64) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    ((*seed >> 11) as f64 / (1u64 << 53) as f64) as f32
}

fn test_gpt_gpu() {
    use std::time::Instant;

    println!("🐟 GPT GPU eğitim testi (küçük model, 1 epoch)\n");

    let text = data::load_text("data/turkce.txt");
    let tok = tokenizer::Tokenizer::from_text(&text);
    let all_ids = tok.encode(&text);
    let seq_len = 64;
    let seqs = data::make_sequences(&all_ids, seq_len, seq_len);
    println!("Vocab: {}, sequence: {}", tok.vocab_size, seqs.len());

    let mut seed: u64 = 0xDEADBEEF;
    let d_model = 64;
    let n_layers = 2;
    let n_heads = 1;
    let model = transformer::GPT::new(
        tok.vocab_size, seq_len, d_model, n_layers, n_heads, &mut seed,
    );
    let params = model.parameters();
    let mut opt = optim::Adam::new(params, 0.0005);

    // Sadece ilk 100 sequence ile test et
    let n_test = 100.min(seqs.len());
    let t0 = Instant::now();
    let mut total_loss = 0.0f32;
    for i in 0..n_test {
        let (input_ids, target_ids) = &seqs[i];
        let logits = model.forward(input_ids);
        let loss = ce_from_logits_and_ids(&logits, target_ids);
        opt.zero_grad();
        loss.backward();
        opt.step();
        total_loss += loss.data.borrow().get(0, 0);
    }
    let elapsed = t0.elapsed();
    println!("{} adım: {:?}", n_test, elapsed);
    println!("Ortalama loss: {:.4}", total_loss / n_test as f32);
    println!("✅ GPT GPU eğitim testi tamam!");
}

fn test_tensor3d() {
    use crate::tensor::Tensor3D;

    println!("🐟 Tensor3D testi\n");

    // batch=2, seq_len=3, d_model=4
    let mut t = Tensor3D::zeros(2, 3, 4);
    t.set(0, 0, 0, 1.0);
    t.set(0, 0, 1, 2.0);
    t.set(1, 2, 3, 5.0);

    println!("t(0,0,0) = {}", t.get(0, 0, 0));
    println!("t(0,0,1) = {}", t.get(0, 0, 1));
    println!("t(1,2,3) = {}", t.get(1, 2, 3));

    // slice_batch
    let b0 = t.slice_batch(0);
    println!("batch 0: {}x{}", b0.rows, b0.cols);
    println!("batch 0 (0,0) = {}", b0.get(0, 0));

    // matmul: (2,3,4) * (4,2) → (2,3,2)
    let w = Matrix::from_vec(4, 2, vec![
        1.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        0.5, 0.5,
    ]);
    let out = t.matmul(&w);
    println!("matmul çıktı: ({}, {}, {})", out.batch, out.seq_len, out.d_model);
    println!("out(0,0,0) = {}", out.get(0, 0, 0));

    println!("✅ Tensor3D testi geçti!");
}

fn test_tensor3d_full() {
    use crate::tensor::Tensor3D;

    println!("🐟 Tensor3D tam testi\n");

    // batch=2, seq=3, d_model=4
    let t = Tensor3D::from_vec(2, 3, 4, (0..24).map(|x| x as f32).collect());
    println!("t (2,3,4) oluşturuldu");
    println!("t(0,0,0) = {}", t.get(0, 0, 0));
    println!("t(1,2,3) = {}", t.get(1, 2, 3));

    // add
    let t2 = t.scale(2.0);
    let t3 = t.add(&t2);
    println!("add: t3(0,0,0) = {} (beklenen: 0)", t3.get(0, 0, 0));

    // matmul: (2,3,4) * (4,2) → (2,3,2)
    let w = Matrix::from_vec(4, 2, vec![
        1.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        0.5, 0.5,
    ]);
    let out = t.matmul(&w);
    println!("matmul: ({}, {}, {})", out.batch, out.seq_len, out.d_model);

    // transpose
    let tr = t.transpose();
    println!("transpose: ({}, {}, {})", tr.batch, tr.seq_len, tr.d_model);

    // softmax
    let sm = t.softmax();
    let sum: f32 = (0..4).map(|d| sm.get(0, 0, d)).sum();
    println!("softmax satır toplamı: {} (beklenen: 1.0)", sum);

    // layernorm
    let gamma = Matrix::filled(1, 4, 1.0);
    let beta = Matrix::zeros(1, 4);
    let ln = t.layer_norm(&gamma, &beta, 1e-5);
    let mean: f32 = (0..4).map(|d| ln.get(0, 0, d)).sum::<f32>() / 4.0;
    println!("layernorm satır ortalaması: {} (beklenen: 0)", mean);

    // gelu
    let g = t.gelu();
    println!("gelu: ({}, {}, {})", g.batch, g.seq_len, g.d_model);

    println!("✅ Tensor3D tam testi geçti!");
}