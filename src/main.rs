// src/main.rs
// CLI kabuğu. Motorun tamamı `stonefish_ai` kütüphanesinde (src/lib.rs).
use stonefish_ai::autograd::Tensor;
use stonefish_ai::checkpoint;
use stonefish_ai::data::{load_images, load_labels_onehot, load_labels_raw, slice_rows};
use stonefish_ai::loss::softmax_cross_entropy;
use stonefish_ai::nn::{load_model, save_model, Linear};
use stonefish_ai::optim::{Adam, SGD};
use stonefish_ai::tensor::Matrix;
use stonefish_ai::tokenizer::Tokenizer;
use stonefish_ai::transformer::{GptConfig, GPT};
use stonefish_ai::{data, gpu, transformer};

const MODEL_PATH: &str = "stonefish.bin";
const GPT_PATH: &str = "stonefish_gpt.bin";
/// Checkpoint'in yanına yazılan vocab dosyası (korpus drift tespiti için).
const GPT_VOCAB_PATH: &str = "stonefish_gpt.vocab";
const TEXT_PATH: &str = "data/turkce.txt";

/// Tüm CLI modları: (isim, kısa açıklama). `help` bunu basar, `main` bunu eşler.
const MODES: &[(&str, &str)] = &[
    ("train", "MNIST MLP eğit (784→128→10, SGD)"),
    ("eval", "kayıtlı MNIST modelini test setinde ölç"),
    ("predict", "kayıtlı MNIST modeliyle tahmin (eval ile aynı)"),
    ("tokenize", "tokenizer'ı doğrula + sequence sayısını bas"),
    ("gpt-train", "mini GPT eğit [epoch] [batch]"),
    ("gpt-gen", "kayıtlı GPT ile metin üret  [prompt] [token] [sıcaklık]"),
    ("test-ops", "op duman testi"),
    ("test-embed", "embedding duman testi"),
    ("test-attn", "attention duman testi"),
    ("test-cuda", "CUDA context açılabiliyor mu"),
    ("test-gpu-matmul", "GPU matmul == CPU matmul"),
    ("test-gpu-bench", "256² matmul: CPU vs GPU"),
    ("test-gpu-bench-big", "256/512/1024² matmul: CPU vs GPU"),
    ("test-matmul-bench", "gerçek model şeklinde CPU vs GPU matmul"),
    ("test-gpu-train", "küçük MLP'te 10 adım süresi"),
    ("test-gpt-gpu", "küçük GPT'te adım süresi"),
];

fn arg_usize(args: &[String], i: usize, default: usize) -> usize {
    args.get(i).and_then(|s| s.parse().ok()).unwrap_or(default)
}


fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rest: &[String] = if args.len() > 2 { &args[2..] } else { &[] };
    let mode = if args.len() < 2 { "train" } else { args[1].as_str() };

    match mode {
        "help" | "-h" | "--help" => {
            println!("🐟 StonefishAI v{}", stonefish_ai::VERSION);
            println!(
                "GPU yolu: {}",
                if gpu::HAS_CUDA {
                    "açık (CUDA)"
                } else {
                    "kapalı (CPU)"
                }
            );
            println!("\nKullanım: stonefish_ai <mod> [arg…]\n\nModlar:");
            for (name, desc) in MODES {
                println!("  {:<20} {}", name, desc);
            }
        }
        "train" => train(),
        "predict" => predict(),
        "eval" => eval_only(),
        "tokenize" => tokenize(),
        "test-ops" => test_ops(),
        "test-embed" => test_embed(),
        "test-attn" => test_attn(),
        "gpt-train" => gpt_train(rest),
        "gpt-gen" => gpt_generate(rest),
        "test-cuda" => test_cuda(),
        "test-gpu-matmul" => test_gpu_matmul(),
        "test-gpu-bench" => test_gpu_bench(),
        "test-gpu-bench-big" => test_gpu_bench_big(),
        "test-matmul-bench" => test_matmul_bench(),
        "test-gpu-train" => test_gpu_train(),
        "test-gpt-gpu" => test_gpt_gpu(rest),
        other => {
            eprintln!("Bilinmeyen mod: {}", other);
            eprintln!(
                "Geçerli modlar: {}",
                MODES.iter().map(|(n, _)| *n).collect::<Vec<_>>().join(" | ")
            );
            eprintln!("Ayrıntı için: stonefish_ai help");
            std::process::exit(1);
        }
    }
}

// ============================================================
// GPT EĞİTİMİ — gerçek batch (tek forward/backward/step)
// ============================================================

fn gpt_train(args: &[String]) {
    let epochs = arg_usize(args, 0, 3);
    let batch_size = arg_usize(args, 1, 8);
    let max_steps = arg_usize(args, 2, 0); // 0 = sınırsız (duman testi için)

    println!(
        "🐟 StonefishAI v{} — GPT eğitimi (batch={}, GPU={})\n",
        stonefish_ai::VERSION,
        batch_size,
        gpu::HAS_CUDA
    );

    let text = data::load_text(TEXT_PATH);
    let tok = Tokenizer::from_text(&text);
    let all_ids = tok.encode(&text);
    let seq_len = 128;
    let seqs = data::make_sequences(&all_ids, seq_len, seq_len);

    // Sondaki bir kısmı validasyon için ayır (ezber mi öğrenme mi?).
    let n_val = (seqs.len() / 50).clamp(4, 256).min(seqs.len() / 4);
    let n_train = seqs.len() - n_val;
    println!(
        "Vocab: {}, sequence: {} (train {}, val {}), seq_len: {}",
        tok.vocab_size, seqs.len(), n_train, n_val, seq_len
    );

    let mut seed: u64 = 0xDEADBEEF;
    let config = GptConfig {
        vocab_size: tok.vocab_size,
        max_seq_len: seq_len,
        d_model: 128,
        n_layers: 4,
        n_heads: 4,
    };
    let model = GPT::new(config, &mut seed);
    let params = model.parameters();
    let n_params = model.n_params();
    println!(
        "Model: d_model={}, layers={}, heads={} (d_head={}), params={}\n",
        config.d_model, config.n_layers, config.n_heads, config.d_head(), n_params
    );

    let mut opt = Adam::new(params, 0.0005);
    let tokens_per_step = batch_size * seq_len;
    let steps_per_epoch = n_train.div_ceil(batch_size);
    let mut global_step = 0usize;
    let mut addrng = 0xA5A5_1234u64;

    for epoch in 0..epochs {
        // Epoch'a özgü karıştırma (eskiden her epoch aynı sırayı kullanıyordu).
        let mut order: Vec<usize> = (0..n_train).collect();
        for i in (1..order.len()).rev() {
            let j = (xorshift(&mut addrng) as usize) % (i + 1);
            order.swap(i, j);
        }

        let epoch_t0 = std::time::Instant::now();
        let mut sum_loss = 0.0f64;
        let mut n_steps = 0usize;

        let mut start = 0;
        while start < n_train {
            let end = (start + batch_size).min(n_train);
            let bs = end - start;

            // Batch'i satır paketle: (bs*seq_len, …)
            let mut input_ids = Vec::with_capacity(bs * seq_len);
            let mut targets = Vec::with_capacity(bs * seq_len);
            for k in start..end {
                let (inp, tgt) = &seqs[order[k]];
                input_ids.extend_from_slice(inp);
                targets.extend_from_slice(tgt);
            }

            let logits = model.forward_batch(&input_ids, bs, seq_len);
            let loss = logits.softmax_cross_entropy_ids(&targets);

            opt.zero_grad();
            loss.backward();
            opt.step();

            let l = loss.item();
            sum_loss += l as f64;
            n_steps += 1;
            global_step += 1;

            start = end;

            if n_steps % 50 == 0 {
                let avg = sum_loss / n_steps as f64;
                let elapsed = epoch_t0.elapsed().as_secs_f64();
                let tps = (n_steps * tokens_per_step) as f64 / elapsed;
                println!(
                    "epoch {} step {:5}/{}  loss = {:.4}  ({:.0} tok/s)",
                    epoch, n_steps, steps_per_epoch, avg, tps
                );
            }

            if max_steps > 0 && global_step >= max_steps {
                println!(
                    "\n⏹  max_steps={} sınırına ulaşıldı (duman testi) — checkpoint yazılmadı.",
                    max_steps
                );
                return;
            }
        }

        let train_loss = sum_loss / n_steps as f64;
        let val_loss = eval_loss(&model, &seqs[n_train..], batch_size, seq_len);
        println!(
            "\nEpoch {} → train CE = {:.4}, val CE = {:.4}, {:.1}s\n",
            epoch,
            train_loss,
            val_loss,
            epoch_t0.elapsed().as_secs_f64()
        );
    }

    println!("Model kaydediliyor → {}", GPT_PATH);
    checkpoint::save_gpt(GPT_PATH, &model, &tok).expect("checkpoint kaydedilemedi");
    if let Err(e) = tok.save(GPT_VOCAB_PATH) {
        eprintln!("⚠️  vocab yan dosyası yazılamadı: {e}");
    }
    println!("✅ Kaydedildi ({} adım).\n🐟 GPT eğitimi tamam.", global_step);
}

/// Gradyansız değerlendirme: ortalama cross-entropy.
fn eval_loss(model: &GPT, seqs: &[(Vec<usize>, Vec<usize>)], batch_size: usize, seq_len: usize) -> f64 {
    if seqs.is_empty() {
        return f64::NAN;
    }
    let mut sum = 0.0f64;
    let mut n = 0usize;
    let mut start = 0;
    while start < seqs.len() {
        let end = (start + batch_size).min(seqs.len());
        let bs = end - start;
        let mut input_ids = Vec::with_capacity(bs * seq_len);
        let mut targets = Vec::with_capacity(bs * seq_len);
        for k in start..end {
            let (inp, tgt) = &seqs[k];
            input_ids.extend_from_slice(inp);
            targets.extend_from_slice(tgt);
        }
        let logits = model.forward_batch(&input_ids, bs, seq_len);
        sum += logits.softmax_cross_entropy_ids(&targets).item() as f64;
        n += 1;
        start = end;
    }
    sum / n as f64
}

// ============================================================
// GPT ÜRETİMİ
// ============================================================

fn gpt_generate(args: &[String]) {
    println!("🐟 StonefishAI v{} — GPT metin üretimi\n", stonefish_ai::VERSION);

    let (model, tok) = match checkpoint::load_gpt(GPT_PATH) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Checkpoint yüklenemedi ({GPT_PATH}): {e}");
            eprintln!("Önce `stonefish_ai gpt-train` çalıştır.");
            std::process::exit(1);
        }
    };
    let cfg = model.config;
    println!(
        "✅ Yüklendi: d_model={}, layers={}, heads={}, vocab={}",
        cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.vocab_size
    );

    // Korpus drift kontrolü: yan dosyadaki vocab checkpoint'le aynı mı?
    match Tokenizer::load(GPT_VOCAB_PATH) {
        Ok(disc) if !disc.same_vocab(&tok) => {
            eprintln!("⚠️  {GPT_VOCAB_PATH} checkpoint vocab'ı ile uyuşmuyor — dosya eski olabilir.");
        }
        Ok(_) => {}
        Err(_) => {}
    }

    let prompt = args
        .first()
        .cloned()
        .unwrap_or_else(|| "Akşam erken iner".to_string());
    let n_generate = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(300usize);
    let temperature = args
        .get(2)
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.8f32);

    println!("Prompt: {:?}  ({} token, sıcaklık {})\n", prompt, n_generate, temperature);

    let mut generated: Vec<usize> = match tok.try_encode(&prompt) {
        Ok(v) => v,
        Err(c) => {
            eprintln!("❌ Prompt'taki {:?} karakteri bu modelin vocab'ında yok.", c);
            std::process::exit(1);
        }
    };
    if generated.is_empty() {
        eprintln!("❌ Prompt boş.");
        std::process::exit(1);
    }

    let mut rng = 0x1234_5678_9ABCu64;

    for _ in 0..n_generate {
        // Bağlamı son seq_len token'a kırp (modelin gördüğü maksimum pencere).
        let start = generated.len().saturating_sub(cfg.max_seq_len);
        let context = &generated[start..];

        let logits = model.forward(context);
        let ld = logits.data.borrow();
        let vocab = ld.cols;
        let last_row = ld.rows - 1;

        let next = if temperature <= 0.0 {
            // açgözlü (deterministik)
            let mut best = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for j in 0..vocab {
                let v = ld.get(last_row, j);
                if v > best_v {
                    best_v = v;
                    best = j;
                }
            }
            best
        } else {
            // sıcaklıklı örnekleme
            let mut max = f32::NEG_INFINITY;
            for j in 0..vocab {
                max = max.max(ld.get(last_row, j));
            }
            let mut probs = vec![0.0f32; vocab];
            let mut sum = 0.0f32;
            for j in 0..vocab {
                let e = ((ld.get(last_row, j) - max) / temperature).exp();
                probs[j] = e;
                sum += e;
            }
            let r = (xorshift(&mut rng) >> 11) as f32 / (1u64 << 53) as f32;
            let mut acc = 0.0f32;
            let mut pick = vocab - 1;
            for (j, p) in probs.iter().enumerate() {
                acc += p / sum;
                if r < acc {
                    pick = j;
                    break;
                }
            }
            pick
        };

        drop(ld);
        generated.push(next);
    }

    println!("--- Üretilen metin ---\n");
    println!("{}", tok.decode(&generated));
}

// ============================================================
// MNIST MLP
// ============================================================

fn train() {
    println!("🐟 StonefishAI v{} — MNIST eğitimi\n", stonefish_ai::VERSION);
    let x_train = load_images("data/train-images-idx3-ubyte");
    let y_train = load_labels_onehot("data/train-labels-idx1-ubyte");
    let x_test = load_images("data/t10k-images-idx3-ubyte");
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
            let j = (xorshift(&mut rng_state) as usize) % (i + 1);
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
                xb.extend_from_slice(&x_train.data[src * 784..(src + 1) * 784]);
                yb.extend_from_slice(&y_train.data[src * 10..(src + 1) * 10]);
            }
            let x_batch = Tensor::leaf(Matrix::from_vec(bs, 784, xb), false);
            let y_batch = Tensor::leaf(Matrix::from_vec(bs, 10, yb), false);
            let logits = l2.forward(&l1.forward(&x_batch).relu());
            let loss = softmax_cross_entropy(&logits, &y_batch);
            opt.zero_grad();
            loss.backward();
            opt.step();
            epoch_loss += loss.item();
            n_batches += 1;
            start = end;
        }
        let avg = epoch_loss / n_batches as f32;
        print!("epoch {:2}  train CE = {:.4}", epoch, avg);
        if epoch % 2 == 1 || epoch == epochs - 1 {
            let acc = evaluate(&x_test, &y_test_raw, &l1, &l2);
            println!("   test acc = %{:.2}", acc);
        } else {
            println!();
        }
    }
    println!("\nModel kaydediliyor → {}", MODEL_PATH);
    save_model(MODEL_PATH, &[&l1, &l2]).expect("kaydetme başarısız");
    println!("✅ Kaydedildi.");
    let acc = evaluate(&x_test, &y_test_raw, &l1, &l2);
    println!("\n🐟 Final test acc = %{:.2}", acc);
}

fn predict() {
    println!("🐟 StonefishAI v{} — tahmin\n", stonefish_ai::VERSION);
    let layers = load_model(MODEL_PATH, 2).expect("model yüklenemedi");
    let x_test = load_images("data/t10k-images-idx3-ubyte");
    let y_test_raw = load_labels_raw("data/t10k-labels-idx1-ubyte");
    let acc = evaluate(&x_test, &y_test_raw, &layers[0], &layers[1]);
    println!("Test accuracy: %{:.2}", acc);
}

fn eval_only() {
    println!("🐟 StonefishAI v{} — eval\n", stonefish_ai::VERSION);
    let layers = load_model(MODEL_PATH, 2).expect("model yüklenemedi");
    let x_test = load_images("data/t10k-images-idx3-ubyte");
    let y_test_raw = load_labels_raw("data/t10k-labels-idx1-ubyte");
    let acc = evaluate(&x_test, &y_test_raw, &layers[0], &layers[1]);
    println!("Test accuracy: %{:.2}", acc);
}

fn evaluate(x: &Matrix, y_raw: &[usize], l1: &Linear, l2: &Linear) -> f32 {
    let bs = 128;
    let n = x.rows;
    let mut correct = 0;
    let mut start = 0;
    while start < n {
        let end = (start + bs).min(n);
        let x_t = Tensor::leaf(slice_rows(x, start, end), false);
        let logits = l2.forward(&l1.forward(&x_t).relu());
        let ld = logits.data.borrow();
        for i in 0..(end - start) {
            let mut best_j = 0;
            let mut best_v = ld.get(i, 0);
            for j in 1..10 {
                let v = ld.get(i, j);
                if v > best_v {
                    best_v = v;
                    best_j = j;
                }
            }
            if best_j == y_raw[start + i] {
                correct += 1;
            }
        }
        drop(ld);
        start = end;
    }
    100.0 * correct as f32 / n as f32
}

// ============================================================
// TOKENIZER
// ============================================================

fn tokenize() {
    println!("🐟 StonefishAI v{} — tokenizer\n", stonefish_ai::VERSION);
    let text = data::load_text(TEXT_PATH);
    let tok = Tokenizer::from_text(&text);
    println!("Vocab: {}", tok.vocab_size);

    // Round-trip: korpustan bir parça ve Türkçe karakterler
    let sample = "Akşam erken iner mahpushaneye";
    match tok.try_encode(sample) {
        Ok(ids) => {
            let decoded = tok.decode(&ids);
            println!("Round-trip: {}", if sample == decoded { "✅" } else { "❌" });
        }
        Err(c) => println!("❌ Örnek metinde bilinmeyen karakter: {:?}", c),
    }

    let all_ids = tok.encode(&text);
    let seqs = data::make_sequences(&all_ids, 64, 64);
    println!("Sequence: {}\n✅ Hazır.", seqs.len());
}

// ============================================================
// DUMAN TESTLERİ
// ============================================================

fn test_ops() {
    println!("🐟 Op testleri\n");
    let x = Tensor::leaf(Matrix::from_vec(1, 3, vec![1.0, 2.0, 3.0]), true);
    x.softmax().sum().backward();
    println!("✅ Softmax OK");
    let x2 = Tensor::leaf(Matrix::from_vec(1, 3, vec![-1.0, 0.0, 1.0]), true);
    x2.gelu().sum().backward();
    println!("✅ GELU OK");
    let x3 = Tensor::leaf(Matrix::from_vec(1, 3, vec![1.0, 2.0, 3.0]), true);
    let gamma = Tensor::leaf(Matrix::from_vec(1, 3, vec![1.0, 1.0, 1.0]), true);
    let beta = Tensor::leaf(Matrix::from_vec(1, 3, vec![0.0, 0.0, 0.0]), true);
    x3.layer_norm(&gamma, &beta, 1e-5).sum().backward();
    println!("✅ LayerNorm OK\n🎉 Tüm op testleri geçti!");
}

fn test_embed() {
    println!("🐟 Embedding testi\n");
    let mut seed = 42u64;
    let (vocab, seq_len, d_model) = (10, 4, 8);
    let emb = transformer::Embedding::new(vocab, 16, d_model, &mut seed);
    let ids = vec![1, 3, 5, 7];
    let out = emb.forward(&ids, seq_len);
    {
        let od = out.data.borrow();
        println!("Çıktı boyutu: {}x{}", od.rows, od.cols);
        assert_eq!((od.rows, od.cols), (seq_len, d_model));
    }
    out.sum().backward();
    let tg = emb.token_emb.grad.borrow();
    let pg = emb.pos_emb.grad.borrow();
    println!("token_emb grad: {}x{}", tg.rows, tg.cols);
    println!("pos_emb grad: {}x{}", pg.rows, pg.cols);

    let mut pos0_sum = 0.0f32;
    for d in 0..d_model {
        pos0_sum += pg.get(0, d);
    }
    assert!((pos0_sum - d_model as f32).abs() < 1e-4, "pos_emb grad toplamı yanlış");
    println!("✅ Embedding OK");
}

fn test_attn() {
    println!("🐟 Attention testi\n");
    let mut seed = 42u64;
    let (d_model, seq_len, n_heads) = (8, 4, 2);
    let attn = transformer::MultiHeadSelfAttention::new(d_model, n_heads, &mut seed);
    let x = Tensor::leaf(Matrix::random(seq_len, d_model, 0.1, &mut seed), true);
    let out = attn.forward(&x, 1, seq_len);
    {
        let od = out.data.borrow();
        println!("Çıktı boyutu: {}x{}", od.rows, od.cols);
        assert_eq!((od.rows, od.cols), (seq_len, d_model));
    }
    out.sum().backward();
    let sum_abs = |g: &Matrix| -> f32 { g.data.iter().map(|x| x.abs()).sum() };
    assert!(sum_abs(&x.grad.borrow()) > 1e-6, "x gradyanı sıfır");
    assert!(sum_abs(&attn.w_q.grad.borrow()) > 1e-6, "w_q gradyanı sıfır");
    assert!(sum_abs(&attn.w_k.grad.borrow()) > 1e-6, "w_k gradyanı sıfır");
    assert!(sum_abs(&attn.w_v.grad.borrow()) > 1e-6, "w_v gradyanı sıfır");
    assert!(sum_abs(&attn.w_o.grad.borrow()) > 1e-6, "w_o gradyanı sıfır");
    println!("✅ Tüm gradyanlar sıfır değil\n🎉 Attention testi geçti!");
}

// ============================================================
// GPU TESTLERİ
// ============================================================

fn test_cuda() {
    println!("🐟 CUDA testi\n");
    if !gpu::HAS_CUDA {
        println!("❌ Bu derlemede CUDA kapalı (--no-default-features).");
        return;
    }
    #[cfg(feature = "cuda")]
    {
        let ctx = gpu::get_ctx();
        println!("GPU: {:?}", ctx.name());
        println!("✅ CUDA çalışıyor!");
    }
}

/// CPU naif matmul ile GPU'yu karşılaştırır.
fn compare_matmul(n: usize) -> (std::time::Duration, std::time::Duration, f32) {
    let mut seed = 42u64;
    let a = Matrix::random(n, n, 1.0, &mut seed);
    let b = Matrix::random(n, n, 1.0, &mut seed);

    let t0 = std::time::Instant::now();
    let c_cpu = a.matmul(&b);
    let cpu_time = t0.elapsed();

    let t0 = std::time::Instant::now();
    let c_gpu = gpu::matmul_gpu(&a, &b);
    let gpu_time = t0.elapsed();

    let mut max_diff = 0.0f32;
    for i in 0..c_cpu.data.len() {
        max_diff = max_diff.max((c_cpu.data[i] - c_gpu.data[i]).abs());
    }
    (cpu_time, gpu_time, max_diff)
}

fn test_gpu_matmul() {
    println!("🐟 GPU matmul testi\n");
    let (cpu, gpu_t, diff) = compare_matmul(64);
    println!("CPU: {:?}, GPU: {:?}, max fark: {:.3e}", cpu, gpu_t, diff);
    if diff < 1e-4 {
        println!("✅ GPU matmul doğru!");
    } else {
        println!("❌ GPU matmul yanlış! (max fark: {})", diff);
        std::process::exit(1);
    }
}

fn test_gpu_bench() {
    println!("🐟 GPU matmul hız testi (256²)\n");
    let (cpu, gpu_t, diff) = compare_matmul(256);
    println!("CPU: {:?}\nGPU (transfer+matmul+transfer): {:?}", cpu, gpu_t);
    println!("Max fark: {:.3e}", diff);
    println!(
        "Hızlanma: {:.2}x",
        cpu.as_secs_f64() / gpu_t.as_secs_f64()
    );
}

fn test_gpu_bench_big() {
    println!("🐟 GPU matmul hız testi (büyük matrisler)\n");
    for &n in &[256usize, 512, 1024] {
        println!("--- {}x{} ---", n, n);
        let (cpu, gpu_t, diff) = compare_matmul(n);
        println!("CPU: {:?}", cpu);
        println!("GPU (transfer+matmul+transfer): {:?}", gpu_t);
        println!("Max fark: {:.3e}", diff);
        println!(
            "Hızlanma: {:.2}x\n",
            cpu.as_secs_f64() / gpu_t.as_secs_f64()
        );
    }
}

/// Modelin gerçekte kullandığı matmul şekilleri — asıl karar bunlara dayanır.
fn test_matmul_bench() {
    println!("🐟 Gerçek model şekillerinde matmul (10 tekrar, ortalama)\n");

    // Isınma: ilk GPU çağrısı CUDA context + cuBLAS handle kurar (~200 ms).
    // Bu maliyet ölçüme karışmasın.
    {
        let mut seed = 1u64;
        let a = Matrix::random(64, 64, 1.0, &mut seed);
        let b = Matrix::random(64, 64, 1.0, &mut seed);
        std::hint::black_box(gpu::matmul_gpu(&a, &b));
    }

    // (B*S, k) @ (k, n) — GPT'nin içindeki GEMM'ler
    let shapes = [
        (1024usize, 128usize, 128usize, "attn proj (B*S=1024)"),
        (1024, 128, 512, "ff1"),
        (1024, 512, 128, "ff2"),
        (1024, 128, 136, "lm head"),
        (128, 128, 128, "batch=1 (üretim)"),
    ];
    let reps = 10;
    for (m, k, n, label) in shapes {
        let mut seed = 7u64;
        let a = Matrix::random(m, k, 1.0, &mut seed);
        let b = Matrix::random(k, n, 1.0, &mut seed);

        // her iki yol için de birer ısınma
        std::hint::black_box(a.matmul(&b));
        std::hint::black_box(gpu::matmul_gpu(&a, &b));

        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            std::hint::black_box(a.matmul(&b));
        }
        let cpu = t0.elapsed().as_secs_f64() / reps as f64;

        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            std::hint::black_box(gpu::matmul_gpu(&a, &b));
        }
        let gpu_t = t0.elapsed().as_secs_f64() / reps as f64;

        println!(
            "{:<22} ({m}×{k})@({k}×{n})  CPU {:8.3} ms   GPU {:8.3} ms   {:.2}x",
            label,
            cpu * 1e3,
            gpu_t * 1e3,
            cpu / gpu_t
        );
    }
}

fn test_gpu_train() {
    println!("🐟 GPU eğitim testi (küçük MLP)\n");
    let mut seed = 42u64;
    let l1 = Linear::new(128, 64, &mut seed);
    let l2 = Linear::new(64, 10, &mut seed);
    let mut params = Vec::new();
    params.extend(l1.parameters());
    params.extend(l2.parameters());
    let opt = SGD::new(params, 0.01);

    let batch = 32;
    let mut x_data = Vec::new();
    let mut y_data = Vec::new();
    for _ in 0..batch {
        for _ in 0..128 {
            x_data.push(rand_f32(&mut seed));
        }
        let mut oh = vec![0.0f32; 10];
        oh[(xorshift(&mut seed) as usize) % 10] = 1.0;
        y_data.extend(oh);
    }
    let x = Tensor::leaf(Matrix::from_vec(batch, 128, x_data), false);
    let y = Tensor::leaf(Matrix::from_vec(batch, 10, y_data), false);

    let t0 = std::time::Instant::now();
    for _ in 0..10 {
        let logits = l2.forward(&l1.forward(&x).relu());
        let loss = softmax_cross_entropy(&logits, &y);
        opt.zero_grad();
        loss.backward();
        opt.step();
    }
    println!("10 adım: {:?}  (GPU={})", t0.elapsed(), gpu::HAS_CUDA);
    println!("✅ GPU eğitim testi tamam!");
}

fn test_gpt_gpu(args: &[String]) {
    let steps: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(20);
    println!("🐟 GPT adım hızı (d_model=64, 2 katman, batch=8)\n");

    let text = data::load_text(TEXT_PATH);
    let tok = Tokenizer::from_text(&text);
    let all_ids = tok.encode(&text);
    let seq_len = 64;
    let seqs = data::make_sequences(&all_ids, seq_len, seq_len);

    let mut seed: u64 = 0xDEADBEEF;
    let config = GptConfig {
        vocab_size: tok.vocab_size,
        max_seq_len: seq_len,
        d_model: 64,
        n_layers: 2,
        n_heads: 2,
    };
    let model = GPT::new(config, &mut seed);
    let mut opt = Adam::new(model.parameters(), 0.0005);
    let batch = 8;

    let make_batch = |i: usize| -> (Vec<usize>, Vec<usize>) {
        let mut input_ids = Vec::with_capacity(batch * seq_len);
        let mut targets = Vec::with_capacity(batch * seq_len);
        for k in 0..batch {
            let (inp, tgt) = &seqs[(i * batch + k) % seqs.len()];
            input_ids.extend_from_slice(inp);
            targets.extend_from_slice(tgt);
        }
        (input_ids, targets)
    };

    // Isınma (CUDA context + cuBLAS handle)
    {
        let (ids, tgt) = make_batch(0);
        let l = model.forward_batch(&ids, batch, seq_len).softmax_cross_entropy_ids(&tgt);
        opt.zero_grad();
        l.backward();
    }

    // --- sadece forward ---
    let t0 = std::time::Instant::now();
    for i in 0..steps {
        let (ids, tgt) = make_batch(i);
        std::hint::black_box(
            model.forward_batch(&ids, batch, seq_len).softmax_cross_entropy_ids(&tgt),
        );
    }
    let fwd = t0.elapsed().as_secs_f64() / steps as f64;

    // --- forward + backward + step ---
    let t0 = std::time::Instant::now();
    let mut total = 0.0f32;
    for i in 0..steps {
        let (ids, tgt) = make_batch(i);
        let loss = model.forward_batch(&ids, batch, seq_len).softmax_cross_entropy_ids(&tgt);
        opt.zero_grad();
        loss.backward();
        opt.step();
        total += loss.item();
    }
    let step_t = t0.elapsed().as_secs_f64() / steps as f64;

    let tokens = batch * seq_len;
    println!(
        "{} adım → {:.2} ms/adım ({:.0} tok/s)  GPU={}",
        steps,
        step_t * 1e3,
        tokens as f64 / step_t,
        gpu::HAS_CUDA
    );
    println!(
        "  forward      : {:.2} ms ({:.0}%)",
        fwd * 1e3,
        fwd / step_t * 100.0
    );
    println!(
        "  backward+step: {:.2} ms ({:.0}%)",
        (step_t - fwd) * 1e3,
        (step_t - fwd) / step_t * 100.0
    );
    println!("Ortalama loss: {:.4}", total / steps as f32);
    println!("✅ GPT hız testi tamam!");
}

// ============================================================
// Yardımcılar
// ============================================================

fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

fn rand_f32(seed: &mut u64) -> f32 {
    (xorshift(seed) >> 11) as f32 / (1u64 << 53) as f32
}
