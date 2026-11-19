// src/main.rs
// CLI shell. The whole engine lives in the `stonefish_ai` library (src/lib.rs).
//
// Inner loops index several arrays at once (see src/lib.rs).
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_is_multiple_of)]
use stonefish_ai::autograd::Tensor;
use stonefish_ai::checkpoint;
use stonefish_ai::data::{load_images, load_labels_onehot, load_labels_raw, slice_rows};
use stonefish_ai::loss::softmax_cross_entropy;
use stonefish_ai::nn::{load_model, save_model, Linear};
use stonefish_ai::optim::{Adam, SGD};
use stonefish_ai::tensor::Matrix;
use stonefish_ai::tokenizer::Tokenizer;
use stonefish_ai::trainer::Trainer;
use stonefish_ai::transformer::{GptConfig, GPT};
use stonefish_ai::{data, gpu, profile, transformer};

const MODEL_PATH: &str = "stonefish.bin";
const GPT_PATH: &str = "stonefish_gpt.bin";
/// The vocab file written next to the checkpoint (to detect corpus drift).
const GPT_VOCAB_PATH: &str = "stonefish_gpt.vocab";
const TEXT_PATH: &str = "data/turkish.txt";

/// All CLI modes: (name, short description). `help` prints it, `main` dispatches it.
const MODES: &[(&str, &str)] = &[
    ("train", "train the MNIST MLP (784->128->10, SGD)"),
    ("eval", "measure the saved MNIST model on the test set"),
    (
        "predict",
        "predict with the saved MNIST model (same as eval)",
    ),
    (
        "tokenize",
        "validate the tokenizer and print the sequence count",
    ),
    (
        "gpt-train",
        "train the mini GPT [epochs] [batch] [max_steps] [max_lr]",
    ),
    (
        "gpt-gen",
        "generate text with the saved GPT  [prompt] [tokens] [temperature]",
    ),
    ("test-ops", "op smoke test"),
    ("test-embed", "embedding smoke test"),
    ("test-attn", "attention smoke test"),
    ("test-cuda", "can a CUDA context be created"),
    ("test-gpu-matmul", "GPU matmul == CPU matmul"),
    ("test-gpu-bench", "256² matmul: CPU vs GPU"),
    ("test-gpu-bench-big", "256/512/1024² matmul: CPU vs GPU"),
    (
        "test-matmul-bench",
        "CPU vs GPU matmul at real model shapes",
    ),
    ("test-gpu-train", "time 10 steps on a small MLP"),
    ("test-gpt-gpu", "step time on a small GPT"),
    (
        "test-gpt-profile",
        "where the time goes in the real model (matmul/transfer/attn)",
    ),
    (
        "test-gpu-native",
        "measure the GPU-native trainer (device-resident, kernels)",
    ),
    (
        "en-prepare",
        "English corpus: learn BPE + build the u16 token cache",
    ),
    ("en-train", "train the English GPT [token_budget] [d_model]"),
    (
        "demo",
        "one-command demo: train from scratch in 90 s + generate [seconds]",
    ),
    (
        "test-ffi",
        "self-test of our own CUDA FFI (driver, cuBLAS, NVRTC)",
    ),
    (
        "en-tokenize",
        "tokenize a text file with the model's BPE so en-eval can score it (probe)",
    ),
    (
        "en-eval",
        "evaluate the English model on the validation set (loss/perplexity/BPC)",
    ),
    (
        "en-gen",
        "generate English text [prompt] [tokens] [temperature]",
    ),
];

fn arg_usize(args: &[String], i: usize, default: usize) -> usize {
    args.get(i).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn arg_f32(args: &[String], i: usize, default: f32) -> f32 {
    args.get(i).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rest: &[String] = if args.len() > 2 { &args[2..] } else { &[] };
    // No argument prints the menu. It used to default to "train", which meant
    // running the binary to see the options started a six-epoch MNIST run and
    // wrote a model file - a surprise nobody wants from --help-less usage.
    let mode = if args.len() < 2 {
        "help"
    } else {
        args[1].as_str()
    };

    match mode {
        "help" | "-h" | "--help" => {
            println!("🐟 StonefishAI v{}", stonefish_ai::VERSION);
            println!(
                "GPU path: {}",
                if gpu::HAS_CUDA {
                    "available (CUDA)"
                } else {
                    "not available (CPU)"
                }
            );
            println!("\nUsage: stonefish_ai <mode> [arg...]\n\nModes:");
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
        "test-gpt-profile" => test_gpt_profile(rest),
        "test-gpu-native" => test_gpu_native(rest),
        "en-prepare" => en_prepare(),
        "en-train" => en_train(rest),
        "demo" => demo(rest),
        "bench" => bench(rest),
        "test-ffi" => test_ffi(),
        "en-tokenize" => en_tokenize(rest),
        "en-eval" => en_eval(rest),
        "en-gen" => en_gen(rest),
        other => {
            eprintln!("Unknown mode: {}", other);
            eprintln!(
                "Valid modes: {}",
                MODES
                    .iter()
                    .map(|(n, _)| *n)
                    .collect::<Vec<_>>()
                    .join(" | ")
            );
            eprintln!("For details: stonefish_ai help");
            std::process::exit(1);
        }
    }
}

// ============================================================
// GPT TRAINING - real batch (a single forward/backward/step)
// ============================================================

fn gpt_train(args: &[String]) {
    let epochs = arg_usize(args, 0, 6);
    let batch_size = arg_usize(args, 1, 8);
    let max_steps = arg_usize(args, 2, 0); // 0 = unlimited (for smoke tests)
    let max_lr = arg_f32(args, 3, 1e-3);

    // --- Hyperparameters (one place, logged and written to the manifest) ---
    const SEQ_LEN: usize = 128;
    const D_MODEL: usize = 128;
    const N_LAYERS: usize = 4;
    const N_HEADS: usize = 4;
    const WARMUP_STEPS: usize = 150;
    const MIN_LR_RATIO: f32 = 0.1;
    const CLIP_NORM: f32 = 1.0;
    let model_seed = 0xDEADBEEFu64;
    let shuffle_seed = 0xA5A5_1234u64;

    let run_t0 = std::time::Instant::now();
    println!("🐟 StonefishAI v{} - GPT training\n", stonefish_ai::VERSION);
    println!(
        "Hardware : GPU={} ({})\n",
        gpu::HAS_CUDA,
        if gpu::HAS_CUDA { "CUDA" } else { "CPU" }
    );

    let text = data::load_text(TEXT_PATH);
    let tok = Tokenizer::from_text(&text);
    let all_ids = tok.encode(&text);
    let seqs = data::make_sequences(&all_ids, SEQ_LEN, SEQ_LEN);

    // Hold out the tail for validation (memorization or learning?).
    let n_val = (seqs.len() / 50).clamp(4, 256).min(seqs.len() / 4);
    let mut n_train = seqs.len() - n_val;
    // The GPU trainer is built for a fixed (batch, seq), so train and val are
    // dilimleri tam batch'lere indirilir; artan en fazla batch_size-1 sequence
    // skipped (the measurement impact is negligible).
    if n_train % batch_size != 0 {
        n_train -= n_train % batch_size;
    }
    let n_val = {
        let r = seqs.len() - n_train;
        r - r % batch_size
    };

    let mut seed = model_seed;
    let config = GptConfig {
        vocab_size: tok.vocab_size,
        max_seq_len: SEQ_LEN,
        d_model: D_MODEL,
        n_layers: N_LAYERS,
        n_heads: N_HEADS,
    };
    let mut model = GPT::new(config, &mut seed);
    let n_params = model.n_params();

    let tokens_per_step = batch_size * SEQ_LEN;
    let steps_per_epoch = n_train.div_ceil(batch_size);
    let total_steps = steps_per_epoch * epochs;

    println!(
        "Veri    : {} karakter → {} sequence (train {}, val {}), seq_len {}",
        all_ids.len(),
        seqs.len(),
        n_train,
        n_val,
        SEQ_LEN
    );
    println!(
        "Model   : d_model={}, layers={}, heads={} (d_head={}), params={}",
        config.d_model,
        config.n_layers,
        config.n_heads,
        config.d_head(),
        n_params
    );
    println!(
        "Optim   : Adam beta=(0.9,0.999), max_lr={}, warmup={} steps, cosine -> {}x, clip_norm={}",
        max_lr, WARMUP_STEPS, MIN_LR_RATIO, CLIP_NORM
    );
    println!(
        "Plan    : {} epochs x {} steps = {} steps, batch={} ({} tokens/step)",
        epochs, steps_per_epoch, total_steps, batch_size, tokens_per_step
    );
    println!(
        "Seed    : model=0x{:X}, shuffle=0x{:X}",
        model_seed, shuffle_seed
    );
    println!();

    let mut trainer = Trainer::new(config, model_seed, batch_size, SEQ_LEN);
    let mut global_step = 0usize;
    let mut addrng = shuffle_seed;

    // Track the best validation: refresh the checkpoint whenever it improves.
    let mut best_val = f64::INFINITY;
    let mut best_epoch = 0usize;
    let mut history: Vec<(usize, f64, f64, f64, f32)> = Vec::new(); // epoch, train, val, seconds, lr

    for epoch in 0..epochs {
        // Epoch-specific shuffling (a different order every epoch).
        let mut order: Vec<usize> = (0..n_train).collect();
        for i in (1..order.len()).rev() {
            let j = (xorshift(&mut addrng) as usize) % (i + 1);
            order.swap(i, j);
        }

        let epoch_t0 = std::time::Instant::now();
        let mut sum_loss = 0.0f64;
        let mut n_steps = 0usize;
        let mut grad_norm_sum = 0.0f64;
        let mut clipped = 0usize;

        let mut start = 0;
        while start < n_train {
            let end = (start + batch_size).min(n_train);
            let bs = end - start;

            // --- LR schedule: linear warmup + cosine decay ---
            let lr = lr_at(global_step, total_steps, max_lr, WARMUP_STEPS, MIN_LR_RATIO);

            // Pack the batch by rows: (bs*seq_len, ...)
            let mut input_ids = Vec::with_capacity(bs * SEQ_LEN);
            let mut targets = Vec::with_capacity(bs * SEQ_LEN);
            for k in start..end {
                let (inp, tgt) = &seqs[order[k]];
                input_ids.extend_from_slice(inp);
                targets.extend_from_slice(tgt);
            }

            // One step: forward + backward + Adam + gradient clipping
            let (l, gnorm) = trainer.train_step(&input_ids, &targets, bs, SEQ_LEN, lr, CLIP_NORM);
            if gnorm > CLIP_NORM {
                clipped += 1;
            }
            grad_norm_sum += gnorm as f64;
            sum_loss += l as f64;
            n_steps += 1;
            global_step += 1;
            start = end;

            if n_steps % 50 == 0 {
                let avg = sum_loss / n_steps as f64;
                let elapsed = epoch_t0.elapsed().as_secs_f64();
                let tps = (n_steps * tokens_per_step) as f64 / elapsed;
                println!(
                    "epoch {} step {:5}/{}  loss {:.4}  lr {:.2e}  gnorm {:.2}  ({:.0} tok/s)",
                    epoch, n_steps, steps_per_epoch, avg, lr, gnorm, tps
                );
            }

            if max_steps > 0 && global_step >= max_steps {
                println!(
                    "\n⏹  reached max_steps={} (smoke test) - no checkpoint written.",
                    max_steps
                );
                return;
            }
        }

        let epoch_secs = epoch_t0.elapsed().as_secs_f64();
        let train_loss = sum_loss / n_steps as f64;
        let val_loss = trainer.eval_loss(&seqs[n_train..n_train + n_val], batch_size, SEQ_LEN);
        let last_lr = lr_at(
            global_step.saturating_sub(1),
            total_steps,
            max_lr,
            WARMUP_STEPS,
            MIN_LR_RATIO,
        );
        history.push((epoch, train_loss, val_loss, epoch_secs, last_lr));

        let improved = val_loss < best_val;
        println!(
            "\nEpoch {} -> train CE = {:.4}, val CE = {:.4} ({:.1}s, mean gnorm {:.2}, clipped steps {:.1}%){}\n",
            epoch,
            train_loss,
            val_loss,
            epoch_secs,
            grad_norm_sum / n_steps as f64,
            100.0 * clipped as f64 / n_steps as f64,
            if improved { "  ✅ new best" } else { "" }
        );

        if improved {
            best_val = val_loss;
            best_epoch = epoch;
            trainer.sync_to_cpu(&mut model);
            checkpoint::save_gpt(GPT_PATH, &model, &tok)
                .expect("could not save the best checkpoint");
            let _ = tok.save(GPT_VOCAB_PATH);
        }
    }

    // Also save the final state (to compare against the best one).
    trainer.sync_to_cpu(&mut model);
    let final_path = GPT_PATH.replace(".bin", "_final.bin");
    if let Err(e) = checkpoint::save_gpt(&final_path, &model, &tok) {
        eprintln!("⚠️  could not write the final checkpoint ({final_path}): {e}");
    }

    let total_secs = run_t0.elapsed().as_secs_f64();
    let manifest = write_manifest(
        config,
        n_params,
        max_lr,
        batch_size,
        epochs,
        global_step,
        tokens_per_step,
        model_seed,
        shuffle_seed,
        WARMUP_STEPS,
        MIN_LR_RATIO,
        CLIP_NORM,
        best_epoch,
        best_val,
        &history,
        total_secs,
    );

    println!(
        "En iyi model : epoch {} (val CE = {:.4}) → {}",
        best_epoch, best_val, GPT_PATH
    );
    println!("Nihai model  : {}", final_path);
    match manifest {
        Ok(path) => println!("Manifest     : {}", path),
        Err(e) => eprintln!("⚠️  could not write the manifest: {e}"),
    }
    println!(
        "\nTotal {:.1} min, {} steps, average {:.0} tok/s\n🐟 GPT training complete.",
        total_secs / 60.0,
        global_step,
        (global_step * tokens_per_step) as f64 / total_secs
    );
}

/// Linear warmup + cosine decay. `step` is 0-based.
fn lr_at(step: usize, total: usize, max_lr: f32, warmup: usize, min_ratio: f32) -> f32 {
    if step < warmup {
        // A small but positive LR on the first step instead of starting from 0
        return max_lr * (step + 1) as f32 / warmup as f32;
    }
    let span = total.saturating_sub(warmup).max(1) as f32;
    let t = ((step - warmup) as f32 / span).min(1.0);
    let cos = 0.5 * (1.0 + (std::f32::consts::PI * t).cos());
    max_lr * (min_ratio + (1.0 - min_ratio) * cos)
}

/// The manifest (JSON) that makes the run reproducible. Written under `runs/`.
#[allow(clippy::too_many_arguments)]
fn write_manifest(
    config: GptConfig,
    n_params: usize,
    max_lr: f32,
    batch_size: usize,
    epochs: usize,
    total_steps: usize,
    tokens_per_step: usize,
    model_seed: u64,
    shuffle_seed: u64,
    warmup: usize,
    min_lr_ratio: f32,
    clip_norm: f32,
    best_epoch: usize,
    best_val: f64,
    history: &[(usize, f64, f64, f64, f32)],
    total_secs: f64,
) -> std::io::Result<String> {
    use std::io::Write;

    std::fs::create_dir_all("runs")?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let path = format!("runs/gpt-{stamp}.json");

    let mut s = String::new();
    s.push_str("{\n");
    s.push_str(&format!(
        "  \"stonefish_version\": \"{}\",\n",
        stonefish_ai::VERSION
    ));
    s.push_str(&format!("  \"gpu\": {},\n", gpu::HAS_CUDA));
    s.push_str("  \"config\": {\n");
    s.push_str(&format!("    \"vocab_size\": {},\n", config.vocab_size));
    s.push_str(&format!("    \"max_seq_len\": {},\n", config.max_seq_len));
    s.push_str(&format!("    \"d_model\": {},\n", config.d_model));
    s.push_str(&format!("    \"n_layers\": {},\n", config.n_layers));
    s.push_str(&format!("    \"n_heads\": {},\n", config.n_heads));
    s.push_str(&format!("    \"d_head\": {},\n", config.d_head()));
    s.push_str(&format!("    \"n_params\": {}\n", n_params));
    s.push_str("  },\n");
    s.push_str("  \"optim\": {\n");
    s.push_str("    \"name\": \"adam\",\n");
    s.push_str("    \"beta1\": 0.9,\n");
    s.push_str("    \"beta2\": 0.999,\n");
    s.push_str(&format!("    \"max_lr\": {max_lr},\n"));
    s.push_str(&format!("    \"warmup_steps\": {warmup},\n"));
    s.push_str(&format!("    \"min_lr_ratio\": {min_lr_ratio},\n"));
    s.push_str("    \"schedule\": \"linear_warmup+cosine\",\n");
    s.push_str(&format!("    \"clip_norm\": {clip_norm}\n"));
    s.push_str("  },\n");
    s.push_str("  \"run\": {\n");
    s.push_str(&format!("    \"epochs\": {epochs},\n"));
    s.push_str(&format!("    \"batch_size\": {batch_size},\n"));
    s.push_str(&format!("    \"tokens_per_step\": {tokens_per_step},\n"));
    s.push_str(&format!("    \"total_steps\": {total_steps},\n"));
    s.push_str(&format!("    \"model_seed\": \"0x{model_seed:X}\",\n"));
    s.push_str(&format!("    \"shuffle_seed\": \"0x{shuffle_seed:X}\",\n"));
    s.push_str(&format!("    \"wall_seconds\": {:.1}\n", total_secs));
    s.push_str("  },\n");
    s.push_str(&format!("  \"best_epoch\": {best_epoch},\n"));
    s.push_str(&format!("  \"best_val_ce\": {:.6},\n", best_val));
    s.push_str("  \"epochs\": [\n");
    for (i, (ep, train, val, secs, lr)) in history.iter().enumerate() {
        s.push_str(&format!(
            "    {{\"epoch\": {ep}, \"train_ce\": {train:.6}, \"val_ce\": {val:.6}, \"seconds\": {secs:.1}, \"last_lr\": {lr:.6}}}{}\n",
            if i + 1 < history.len() { "," } else { "" }
        ));
    }
    s.push_str("  ]\n}\n");

    let mut f = std::fs::File::create(&path)?;
    f.write_all(s.as_bytes())?;
    Ok(path)
}

// ============================================================
// GPT GENERATION
// ============================================================

fn gpt_generate(args: &[String]) {
    println!(
        "🐟 StonefishAI v{} - GPT text generation\n",
        stonefish_ai::VERSION
    );

    let (model, tok) = match checkpoint::load_gpt(GPT_PATH) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Could not load the checkpoint ({GPT_PATH}): {e}");
            eprintln!("Run `stonefish_ai gpt-train` first.");
            std::process::exit(1);
        }
    };
    let cfg = model.config;
    println!(
        "✅ Loaded: d_model={}, layers={}, heads={}, vocab={}",
        cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.vocab_size
    );

    // Corpus drift check: does the sidecar vocab match the checkpoint?
    match Tokenizer::load(GPT_VOCAB_PATH) {
        Ok(disc) if !disc.same_vocab(&tok) => {
            eprintln!(
                "⚠️  {GPT_VOCAB_PATH} does not match the checkpoint vocab - the file may be stale."
            );
        }
        Ok(_) => {}
        Err(_) => {}
    }

    let prompt = args
        .first()
        .cloned()
        .unwrap_or_else(|| "evening falls early".to_string());
    let n_generate = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(300usize);
    let temperature = args
        .get(2)
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.8f32);

    println!(
        "Prompt: {:?}  ({} tokens, temperature {})\n",
        prompt, n_generate, temperature
    );

    let mut generated: Vec<usize> = match tok.try_encode(&prompt) {
        Ok(v) => v,
        Err(c) => {
            eprintln!(
                "❌ The character {:?} in the prompt is not in this model's vocabulary.",
                c
            );
            std::process::exit(1);
        }
    };
    if generated.is_empty() {
        eprintln!("❌ The prompt is empty.");
        std::process::exit(1);
    }

    let mut rng = 0x1234_5678_9ABCu64;

    for _ in 0..n_generate {
        // Clip the context to the last seq_len tokens (the maximum window the model sees).
        let start = generated.len().saturating_sub(cfg.max_seq_len);
        let context = &generated[start..];

        let logits = model.forward(context);
        let ld = logits.data.borrow();
        let vocab = ld.cols;
        let last_row = ld.rows - 1;

        let next = if temperature <= 0.0 {
            // greedy (deterministic)
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
            // temperature sampling
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

    println!("--- Generated text ---\n");
    println!("{}", tok.decode(&generated));
}

// ============================================================
// MNIST MLP
// ============================================================

fn train() {
    println!(
        "🐟 StonefishAI v{} - MNIST training\n",
        stonefish_ai::VERSION
    );
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
    println!("\nSaving model -> {}", MODEL_PATH);
    save_model(MODEL_PATH, &[&l1, &l2]).expect("save failed");
    println!("✅ Saved.");
    let acc = evaluate(&x_test, &y_test_raw, &l1, &l2);
    println!("\n🐟 Final test acc = %{:.2}", acc);
}

fn predict() {
    println!("🐟 StonefishAI v{} — tahmin\n", stonefish_ai::VERSION);
    let layers = load_model(MODEL_PATH, 2).expect("could not load the model");
    let x_test = load_images("data/t10k-images-idx3-ubyte");
    let y_test_raw = load_labels_raw("data/t10k-labels-idx1-ubyte");
    let acc = evaluate(&x_test, &y_test_raw, &layers[0], &layers[1]);
    println!("Test accuracy: %{:.2}", acc);
}

fn eval_only() {
    println!("🐟 StonefishAI v{} — eval\n", stonefish_ai::VERSION);
    let layers = load_model(MODEL_PATH, 2).expect("could not load the model");
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

    // Round-trip: a slice of the corpus plus non-ASCII characters
    let sample = "the quick brown fox jumps";
    match tok.try_encode(sample) {
        Ok(ids) => {
            let decoded = tok.decode(&ids);
            println!(
                "Round-trip: {}",
                if sample == decoded { "✅" } else { "❌" }
            );
        }
        Err(c) => println!("❌ Unknown character in the sample text: {:?}", c),
    }

    let all_ids = tok.encode(&text);
    let seqs = data::make_sequences(&all_ids, 64, 64);
    println!("Sequences: {}\n✅ Ready.", seqs.len());
}

// ============================================================
// SMOKE TESTS
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
    println!("✅ LayerNorm OK\n🎉 All op tests passed!");
}

fn test_embed() {
    println!("🐟 Embedding test\n");
    let mut seed = 42u64;
    let (vocab, seq_len, d_model) = (10, 4, 8);
    let emb = transformer::Embedding::new(vocab, d_model, &mut seed);
    let ids = vec![1, 3, 5, 7];
    let out = emb.forward(&ids);
    {
        let od = out.data.borrow();
        println!("Output size: {}x{}", od.rows, od.cols);
        assert_eq!((od.rows, od.cols), (seq_len, d_model));
    }
    out.sum().backward();
    let tg = emb.token_emb.grad.borrow();
    println!("token_emb grad: {}x{}", tg.rows, tg.cols);
    // since loss = sum(output) every token row's gradient is 1;
    // tokens 1 and 3 appear once, so those rows must sum to d_model.
    let mut s1 = 0.0f32;
    for d in 0..d_model {
        s1 += tg.get(1, d);
    }
    assert!(
        (s1 - d_model as f32).abs() < 1e-4,
        "token_emb gradient sum is wrong"
    );
    println!("✅ Embedding OK (RoPE: no positional embedding)");
}

fn test_attn() {
    println!("🐟 Attention test\n");
    let mut seed = 42u64;
    let (d_model, seq_len, n_heads) = (8, 4, 2);
    let attn = transformer::MultiHeadSelfAttention::new(d_model, n_heads, &mut seed);
    let x = Tensor::leaf(Matrix::random(seq_len, d_model, 0.1, &mut seed), true);
    let out = attn.forward(&x, 1, seq_len);
    {
        let od = out.data.borrow();
        println!("Output size: {}x{}", od.rows, od.cols);
        assert_eq!((od.rows, od.cols), (seq_len, d_model));
    }
    out.sum().backward();
    let sum_abs = |g: &Matrix| -> f32 { g.data.iter().map(|x| x.abs()).sum() };
    assert!(sum_abs(&x.grad.borrow()) > 1e-6, "x gradient is zero");
    assert!(
        sum_abs(&attn.w_q.grad.borrow()) > 1e-6,
        "w_q gradient is zero"
    );
    assert!(
        sum_abs(&attn.w_k.grad.borrow()) > 1e-6,
        "w_k gradient is zero"
    );
    assert!(
        sum_abs(&attn.w_v.grad.borrow()) > 1e-6,
        "w_v gradient is zero"
    );
    assert!(
        sum_abs(&attn.w_o.grad.borrow()) > 1e-6,
        "w_o gradient is zero"
    );
    println!("✅ No gradient is zero\n🎉 Attention test passed!");
}

// ============================================================
// GPU TESTS
// ============================================================

// `needless_return`: with the `cuda` feature off the cfg block below vanishes,
// which makes the early `return` the last statement of the function. With CUDA
// on it is a genuine early exit, so the lint is a false positive here.
#[allow(clippy::needless_return)]
fn test_cuda() {
    println!("🐟 CUDA test\n");
    if !gpu::HAS_CUDA {
        println!("❌ CUDA is off in this build (--no-default-features).");
        return;
    }
    #[cfg(feature = "cuda")]
    {
        let (name, total) = stonefish_ai::sys::dev::device_info();
        let (maj, min) = stonefish_ai::sys::dev::compute_capability();
        println!("GPU: {} (sm_{}{})", name, maj, min);
        println!("Memory: {:.1} GB", total as f64 / (1u64 << 30) as f64);
        println!("✅ CUDA works! (our own FFI, no external crates)");
    }
}

/// Compares the GPU against a naive CPU matmul.
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
    println!("🐟 GPU matmul test\n");
    let (cpu, gpu_t, diff) = compare_matmul(64);
    println!("CPU: {:?}, GPU: {:?}, max diff: {:.3e}", cpu, gpu_t, diff);
    if diff < 1e-4 {
        println!("✅ GPU matmul is correct!");
    } else {
        println!("❌ GPU matmul is wrong! (max diff: {})", diff);
        std::process::exit(1);
    }
}

fn test_gpu_bench() {
    println!("🐟 GPU matmul speed test (256^2)\n");
    let (cpu, gpu_t, diff) = compare_matmul(256);
    println!(
        "CPU: {:?}\nGPU (transfer+matmul+transfer): {:?}",
        cpu, gpu_t
    );
    println!("Max diff: {:.3e}", diff);
    println!("Speed-up: {:.2}x", cpu.as_secs_f64() / gpu_t.as_secs_f64());
}

fn test_gpu_bench_big() {
    println!("🐟 GPU matmul speed test (large matrices)\n");
    for &n in &[256usize, 512, 1024] {
        println!("--- {}x{} ---", n, n);
        let (cpu, gpu_t, diff) = compare_matmul(n);
        println!("CPU: {:?}", cpu);
        println!("GPU (transfer+matmul+transfer): {:?}", gpu_t);
        println!("Max diff: {:.3e}", diff);
        println!(
            "Speed-up: {:.2}x\n",
            cpu.as_secs_f64() / gpu_t.as_secs_f64()
        );
    }
}

/// The matmul shapes the model really uses - the real decision rests on these.
fn test_matmul_bench() {
    println!("🐟 Matmul at real model shapes (10 repeats, averaged)\n");

    // Warm-up: the first GPU call builds the CUDA context + cuBLAS handle (~200 ms).
    // Keep that cost out of the measurement.
    {
        let mut seed = 1u64;
        let a = Matrix::random(64, 64, 1.0, &mut seed);
        let b = Matrix::random(64, 64, 1.0, &mut seed);
        std::hint::black_box(gpu::matmul_gpu(&a, &b));
    }

    // (B*S, k) @ (k, n) - the GEMMs inside the GPT
    let shapes = [
        (1024usize, 128usize, 128usize, "attn proj (B*S=1024)"),
        (1024, 128, 512, "ff1"),
        (1024, 512, 128, "ff2"),
        (1024, 128, 136, "lm head"),
        (128, 128, 128, "batch=1 (generation)"),
    ];
    let reps = 10;
    for (m, k, n, label) in shapes {
        let mut seed = 7u64;
        let a = Matrix::random(m, k, 1.0, &mut seed);
        let b = Matrix::random(k, n, 1.0, &mut seed);

        // one warm-up for each path
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
    println!("🐟 GPU training test (small MLP)\n");
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
    println!("10 steps: {:?}  (GPU={})", t0.elapsed(), gpu::HAS_CUDA);
    println!("✅ GPU training test complete!");
}

fn test_gpt_gpu(args: &[String]) {
    let steps: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(20);
    println!("🐟 GPT step speed (d_model=64, 2 layers, batch=8)\n");

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

    // Warm-up (CUDA context + cuBLAS handle)
    {
        let (ids, tgt) = make_batch(0);
        let l = model
            .forward_batch(&ids, batch, seq_len)
            .softmax_cross_entropy_ids(&tgt);
        opt.zero_grad();
        l.backward();
    }

    // --- forward only ---
    let t0 = std::time::Instant::now();
    for i in 0..steps {
        let (ids, tgt) = make_batch(i);
        std::hint::black_box(
            model
                .forward_batch(&ids, batch, seq_len)
                .softmax_cross_entropy_ids(&tgt),
        );
    }
    let fwd = t0.elapsed().as_secs_f64() / steps as f64;

    // --- forward + backward + step ---
    let t0 = std::time::Instant::now();
    let mut total = 0.0f32;
    for i in 0..steps {
        let (ids, tgt) = make_batch(i);
        let loss = model
            .forward_batch(&ids, batch, seq_len)
            .softmax_cross_entropy_ids(&tgt);
        opt.zero_grad();
        loss.backward();
        opt.step();
        total += loss.item();
    }
    let step_t = t0.elapsed().as_secs_f64() / steps as f64;

    let tokens = batch * seq_len;
    println!(
        "{} steps -> {:.2} ms/step ({:.0} tok/s)  GPU={}",
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
    println!("Average loss: {:.4}", total / steps as f32);
    println!("✅ GPT speed test complete!");
}

// ============================================================
// English pipeline (BPE + disk cache + 250M-class model)
// ============================================================

#[cfg(feature = "cuda")]
fn en_prepare() {
    stonefish_ai::en_train::prepare(
        "data/en/train.txt",
        "data/gb/books.txt",
        "data/en/valid.txt",
        16_384,
        "data/en/tiny",
    );
}

#[cfg(not(feature = "cuda"))]
fn en_prepare() {
    // BPE learning and tokenization need no GPU; they work in a CUDA-less build.
    use std::io::Read;
    let mut f = std::fs::File::open("data/en/train.txt").expect("could not open the corpus");
    let mut buf = vec![0u8; 100 << 20];
    let got = f.read(&mut buf).expect("okuma");
    buf.truncate(got);
    let sample = String::from_utf8_lossy(&buf).into_owned();
    let bpe = stonefish_ai::bpe::train_bpe(&sample, 16_384, 300_000);
    println!("BPE: vocab={} ({} merge)", bpe.vocab(), bpe.merges.len());
    let n = stonefish_ai::corpus::tokenize_file_to_cache(
        "data/en/train.txt",
        "data/en/tiny.train.bin",
        &bpe,
        64 << 20,
    )
    .expect("tokenize");
    println!("train token: {n}");
}

#[cfg(feature = "cuda")]
fn en_train(args: &[String]) {
    let mut a = stonefish_ai::en_train::EnTrainArgs::default();
    if let Some(t) = args.first().and_then(|s| s.parse::<u64>().ok()) {
        a.total_tokens = t;
    }
    if let Some(d) = args.get(1).and_then(|s| s.parse::<usize>().ok()) {
        a.d_model = d;
        a.n_layers = if d >= 768 { 16 } else { 12 };
        a.n_heads = (d / 128).max(4);
    }
    stonefish_ai::en_train::train(&a);
}

#[cfg(not(feature = "cuda"))]
fn en_train(_args: &[String]) {
    println!("❌ en-train requires CUDA (built with --no-default-features).");
}

#[cfg(feature = "cuda")]
fn bench(args: &[String]) {
    // Simple --key value parsing
    let get = |key: &str, def: usize| -> usize {
        args.iter()
            .position(|a| a == key)
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(def)
    };
    let d = get("--d", 1024);
    let layers = get("--layers", 16);
    let heads = get("--heads", 8);
    let ctx = get("--ctx", 512);
    let batch = get("--batch", 4);
    let accum = get("--accum", 16);
    let steps = get("--steps", 6);
    let vocab = get("--vocab", 16384);

    let cfg = transformer::GptConfig {
        vocab_size: vocab,
        max_seq_len: ctx,
        d_model: d,
        n_layers: layers,
        n_heads: heads,
    };
    let (free0, total) = stonefish_ai::sys::dev::mem_info();
    let mut model = stonefish_ai::gpu_train::GpuGpt::new_random(cfg, batch, ctx, 0x1234_5678);
    model.dropout_p = 0.0;
    let n_params: usize = model.params_mut().iter().map(|p| p.n()).sum();
    let tps = batch * accum * ctx;

    println!(
        "StonefishAI {} | {} | zero dependencies",
        stonefish_ai::VERSION,
        stonefish_ai::sys::dev::device_info().0
    );
    println!(
        "model: d={d} layers={layers} heads={heads} ctx={ctx} vocab={vocab} -> {:.1}M parameters",
        n_params as f64 / 1e6
    );
    println!("batch: {batch} x accum {accum} = {tps} tokens/step");

    let mut x = vec![0i32; batch * ctx];
    let mut y = vec![0i32; batch * ctx];
    for i in 0..x.len() {
        x[i] = (i % (vocab - 1)) as i32;
        y[i] = ((i + 1) % (vocab - 1)) as i32;
    }

    // warm-up
    for _ in 0..2 {
        for _ in 0..accum {
            let _ = model.forward_backward(&x, &y);
        }
        model.step(4e-4, 1.0);
    }
    stonefish_ai::sys::dev::sync();

    let t0 = std::time::Instant::now();
    let mut last_loss = 0.0f32;
    for _ in 0..steps {
        for _ in 0..accum {
            last_loss = model.forward_backward(&x, &y);
        }
        model.step(4e-4, 1.0);
    }
    stonefish_ai::sys::dev::sync();
    let dt = t0.elapsed().as_secs_f64();

    let (free1, _) = stonefish_ai::sys::dev::mem_info();
    let used = (free0.saturating_sub(free1)) as f64 / 1e9;
    println!("\nstep time: {:.0} ms", dt / steps as f64 * 1000.0);
    println!(
        "speed      : {:.0} tokens/s",
        steps as f64 * tps as f64 / dt
    );
    println!(
        "peak VRAM  : ~{used:.1} GB (total {:.1} GB)",
        total as f64 / 1e9
    );
    println!("final loss : {last_loss:.4}");
}

#[cfg(not(feature = "cuda"))]
fn bench(_args: &[String]) {
    println!("❌ bench CUDA gerektirir.");
}

/// Works with or without CUDA: the trainer picks the device-resident path when
/// a GPU is present and the autograd path otherwise.
fn demo(args: &[String]) {
    let secs: u64 = args.first().and_then(|s| s.parse().ok()).unwrap_or(90);
    stonefish_ai::demo::run(secs);
}

#[cfg(feature = "cuda")]
fn test_ffi() {
    stonefish_ai::sys::selftest::run();
}

#[cfg(not(feature = "cuda"))]
fn test_ffi() {
    println!("❌ CUDA is off.");
}

#[cfg(feature = "cuda")]
fn en_tokenize(args: &[String]) {
    let prefix = args.first().map(|s| s.as_str()).unwrap_or("data/en/probe");
    println!("🐟 Tokenize for evaluation: {prefix}\n");
    match stonefish_ai::probe::tokenize(prefix) {
        Ok(n) if n > 0 => println!("\n✅ {n} tokens ready"),
        Ok(_) => println!("\n⚠️  the file produced no tokens"),
        Err(e) => println!("❌ {e}"),
    }
}

#[cfg(not(feature = "cuda"))]
fn en_tokenize(_args: &[String]) {
    println!("❌ en-tokenize needs the CUDA build (it feeds en-eval).");
}

#[cfg(feature = "cuda")]
fn en_eval(args: &[String]) {
    let max_batches: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(500);
    println!("🐟 English model evaluation\n");
    let (loss, ppl, bpc, tokens) =
        stonefish_ai::en_train::eval("data/en/tiny", "stonefish_en.state", max_batches);
    println!("\n--- RESULT ---");
    println!("Sampled        : {tokens} tokens");
    println!("val loss       : {loss:.4} nat/token");
    println!("perplexity     : {ppl:.2}");
    println!("BPC            : {bpc:.4} bit/karakter");
    println!(
        "\nReferans: rastgele tahmin = ln({}) = {:.2}",
        16384,
        (16384f64).ln()
    );
}

#[cfg(not(feature = "cuda"))]
fn en_eval(_args: &[String]) {
    println!("❌ en-eval CUDA gerektirir.");
}

#[cfg(feature = "cuda")]
fn en_gen(args: &[String]) {
    let prompt = args
        .first()
        .cloned()
        .unwrap_or_else(|| "Once upon a time".into());
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let temp: f32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.8);
    stonefish_ai::en_train::generate("data/en/tiny", "stonefish_en.state", &prompt, n, temp, 40);
}

/// CPU fallback for `en-gen`: loads the shipped SFEN001 checkpoint into the CPU
/// model and generates there. This is what lets someone without a GPU run the
/// released model at all - slower (autograd path, one token at a time) but the
/// same weights and the same sampler.
#[cfg(not(feature = "cuda"))]
fn en_gen(args: &[String]) {
    let prompt = args
        .first()
        .cloned()
        .unwrap_or_else(|| "Once upon a time".into());
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(50);
    let temp: f32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.8);
    // prefer the final checkpoint once training has finished
    let state = if std::path::Path::new("stonefish_en.final.state").exists() {
        "stonefish_en.final.state"
    } else {
        "stonefish_en.state"
    };

    let (cfg, step) =
        stonefish_ai::checkpoint::read_sfen_cfg(state).expect("could not read the checkpoint");
    println!(
        "Model  : d={}, {} layers, {} heads, ctx={}, vocab={} (step {step})",
        cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.max_seq_len, cfg.vocab_size
    );
    println!("Device : CPU (autograd path - expect a few tokens per second)");

    let bpe = stonefish_ai::bpe::load_merges("data/en/tiny.bpe").expect("could not load the BPE");
    let mut seed = 0x5EED_1234u64;
    let mut model = stonefish_ai::transformer::GPT::new(cfg, &mut seed);
    stonefish_ai::checkpoint::load_sfen_into(state, &mut model)
        .expect("could not load the weights");

    let mut ids: Vec<usize> = bpe.encode(&prompt).iter().map(|&x| x as usize).collect();
    if ids.is_empty() {
        ids.push(0);
    }
    print!("{prompt}");
    use std::io::Write;
    let _ = std::io::stdout().flush();

    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        let start = ids.len().saturating_sub(cfg.max_seq_len);
        let window: Vec<usize> = ids[start..].to_vec();
        let logits = model.forward(&window);
        let d = logits.data.borrow();
        let row = d.rows - 1;
        let cols = d.cols;

        let next = if temp <= 0.0 {
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for j in 0..cols {
                let v = d.get(row, j);
                if v > bv {
                    bv = v;
                    best = j;
                }
            }
            best
        } else {
            // top-40 sampling, same shape as the GPU path
            let mut idx: Vec<(usize, f32)> = (0..cols).map(|j| (j, d.get(row, j))).collect();
            idx.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            idx.truncate(40.min(cols));
            let mx = idx[0].1;
            let mut probs: Vec<f32> = idx.iter().map(|&(_, v)| ((v - mx) / temp).exp()).collect();
            let sum: f32 = probs.iter().sum();
            for p in probs.iter_mut() {
                *p /= sum;
            }
            let r = (next_rng(&mut rng) >> 11) as f32 / (1u64 << 53) as f32;
            let mut acc = 0.0f32;
            let mut pick = idx[0].0;
            for (k, &(j, _)) in idx.iter().enumerate() {
                acc += probs[k];
                if r < acc {
                    pick = j;
                    break;
                }
            }
            pick
        };
        drop(d);

        ids.push(next);
        let ch = bpe.decode(&[next as u32]);
        print!("{ch}");
        let _ = std::io::stdout().flush();
    }
    let dt = t0.elapsed().as_secs_f64();
    println!();
    if n > 0 {
        println!(
            "\n  ({n} tokens, {dt:.1} s, {:.2} tok/s - CPU)",
            n as f64 / dt
        );
    }
}

#[cfg(not(feature = "cuda"))]
fn next_rng(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

// ============================================================
// Helpers
// ============================================================

/// Speed of the GPU-native (device-resident) trainer in the real config.
#[cfg(feature = "cuda")]
fn test_gpu_native(args: &[String]) {
    let steps: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(10);
    let batch: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(8);

    println!(
        "🐟 GPU-native trainer (device-resident) - d=128, 4 layers, 4 heads, S=128, batch={batch}\n"
    );
    if !gpu::HAS_CUDA {
        println!("❌ CUDA is off.");
        return;
    }

    let text = data::load_text(TEXT_PATH);
    let tok = Tokenizer::from_text(&text);
    let all_ids = tok.encode(&text);
    let seq_len = 128;
    let seqs = data::make_sequences(&all_ids, seq_len, seq_len);

    let mut seed: u64 = 0xDEADBEEF;
    let cfg = GptConfig {
        vocab_size: tok.vocab_size,
        max_seq_len: seq_len,
        d_model: 128,
        n_layers: 4,
        n_heads: 4,
    };
    let cpu_model = GPT::new(cfg, &mut seed);

    let mut g = stonefish_ai::gpu_train::GpuGpt::from_cpu(&cpu_model, batch, seq_len);
    println!(
        "Model: {} parameters, {} tensors on the device",
        cpu_model.n_params(),
        g.params_mut().len()
    );

    let batch_ids = |i: usize| -> (Vec<i32>, Vec<i32>) {
        let mut ids = Vec::with_capacity(batch * seq_len);
        let mut tgt = Vec::with_capacity(batch * seq_len);
        for k in 0..batch {
            let (a, b) = &seqs[(i * batch + k) % seqs.len()];
            ids.extend(a.iter().map(|&x| x as i32));
            tgt.extend(b.iter().map(|&x| x as i32));
        }
        (ids, tgt)
    };

    // warm-up
    for i in 0..3 {
        let (ids, tgt) = batch_ids(i);
        g.forward_backward(&ids, &tgt);
        g.step(1e-3, 1.0);
    }

    let t0 = std::time::Instant::now();
    let mut total_loss = 0.0f32;
    for i in 0..steps {
        let (ids, tgt) = batch_ids(i + 3);
        let l = g.forward_backward(&ids, &tgt);
        g.step(1e-3, 1.0);
        total_loss += l;
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let per_step = elapsed / steps as f64;
    let tokens = batch * seq_len;

    println!(
        "\n{steps} steps -> {:.2} ms/step  ({:.0} tok/s)",
        per_step * 1e3,
        tokens as f64 / per_step
    );
    println!("Average loss: {:.4}", total_loss / steps as f32);
    println!("\nComparison (CPU+cuBLAS path): ~120 ms/step, ~8,500 tok/s");
    println!("Speed-up: {:.1}x", 0.120 / per_step);
}

#[cfg(not(feature = "cuda"))]
fn test_gpu_native(_args: &[String]) {
    println!("❌ CUDA is off in this build (--no-default-features).");
}

/// The time budget of one training step in the real model configuration.
/// To show what a "full GPU" investment buys with measurement, not guesses.
fn test_gpt_profile(args: &[String]) {
    let steps: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(8);
    let batch: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(8);

    println!(
        "🐟 Step profile on the real model (d=128, 4 layers, 4 heads, S=128, batch={batch})\n"
    );

    let text = data::load_text(TEXT_PATH);
    let tok = Tokenizer::from_text(&text);
    let all_ids = tok.encode(&text);
    let seq_len = 128;
    let seqs = data::make_sequences(&all_ids, seq_len, seq_len);

    let mut seed: u64 = 0xDEADBEEF;
    let config = GptConfig {
        vocab_size: tok.vocab_size,
        max_seq_len: seq_len,
        d_model: 128,
        n_layers: 4,
        n_heads: 4,
    };
    let model = GPT::new(config, &mut seed);
    let mut opt = Adam::new(model.parameters(), 1e-3);

    let make_batch = |i: usize| -> (Vec<usize>, Vec<usize>) {
        let mut ids = Vec::with_capacity(batch * seq_len);
        let mut tgt = Vec::with_capacity(batch * seq_len);
        for k in 0..batch {
            let (a, b) = &seqs[(i * batch + k) % seqs.len()];
            ids.extend_from_slice(a);
            tgt.extend_from_slice(b);
        }
        (ids, tgt)
    };

    // Warm-up (CUDA context, cuBLAS handle, buffer pool)
    for i in 0..3 {
        let (ids, tgt) = make_batch(i);
        let loss = model
            .forward_batch(&ids, batch, seq_len)
            .softmax_cross_entropy_ids(&tgt);
        opt.zero_grad();
        loss.backward();
        opt.step();
    }

    profile::reset();
    profile::enable();

    let t0 = std::time::Instant::now();
    for i in 0..steps {
        let (ids, tgt) = make_batch(i + 3);
        let loss = model
            .forward_batch(&ids, batch, seq_len)
            .softmax_cross_entropy_ids(&tgt);
        opt.zero_grad();
        loss.backward();
        opt.step();
    }
    let total = t0.elapsed();
    profile::disable();

    let (calls, mm_ns, tr_ns, af_ns, ab_ns) = profile::snapshot();
    let s = profile::snapshot_full();
    let total_ns = total.as_nanos() as u64;
    let step_ns = total_ns / steps as u64;
    let pct = |v: u64| 100.0 * v as f64 / total_ns as f64;
    let ms = |v: u64| v as f64 / 1e6 / steps as f64;

    let matmul_compute = mm_ns.saturating_sub(tr_ns);
    let attn = af_ns + ab_ns;
    let compute = matmul_compute + attn;
    let diger = total_ns.saturating_sub(mm_ns + attn);

    println!(
        "{steps} steps -> {:.2} ms/step ({:.0} tok/s, GPU={})",
        step_ns as f64 / 1e6,
        (batch * seq_len) as f64 / (step_ns as f64 / 1e9),
        gpu::HAS_CUDA
    );
    println!("  matmul calls: {} / step\n", calls / steps as u64);
    println!(
        "  {:<38} {:8.2} ms/step  %{:5.1}",
        "matmul (transfer dahil)",
        ms(mm_ns),
        pct(mm_ns)
    );
    println!(
        "    {:<36} {:8.2} ms/step  %{:5.1}",
        "├─ transfer (CPU↔GPU)",
        ms(tr_ns),
        pct(tr_ns)
    );
    println!(
        "    {:<36} {:8.2} ms/step  %{:5.1}",
        "+- real GEMM",
        ms(matmul_compute),
        pct(matmul_compute)
    );
    println!(
        "  {:<38} {:8.2} ms/step  %{:5.1}",
        "attention (fwd+bwd, CPU)",
        ms(attn),
        pct(attn)
    );
    println!(
        "  {:<38} {:8.2} ms/step  %{:5.1}",
        "  ├─ forward",
        ms(af_ns),
        pct(af_ns)
    );
    println!(
        "  {:<38} {:8.2} ms/step  %{:5.1}",
        "  └─ backward",
        ms(ab_ns),
        pct(ab_ns)
    );
    println!(
        "  {:<38} {:8.2} ms/step  %{:5.1}",
        "gelu (fwd+bwd)",
        ms(s.gelu),
        pct(s.gelu)
    );
    println!(
        "  {:<38} {:8.2} ms/step  %{:5.1}",
        "layer_norm (fwd+bwd)",
        ms(s.ln),
        pct(s.ln)
    );
    println!(
        "  {:<38} {:8.2} ms/step  %{:5.1}",
        "cross-entropy (fwd+bwd)",
        ms(s.ce),
        pct(s.ce)
    );
    println!(
        "  {:<38} {:8.2} ms/step  %{:5.1}",
        "adam.step",
        ms(s.adam),
        pct(s.adam)
    );
    println!(
        "  {:<38} {:8.2} ms/step  %{:5.1}",
        "float_zero_grad (parameter reset)",
        ms(s.zero_grad),
        pct(s.zero_grad)
    );
    println!(
        "  {:<38} {:8.2} ms/step  %{:5.1}",
        "intermediate tensor allocation (from_op)",
        ms(s.alloc),
        pct(s.alloc)
    );
    println!(
        "  {:<38} {:8.2} ms/step  %{:5.1}",
        "other (softmax, packing, ...)",
        ms(diger),
        pct(diger)
    );
    println!(
        "\n  Compute (GEMM+attention) {:.1}% - transfer {:.1}% - other {:.1}%",
        pct(compute),
        pct(tr_ns),
        pct(diger)
    );
    let cpu_share = pct(attn) + pct(diger);
    println!(
        "\n  Interpretation: {:.0}% of the step runs on the CPU (attention + elementwise + Adam).\n\
         \x20 Keeping only the GEMMs on the GPU (eliminating transfers) buys at most {:.0}%.\n\
         \x20 The real speed-up comes from either vectorizing the CPU work or moving it to CUDA kernels.",
        cpu_share,
        pct(tr_ns)
    );
}

fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

fn rand_f32(seed: &mut u64) -> f32 {
    (xorshift(seed) >> 11) as f32 / (1u64 << 53) as f32
}
