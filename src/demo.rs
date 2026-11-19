// src/demo.rs
// One-command demo: train a GPT from scratch on a small corpus, then generate.
//
// Goal: someone who clones the project sees something working in 90 seconds,
// with or without a GPU. The trainer is picked at runtime by
// `trainer::Trainer` - the device-resident path when CUDA is available, the
// verified autograd path otherwise - so the loop below is written once.

use crate::data;
use crate::tokenizer::Tokenizer;
use crate::trainer::Trainer;
use crate::transformer::{GptConfig, GPT};

/// Steps run before the time budget is extrapolated. Five is enough to smooth
/// out the first-step cost (allocation, first kernel load) and cheap enough that
/// it does not eat the budget.
const PROBE_STEPS: usize = 5;

pub fn run(seconds: u64) {
    println!("🐟 StonefishAI - one-command demo\n");
    println!("Training from scratch: no ML libraries, no dependencies.\n");

    let path = "examples/data/tiny_shakespeare.txt";
    if !std::path::Path::new(path).exists() {
        println!("❌ Demo corpus not found: {path}");
        println!("   (run from the repository root: cargo run --release -- demo)");
        return;
    }
    let full = data::load_text(path);
    // The CPU path trains on a small slice on purpose: a few thousand steps on
    // 40 KB is enough to produce Shakespeare-shaped text, which is what a demo
    // has to do inside its time budget. The GPU path gets the whole file.
    let has_gpu = crate::gpu::HAS_CUDA;
    let text: String = if has_gpu {
        full
    } else {
        full.chars().take(12_000).collect()
    };
    let tok = Tokenizer::from_text(&text);
    let ids: Vec<usize> = tok.encode(&text);
    println!(
        "Corpus : {} characters, vocab {}{}",
        text.chars().count(),
        tok.vocab_size,
        if has_gpu { "" } else { " (slice, CPU demo)" }
    );

    // The CPU path gets a smaller model: the whole point of the demo is that a
    // clone produces readable text inside the time budget, and on the CPU the
    // step cost is what decides how many steps fit.
    let (d_model, n_layers, n_heads, ctx_len) = if has_gpu {
        (192usize, 4usize, 4usize, 128usize)
    } else {
        (96, 2, 4, 64)
    };
    let cfg = GptConfig {
        vocab_size: tok.vocab_size,
        max_seq_len: ctx_len,
        d_model,
        n_layers,
        n_heads,
    };
    let batch = 16usize;
    let seq = cfg.max_seq_len;
    let mut seed = 0xDEAD_BEEFu64;
    let mut model = GPT::new(cfg, &mut seed);
    println!(
        "Model  : d_model={}, {} layers, {} heads, ctx={} -> {} parameters",
        cfg.d_model,
        cfg.n_layers,
        cfg.n_heads,
        cfg.max_seq_len,
        model.n_params()
    );

    let mut trainer = Trainer::new(cfg, 0xDEAD_BEEF, batch, seq);
    let on_gpu = trainer.is_gpu();
    println!(
        "Device : {}",
        if on_gpu {
            "GPU - device-resident trainer"
        } else {
            "CPU - autograd trainer (no CUDA in this build or on this machine)"
        }
    );

    let tokens_per_step = batch * seq;
    println!(
        "Plan   : {} tokens/step, {} s budget (the step count is measured, not guessed)\n",
        tokens_per_step, seconds
    );

    let mut ids_in = vec![0usize; batch * seq];
    let mut targets = vec![0usize; batch * seq];
    let mut rng = 0x1234_5678_9ABC_DEF0u64;
    let t0 = std::time::Instant::now();
    let mut loss_avg = 0.0f64;
    let mut loss_n = 0usize;
    let mut last_report = std::time::Instant::now();
    let mut total_steps: usize = PROBE_STEPS;
    let mut step = 0usize;

    while step < total_steps {
        // random windows over the corpus
        for b in 0..batch {
            let max_start = ids.len().saturating_sub(seq + 1).max(1);
            let start = (xorshift(&mut rng) as usize) % max_start;
            let off = b * seq;
            for j in 0..seq {
                ids_in[off + j] = ids[start + j];
                targets[off + j] = ids[start + j + 1];
            }
        }
        if step + 1 == PROBE_STEPS {
            let rate = ((step + 1) * tokens_per_step) as f64 / t0.elapsed().as_secs_f64();
            let left = (seconds as f64 - t0.elapsed().as_secs_f64()).max(1.0);
            total_steps = PROBE_STEPS + (left * rate / tokens_per_step as f64) as usize;
            println!(
                "  measured {:.0} tok/s -> {} steps fit in the budget\n",
                rate, total_steps
            );
            last_report = std::time::Instant::now();
        }
        let warmup = (total_steps / 20).max(10);
        let lr = crate::trainer::lr_at(step, total_steps, 3e-3, warmup, 0.1);
        let (loss, _gnorm) = trainer.train_step(&ids_in, &targets, batch, seq, lr, 1.0);
        loss_avg += loss as f64;
        loss_n += 1;

        // hard stop: never run more than 30 % past the requested budget
        if t0.elapsed().as_secs_f64() > seconds as f64 * 1.3 {
            println!(
                "  stopped at {:.0} s (budget {} s) - the machine is slower than assumed",
                t0.elapsed().as_secs_f64(),
                seconds
            );
            break;
        }

        if last_report.elapsed().as_secs_f32() > (seconds as f32 / 8.0).max(1.0) {
            last_report = std::time::Instant::now();
            println!(
                "  step {:5}/{}  loss {:.4}  lr {:.2e}  {:6.0} tok/s  [{:.0} s]",
                step + 1,
                total_steps,
                loss_avg / loss_n as f64,
                lr,
                ((step + 1) * tokens_per_step) as f64 / t0.elapsed().as_secs_f64(),
                t0.elapsed().as_secs_f64()
            );
            loss_avg = 0.0;
            loss_n = 0;
        }
        step += 1;
    }
    let secs = t0.elapsed().as_secs_f64();
    println!(
        "\nTraining finished: {:.0} s, {:.0} tok/s\n",
        secs,
        (total_steps * tokens_per_step) as f64 / secs
    );

    // ---- generation: greedy decode from the CPU copy ----
    trainer.sync_to_cpu(&mut model);
    for prompt in ["ROMEO:", "To be, or not", "JULIET:"] {
        let mut ctx: Vec<usize> = tok.encode(prompt);
        print!("{prompt}");
        for _ in 0..80 {
            let start = ctx.len().saturating_sub(seq);
            let window: Vec<usize> = ctx[start..].to_vec();
            let logits = model.forward(&window);
            let d = logits.data.borrow();
            let (rows, cols) = (d.rows, d.cols);
            let row = rows - 1;
            let mut best = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for j in 0..cols {
                let v = d.get(row, j);
                if v > best_val {
                    best_val = v;
                    best = j;
                }
            }
            drop(d);
            ctx.push(best);
            print!("{}", tok.decode(&[best]));
        }
        println!("\n");
    }

    println!(
        "Done. Nothing here came from a library: every op, kernel and optimizer is in this repo."
    );
}

fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}
