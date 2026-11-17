// src/demo.rs
// Tek komutluk demo: küçük bir korpustan sıfırdan GPT eğit, metin üret.
//
// Amaç: projeyi klonlayan birinin 90 saniyede çalışan bir şey görmesi.
// GPU varsa device-resident eğitici kullanılır (yoksa CPU yolu).

#![cfg(feature = "cuda")]

use crate::data;
use crate::gpu_train::GpuGpt;
use crate::tokenizer::Tokenizer;
use crate::transformer::{GptConfig, GPT};

pub fn run(seconds: u64) {
    println!("🐟 StonefishAI — tek komut demo\n");
    println!("Sıfırdan eğitim: harici ML kütüphanesi yok, bağımlılık yok.\n");

    // ---- 1) Korpus + tokenizer ----
    let path = "examples/data/tiny_shakespeare.txt";
    if !std::path::Path::new(path).exists() {
        println!("❌ Demo korpusu bulunamadı: {path}");
        println!("   (depo kökünden çalıştırın: cargo run --release -- demo)");
        return;
    }
    let text = data::load_text(path);
    let tok = Tokenizer::from_text(&text);
    let ids: Vec<usize> = tok.encode(&text);
    println!(
        "Korpus    : {} karakter, vocab {}",
        text.chars().count(),
        tok.vocab_size
    );

    // ---- 2) Model ----
    let cfg = GptConfig {
        vocab_size: tok.vocab_size,
        max_seq_len: 128,
        d_model: 192,
        n_layers: 4,
        n_heads: 4,
    };
    let batch = 16usize;
    let seq = cfg.max_seq_len;
    let mut seed = 0xDEAD_BEEFu64;
    let model = GPT::new(cfg, &mut seed);
    println!(
        "Model     : d_model={}, {} katman, {} head, ctx={} → {} parametre",
        cfg.d_model,
        cfg.n_layers,
        cfg.n_heads,
        cfg.max_seq_len,
        model.n_params()
    );

    // ---- 3) Eğitim (GPU-native) ----
    let mut g = GpuGpt::new_random(cfg, batch, seq, 0xDEAD_BEEF);
    let total_steps = ((seconds * 350_000) / (batch * seq) as u64) as usize; // ~350k tok/s (ölçüldü)
    let warmup = (total_steps / 20).max(10);
    println!(
        "Eğitim    : {} adım × {} token = {:.1} M token, hedef {} s\n",
        total_steps,
        batch * seq,
        (total_steps * batch * seq) as f64 / 1e6,
        seconds
    );

    let mut x_i = vec![0i32; batch * seq];
    let mut y_i = vec![0i32; batch * seq];
    let mut rng = 0x1234_5678_9ABC_DEF0u64;
    let t0 = std::time::Instant::now();
    let mut loss_avg = 0.0f64;
    let mut loss_n = 0usize;
    let mut last_report = std::time::Instant::now();

    for step in 0..total_steps {
        // rastgele pencereler
        for b in 0..batch {
            let max_start = ids.len().saturating_sub(seq + 1).max(1);
            let start = (xorshift(&mut rng) as usize) % max_start;
            let off = b * seq;
            for j in 0..seq {
                x_i[off + j] = ids[start + j] as i32;
                y_i[off + j] = ids[start + j + 1] as i32;
            }
        }
        let lr = crate::en_train::lr_at(step, total_steps, 3e-3, warmup, 0.1);
        let loss = g.forward_backward(&x_i, &y_i);
        g.step(lr, 1.0);
        loss_avg += loss as f64;
        loss_n += 1;

        if last_report.elapsed().as_secs_f32() > (seconds as f32 / 8.0).max(1.0) {
            last_report = std::time::Instant::now();
            println!(
                "  adım {:5}/{}  loss {:.4}  lr {:.2e}  {:6.0} tok/s  [{:.0} s]",
                step + 1,
                total_steps,
                loss_avg / loss_n as f64,
                lr,
                ((step + 1) * batch * seq) as f64 / t0.elapsed().as_secs_f64(),
                t0.elapsed().as_secs_f64()
            );
            loss_avg = 0.0;
            loss_n = 0;
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    println!(
        "\nEğitim bitti: {:.0} s, {:.0} tok/s\n",
        secs,
        (total_steps * batch * seq) as f64 / secs
    );

    // ---- 4) Üretim ----
    let mut cpu = model;
    g.copy_to_cpu(&mut cpu);
    for prompt in ["ROMEO:", "To be, or not", "JULIET:"] {
        let mut ctx: Vec<usize> = tok.encode(prompt);
        print!("{prompt}");
        for _ in 0..80 {
            let start = ctx.len().saturating_sub(seq);
            let window: Vec<usize> = ctx[start..].to_vec();
            let logits = cpu.forward(&window);
            let d = logits.data.borrow();
            let (rows, cols) = (d.rows, d.cols);
            let row = rows - 1;
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for j in 0..cols {
                let v = d.get(row, j);
                if v > bv {
                    bv = v;
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
        "Bu metin {} saniyelik eğitimden geliyor — motor çalışıyor.",
        seconds
    );
    println!("Gerçek model: `stonefish_ai en-gen` (236M parametre, 12 saat eğitim).");
}

fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}
