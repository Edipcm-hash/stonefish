// src/en_train.rs
// English training pipeline (BPE tokenizer + disk cache).
//
// Separate from the Turkish character-level path (`gpt-train`) because:
//  * the data source is STREAMED from a disk cache (not held in RAM),
//  * the tokenizer is BPE (a separate `SFBPE` file plus a vocab size in the
//    vocab_size),
//  * the model is initialized on the device without a CPU twin,
//  * gradient accumulation and periodic resume are supported.

#![cfg(feature = "cuda")]

use crate::bpe::{train_bpe, Bpe};
use crate::corpus::{tokenize_append, tokenize_file_to_cache, TokenDataset};
use crate::gpu_train::GpuGpt;
use crate::transformer::GptConfig;
use std::io::{Read, Write};

pub const MERGES_MAGIC: &[u8; 8] = b"SFBPE001";

// ============================================================
// Tokenizer file
// ============================================================

pub fn save_bpe(path: &str, bpe: &Bpe) -> std::io::Result<()> {
    let f = std::fs::File::create(path)?;
    let mut w = std::io::BufWriter::new(f);
    w.write_all(MERGES_MAGIC)?;
    w.write_all(&1u32.to_le_bytes())?;
    w.write_all(&(bpe.merges.len() as u32).to_le_bytes())?;
    for (a, b) in &bpe.merges {
        w.write_all(&a.to_le_bytes())?;
        w.write_all(&b.to_le_bytes())?;
    }
    w.flush()
}

pub fn load_bpe(path: &str) -> std::io::Result<Bpe> {
    let f = std::fs::File::open(path)?;
    let mut r = std::io::BufReader::new(f);
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != MERGES_MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "BPE file not recognized",
        ));
    }
    let mut u4 = [0u8; 4];
    r.read_exact(&mut u4)?;
    if u32::from_le_bytes(u4) != 1 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "unsupported BPE version",
        ));
    }
    r.read_exact(&mut u4)?;
    let n = u32::from_le_bytes(u4) as usize;
    let mut merges = Vec::with_capacity(n);
    for _ in 0..n {
        r.read_exact(&mut u4)?;
        let a = u32::from_le_bytes(u4);
        r.read_exact(&mut u4)?;
        let b = u32::from_le_bytes(u4);
        merges.push((a, b));
    }
    Ok(Bpe::from_merges(merges))
}

// ============================================================
// prepare: learn BPE + build the cache
// ============================================================

/// Mixed corpus preparation: TinyStories + Gutenberg.
///
/// BPE is learned on a sample from both corpora (Gutenberg's vocabulary is
/// much larger; a BPE trained on TinyStories alone encodes books badly).
/// Both corpora are then written into ONE cache file: TinyStories first,
/// then Gutenberg. Since training samples random windows, this ordering
/// yields a balanced mixture.
pub fn prepare(
    stories_txt: &str,
    books_txt: &str,
    valid_txt: &str,
    vocab: usize,
    out_prefix: &str,
) {
    let t0 = std::time::Instant::now();
    println!("🐟 English data preparation (mixed corpus)\n");
    println!("TinyStories : {stories_txt}");
    println!("Gutenberg   : {books_txt}");
    println!("Valid       : {valid_txt}");
    println!("Target      : vocab={vocab}, output={out_prefix}.{{bpe,train.bin,valid.bin}}\n");

    // --- 1) BPE sample: 100 MB from each corpus ---
    let mut sample = String::new();
    for path in [stories_txt, books_txt] {
        match std::fs::File::open(path) {
            Ok(mut f) => {
                let mut buf = vec![0u8; 100 << 20];
                let got = f.read(&mut buf).unwrap_or(0);
                buf.truncate(got);
                sample.push_str(&String::from_utf8_lossy(&buf));
                println!("BPE sample: {} MB <- {path}", got >> 20);
            }
            Err(e) => println!("⚠️  could not open {path} ({e}) - skipping"),
        }
    }
    if sample.is_empty() {
        println!("❌ No corpus found at all.");
        return;
    }
    let bpe = train_bpe(&sample, vocab, 300_000);
    println!(
        "BPE ready: vocab={} ({} merges), {:.1} s",
        bpe.vocab(),
        bpe.merges.len(),
        t0.elapsed().as_secs_f64()
    );
    let bpe_path = format!("{out_prefix}.bpe");
    save_bpe(&bpe_path, &bpe).expect("could not write BPE");
    drop(sample);

    // --- 2) Tokenize: TinyStories (truncate) + Gutenberg (append) ---
    let tr_bin = format!("{out_prefix}.train.bin");
    let mut total: u64 = 0;
    for (i, path) in [stories_txt, books_txt].iter().enumerate() {
        if !std::path::Path::new(path).exists() {
            continue;
        }
        let t = std::time::Instant::now();
        let r = if i == 0 {
            tokenize_file_to_cache(path, &tr_bin, &bpe, 64 << 20)
        } else {
            tokenize_append(path, &tr_bin, &bpe, 64 << 20)
        };
        match r {
            Ok(n) => {
                total += n;
                println!(
                    "{:12}: {n} tokens, {:.1} s (cumulative {})",
                    if i == 0 { "TinyStories" } else { "Gutenberg" },
                    t.elapsed().as_secs_f64(),
                    total
                );
            }
            Err(e) => println!("⚠️  {path} tokenize edilemedi: {e}"),
        }
    }

    // --- 3) Validation (stories only) ---
    let va_bin = format!("{out_prefix}.valid.bin");
    let t = std::time::Instant::now();
    match tokenize_file_to_cache(valid_txt, &va_bin, &bpe, 64 << 20) {
        Ok(n) => println!(
            "valid       : {n} token, {:.1} s",
            t.elapsed().as_secs_f64()
        ),
        Err(e) => println!("⚠️  valid tokenize edilemedi: {e}"),
    }

    let corpus_bytes: u64 = [stories_txt, books_txt]
        .iter()
        .filter_map(|p| std::fs::metadata(p).ok().map(|m| m.len()))
        .sum();
    println!(
        "\nTotal: {} tokens, compression {:.2} chars/token",
        total,
        corpus_bytes as f64 / total.max(1) as f64
    );
    println!("Elapsed: {:.1} min", t0.elapsed().as_secs_f64() / 60.0);
}
// ============================================================
// train: the 250M configuration
// ============================================================

pub struct EnTrainArgs {
    pub corpus_prefix: String,
    pub out_prefix: String,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub seq_len: usize,
    pub batch: usize,
    pub accum: usize,
    pub total_tokens: u64,
    pub max_lr: f32,
    pub warmup: usize,
    pub clip: f32,
    pub ckpt_every: usize,
    pub log_every: usize,
}

impl Default for EnTrainArgs {
    fn default() -> Self {
        Self {
            corpus_prefix: "data/en/tiny".into(),
            out_prefix: "stonefish_en".into(),
            d_model: 1024,
            n_layers: 16,
            n_heads: 8,
            seq_len: 512,
            batch: 4,
            accum: 16,
            total_tokens: 2_000_000_000,
            max_lr: 4e-4,
            warmup: 1000,
            clip: 1.0,
            ckpt_every: 500,
            log_every: 20,
        }
    }
}

pub fn train(a: &EnTrainArgs) {
    let bpe = load_bpe(&format!("{}.bpe", a.corpus_prefix)).expect("could not load BPE");
    let mut data = TokenDataset::open(&format!("{}.train.bin", a.corpus_prefix))
        .expect("could not open cache");
    let vocab = bpe.vocab();
    let cfg = GptConfig {
        vocab_size: vocab,
        max_seq_len: a.seq_len,
        d_model: a.d_model,
        n_layers: a.n_layers,
        n_heads: a.n_heads,
    };
    let tokens_per_step = (a.batch * a.accum * a.seq_len) as u64;
    let total_steps = (a.total_tokens / tokens_per_step) as usize;

    println!("🐟 StonefishAI - English GPT training (250M class)\n");
    println!("Tokenizer : BPE vocab={vocab}");
    println!(
        "Corpus    : {} tokens (cache), seq_len={}",
        data.len(),
        a.seq_len
    );
    println!(
        "Model     : d_model={}, layers={}, heads={} (d_head={}), ctx={}",
        a.d_model,
        a.n_layers,
        a.n_heads,
        a.d_model / a.n_heads,
        a.seq_len
    );
    println!(
        "Batch     : {} x accum {} = {} sequences ({} tokens/step)",
        a.batch,
        a.accum,
        a.batch * a.accum,
        tokens_per_step
    );
    println!(
        "Plan      : {:.2} B tokens -> {} steps, max_lr={}, clip={}",
        a.total_tokens as f64 / 1e9,
        total_steps,
        a.max_lr,
        a.clip
    );
    println!(
        "Checkpoint: every {} steps -> {}.state\n",
        a.ckpt_every, a.out_prefix
    );

    let mut model = GpuGpt::new_random(cfg, a.batch, a.seq_len, 0xDEADBEEF);
    model.dropout_p = std::env::var("SFISH_DROP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.1);
    let n_params: usize = model.params_mut().iter().map(|p| p.n()).sum();
    println!("Parameters: {n_params} ({:.0} M)", n_params as f64 / 1e6);

    // resume
    let ckpt = format!("{}.state", a.out_prefix);
    let mut start_step = 0usize;
    if std::path::Path::new(&ckpt).exists() {
        match model.load_state(&ckpt) {
            Ok(t) => {
                start_step = t;
                println!("▶ Resume: {ckpt} (step {start_step})");
            }
            Err(e) => println!("⚠️  Could not resume ({e}), starting from scratch"),
        }
    }

    // --- Timing probe (SFISH_TIME=1) ---
    if std::env::var("SFISH_TIME").is_ok() {
        use std::time::Instant;
        let bs = a.batch;
        let mut xi = vec![0i32; bs * a.seq_len];
        let mut yi = vec![0i32; bs * a.seq_len];
        for (i, v) in xi.iter_mut().enumerate() {
            *v = (i % 1000) as i32;
        }
        for (i, v) in yi.iter_mut().enumerate() {
            *v = ((i + 1) % 1000) as i32;
        }
        model.training = false;
        let _ = model.forward_loss(&xi, &yi); // warm-up
        let t = Instant::now();
        for _ in 0..3 {
            let _ = model.forward_loss(&xi, &yi);
        }
        let fwd = t.elapsed().as_secs_f64() / 3.0;

        model.dropout_p = 0.0;
        let _ = model.forward_backward(&xi, &yi); // warm-up
        let t = Instant::now();
        for _ in 0..3 {
            let _ = model.forward_backward(&xi, &yi);
        }
        let tot = t.elapsed().as_secs_f64() / 3.0;
        println!(
            "\n[time] forward {:.3} s | forward+backward {:.3} s -> backward {:.3} s | ratio {:.1}x",
            fwd,
            tot,
            tot - fwd,
            tot / fwd
        );
        return;
    }

    let mut rng = 0x1234_5678_9ABC_DEF0u64;
    let mut x = vec![0u16; a.seq_len];
    let mut y = vec![0u16; a.seq_len];
    let mut x_i: Vec<i32> = vec![0; a.seq_len * a.batch];
    let mut y_i: Vec<i32> = vec![0; a.seq_len * a.batch];

    let run_t0 = std::time::Instant::now();
    let mut loss_acc = 0.0f64;
    let mut loss_n = 0usize;
    let mut last_log = run_t0;
    let mut tokens_done = start_step as u64 * tokens_per_step;

    for step in start_step..total_steps {
        let lr = lr_at(step, total_steps, a.max_lr, a.warmup, 0.1);
        let mut step_loss = 0.0f64;

        for micro in 0..a.accum {
            // a random position for each micro-batch
            for b in 0..a.batch {
                let max_start = data.len() - a.seq_len as u64 - 1;
                let start = (xorshift(&mut rng) % max_start) as u64;
                data.fill_pair(start, a.seq_len, &mut x, &mut y)
                    .expect("cache okuma");
                let off = b * a.seq_len;
                for k in 0..a.seq_len {
                    x_i[off + k] = x[k] as i32;
                    y_i[off + k] = y[k] as i32;
                }
            }
            let zero = micro == 0;
            let l = model.forward_backward_with(&x_i, &y_i, zero, 1.0 / a.accum as f32);
            step_loss += l as f64;
        }
        let gnorm = model.step(lr, a.clip);

        loss_acc += step_loss / a.accum as f64;
        loss_n += 1;
        tokens_done += tokens_per_step;

        if (step + 1) % a.log_every == 0 {
            let dt = last_log.elapsed().as_secs_f64();
            last_log = std::time::Instant::now();
            let tps = (a.log_every as u64 * tokens_per_step) as f64 / dt;
            println!(
                "step {:6}/{}  loss {:.4}  lr {:.2e}  gnorm {:.2}  {:.0} tok/s  {:.1}%",
                step + 1,
                total_steps,
                loss_acc / loss_n as f64,
                lr,
                gnorm,
                tps,
                100.0 * tokens_done as f64 / a.total_tokens as f64
            );
            loss_acc = 0.0;
            loss_n = 0;
        }

        if (step + 1) % a.ckpt_every == 0 {
            match model.save_state(&ckpt) {
                Ok(()) => println!("  💾 checkpoint -> {ckpt} (step {})", step + 1),
                Err(e) => println!("  ⚠️  could not write checkpoint: {e}"),
            }
        }
    }

    let _ = model.save_state(&format!("{}.final.state", a.out_prefix));
    println!(
        "\n✅ Training complete: {} steps, {:.1} min, average {:.0} tok/s",
        total_steps - start_step,
        run_t0.elapsed().as_secs_f64() / 60.0,
        (tokens_done as f64) / run_t0.elapsed().as_secs_f64()
    );
}

pub fn lr_at(step: usize, total: usize, max_lr: f32, warmup: usize, min_ratio: f32) -> f32 {
    // one implementation, in trainer.rs, so the demo can use it without CUDA
    crate::trainer::lr_at(step, total, max_lr, warmup, min_ratio)
}

// ============================================================
// eval: loss + perplexity + BPC on the validation corpus
// ============================================================

/// Evaluation. Returns (val loss in nats/token, perplexity, BPC, sampled tokens).
pub fn eval(corpus: &str, state_path: &str, max_batches: usize) -> (f64, f64, f64, u64) {
    let _ = load_bpe(&format!("{corpus}.bpe")).expect("could not load BPE");
    let mut data = TokenDataset::open(&format!("{corpus}.valid.bin")).expect("no validation cache");
    let state = state_path.to_string();

    // Read the model size from the state file (the config sits in the header)
    let cfg = read_state_cfg(&state).expect("could not read the state file");
    let batch = 8usize;
    let seq = cfg.max_seq_len;
    let mut model = GpuGpt::new_random(cfg, batch, seq, 1);
    let t = model
        .load_state(&state)
        .expect("could not load the weights");
    println!(
        "Model: d={}, layers={}, heads={}, ctx={}, vocab={} (step {t})",
        cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.max_seq_len, cfg.vocab_size
    );

    let per_batch = (batch * seq) as u64;
    let n_batches = ((data.len().saturating_sub(1)) / per_batch) as usize;
    let n_batches = n_batches.min(max_batches.max(1));

    let mut x = vec![0u16; seq];
    let mut y = vec![0u16; seq];
    let mut x_i = vec![0i32; batch * seq];
    let mut y_i = vec![0i32; batch * seq];
    let mut sum = 0.0f64;
    let t0 = std::time::Instant::now();

    for b in 0..n_batches {
        for k in 0..batch {
            let start = (b * batch + k) as u64 * seq as u64;
            data.fill_pair(start, seq, &mut x, &mut y).expect("cache");
            let off = k * seq;
            for j in 0..seq {
                x_i[off + j] = x[j] as i32;
                y_i[off + j] = y[j] as i32;
            }
        }
        sum += model.forward_loss(&x_i, &y_i) as f64;
        if (b + 1) % 50 == 0 {
            println!(
                "  {}/{} batches (average loss {:.4}, {:.0} tok/s)",
                b + 1,
                n_batches,
                sum / (b + 1) as f64,
                ((b + 1) as u64 * per_batch) as f64 / t0.elapsed().as_secs_f64()
            );
        }
    }

    let loss = sum / n_batches.max(1) as f64;
    let ppl = loss.exp();
    // BPC: bits per character. chars/token = valid.txt bytes / valid.bin tokens
    let chars_per_token = {
        // BPC needs bytes-per-token for the SAME text that was tokenized. The
        // training corpus keeps its text in data/en/valid.txt while the cache
        // is {corpus}.valid.bin, so try {corpus}.txt first and fall back to the
        // training text. A probe (en-tokenize) has its own {prefix}.txt and
        // must not inherit the training ratio, or its BPC would be meaningless.
        let own = format!("{corpus}.txt");
        let txt_path = if std::path::Path::new(&own).exists() {
            own
        } else {
            "data/en/valid.txt".to_string()
        };
        let txt = std::fs::metadata(&txt_path)
            .map(|m| m.len() as f64)
            .unwrap_or(0.0);
        let tokens = data.len() as f64;
        if tokens > 0.0 {
            txt / tokens
        } else {
            4.7
        }
    };
    let bpc = loss / (std::f64::consts::LN_2 * chars_per_token);
    (loss, ppl, bpc, n_batches as u64 * per_batch)
}

/// Read the config from a state file header (without loading the weights).
fn read_state_cfg(path: &str) -> std::io::Result<GptConfig> {
    let mut r = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != b"SFEN001\0" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "state file not recognized",
        ));
    }
    let mut u4 = [0u8; 4];
    r.read_exact(&mut u4)?; // version
    let mut c = [0usize; 5];
    for v in c.iter_mut() {
        r.read_exact(&mut u4)?;
        *v = u32::from_le_bytes(u4) as usize;
    }
    Ok(GptConfig {
        vocab_size: c[0],
        max_seq_len: c[1],
        d_model: c[2],
        n_layers: c[3],
        n_heads: c[4],
    })
}

// ============================================================
// generate: BPE + sliding-window generation
// ============================================================

pub fn generate(
    corpus: &str,
    state_path: &str,
    prompt: &str,
    n_tokens: usize,
    temperature: f32,
    top_k: usize,
) {
    let bpe = load_bpe(&format!("{corpus}.bpe")).expect("could not load BPE");
    let state = state_path.to_string();
    let cfg = read_state_cfg(&state).expect("could not read the state");
    let seq = cfg.max_seq_len;
    let mut model = GpuGpt::new_random(cfg, 1, seq, 1);
    model
        .load_state(&state)
        .expect("could not load the weights");

    let mut ids: Vec<u32> = bpe.encode(prompt);
    if ids.is_empty() {
        ids = bpe.encode(" ");
    }

    // KV-cache: instead of recomputing the whole context for every token, K/V
    // are stored. Generation therefore does O(n) work instead of O(n^2).
    let mut ctx = crate::kv_cache::CachedModel::new(&mut model, seq);
    let t0 = std::time::Instant::now();

    // --- process the prompt (fill the cache) ---
    let mut last_logits = Vec::new();
    for (pos, &t) in ids.iter().enumerate() {
        if pos >= seq {
            break;
        }
        last_logits = ctx.step(t, pos);
    }

    print!("{prompt}");
    use std::io::Write;
    let _ = std::io::stdout().flush();

    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    let mut pos = ids.len().min(seq);
    let t_gen = std::time::Instant::now();

    for _ in 0..n_tokens {
        let next = if temperature <= 0.0 {
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, &v) in last_logits.iter().enumerate() {
                if v > bv {
                    bv = v;
                    best = i;
                }
            }
            best as u32
        } else {
            let mut idx: Vec<usize> = (0..last_logits.len()).collect();
            let k = top_k.max(1).min(last_logits.len());
            idx.sort_unstable_by(|&a, &b| last_logits[b].partial_cmp(&last_logits[a]).unwrap());
            idx.truncate(k);
            let mx = last_logits[idx[0]];
            let mut probs: Vec<f32> = idx
                .iter()
                .map(|&i| ((last_logits[i] - mx) / temperature).exp())
                .collect();
            let sum: f32 = probs.iter().sum();
            for p in probs.iter_mut() {
                *p /= sum;
            }
            let r = (xorshift(&mut rng) >> 11) as f32 / (1u64 << 53) as f32;
            let mut acc = 0.0f32;
            let mut pick = idx[idx.len() - 1];
            for (j, &i) in idx.iter().enumerate() {
                acc += probs[j];
                if r < acc {
                    pick = i;
                    break;
                }
            }
            pick as u32
        };

        ids.push(next);
        print!("{}", bpe.decode(&[next]));
        let _ = std::io::stdout().flush();

        if pos >= seq {
            // window full: reset the cache and reprocess the sliding window
            ctx.reset();
            let start = ids.len().saturating_sub(seq);
            let window: Vec<u32> = ids[start..].to_vec();
            for (p, &t) in window.iter().enumerate() {
                last_logits = ctx.step(t, p);
            }
            pos = window.len();
        } else {
            last_logits = ctx.step(next, pos);
            pos += 1;
        }
    }
    println!();
    let dt = t_gen.elapsed().as_secs_f64();
    if n_tokens > 0 {
        println!(
            "  ({n_tokens} token, {:.2} s, {:.1} token/s — KV-cache)",
            dt,
            n_tokens as f64 / dt
        );
    }
    let _ = t0;
}
fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}
