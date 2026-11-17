// src/en_train.rs
// İngilizce (BPE tokenizer + disk cache) eğitim hattı.
//
// Türkçe karakter-seviyesi yoldan (`gpt-train`) ayrıdır çünkü:
//  * veri kaynağı disk cache'ten AKIŞLA okunur (RAM'de tutulmaz),
//  * tokenizer BPE'dir (checkpoint yerine ayrı `SFBPE` dosyası + config'te
//    vocab_size),
//  * model CPU ikizi olmadan doğrudan cihazda başlatılır,
//  * gradyan biriktirme ve periyodik resume vardır.

#![cfg(feature = "cuda")]

use crate::bpe::{train_bpe, Bpe};
use crate::corpus::{tokenize_file_to_cache, TokenDataset};
use crate::gpu_train::GpuGpt;
use crate::transformer::GptConfig;
use std::io::{Read, Write};

pub const MERGES_MAGIC: &[u8; 8] = b"SFBPE001";

// ============================================================
// Tokenizer dosyası
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
            "BPE dosyası tanınmadı",
        ));
    }
    let mut u4 = [0u8; 4];
    r.read_exact(&mut u4)?;
    if u32::from_le_bytes(u4) != 1 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "desteklenmeyen BPE sürümü",
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
// prepare: BPE öğren + cache üret
// ============================================================

pub fn prepare(train_txt: &str, valid_txt: &str, vocab: usize, out_prefix: &str) {
    let t0 = std::time::Instant::now();
    println!("🐟 İngilizce veri hazırlığı\n");
    println!("Korpus : {train_txt}");
    println!("Hedef  : vocab={vocab}, çıktı={out_prefix}.{{bpe,train.bin,valid.bin}}\n");

    // BPE öğrenimi için korpusun başından örneklem al (tüm 2 GB gerekmez;
    // 200 MB, 16k merge için fazlasıyla temsilî).
    let mut f = std::fs::File::open(train_txt).expect("korpus açılamadı");
    let mut buf = vec![0u8; 200 << 20];
    let got = f.read(&mut buf).expect("okuma hatası");
    buf.truncate(got);
    let sample = String::from_utf8_lossy(&buf).into_owned();
    println!("BPE örneklemi: {} MB", got >> 20);

    let bpe = train_bpe(&sample, vocab, 300_000);
    println!("BPE hazır: vocab={} ({} merge), {:.1} s", bpe.vocab(), bpe.merges.len(), t0.elapsed().as_secs_f64());

    let bpe_path = format!("{out_prefix}.bpe");
    save_bpe(&bpe_path, &bpe).expect("BPE yazılamadı");

    // Tokenize (akışlı)
    let tr_bin = format!("{out_prefix}.train.bin");
    let va_bin = format!("{out_prefix}.valid.bin");
    let t1 = std::time::Instant::now();
    let n_tr = tokenize_file_to_cache(train_txt, &tr_bin, &bpe, 64 << 20).expect("train tokenize");
    println!("train tokenize: {n_tr} token, {:.1} s", t1.elapsed().as_secs_f64());
    let t2 = std::time::Instant::now();
    let n_va = tokenize_file_to_cache(valid_txt, &va_bin, &bpe, 64 << 20).expect("valid tokenize");
    println!("valid tokenize: {n_va} token, {:.1} s", t2.elapsed().as_secs_f64());
    let corpus_bytes = std::fs::metadata(train_txt).map(|m| m.len()).unwrap_or(0);
    println!(
        "\nSıkıştırma: {:.2} karakter/token",
        corpus_bytes as f64 / n_tr.max(1) as f64
    );
    println!("Toplam {:.1} dk", t0.elapsed().as_secs_f64() / 60.0);
}

// ============================================================
// train: 250M konfigürasyonu
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
            total_tokens: 470_000_000,
            max_lr: 6e-4,
            warmup: 500,
            clip: 1.0,
            ckpt_every: 500,
            log_every: 20,
        }
    }
}

pub fn train(a: &EnTrainArgs) {
    let bpe = load_bpe(&format!("{}.bpe", a.corpus_prefix)).expect("BPE yüklenemedi");
    let mut data =
        TokenDataset::open(&format!("{}.train.bin", a.corpus_prefix)).expect("cache açılamadı");
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

    println!("🐟 StonefishAI — İngilizce GPT eğitimi (250M sınıfı)\n");
    println!("Tokenizer : BPE vocab={vocab}");
    println!("Korpus    : {} token (cache), seq_len={}", data.len(), a.seq_len);
    println!(
        "Model     : d_model={}, layers={}, heads={} (d_head={}), ctx={}",
        a.d_model,
        a.n_layers,
        a.n_heads,
        a.d_model / a.n_heads,
        a.seq_len
    );
    println!(
        "Batch     : {} × accum {} = {} dizi ({} token/adım)",
        a.batch, a.accum, a.batch * a.accum, tokens_per_step
    );
    println!(
        "Plan      : {:.2} B token → {} adım, max_lr={}, clip={}",
        a.total_tokens as f64 / 1e9,
        total_steps,
        a.max_lr,
        a.clip
    );
    println!("Checkpoint: her {} adım → {}.state\n", a.ckpt_every, a.out_prefix);

    let mut model = GpuGpt::new_random(cfg, a.batch, a.seq_len, 0xDEADBEEF);
    let n_params: usize = model.params_mut().iter().map(|p| p.n()).sum();
    println!("Parametre : {n_params} ({:.0} M)", n_params as f64 / 1e6);

    // resume
    let ckpt = format!("{}.state", a.out_prefix);
    let mut start_step = 0usize;
    if std::path::Path::new(&ckpt).exists() {
        match model.load_state(&ckpt) {
            Ok(t) => {
                start_step = t;
                println!("▶ Resume: {ckpt} (adım {start_step})");
            }
            Err(e) => println!("⚠️  Resume edilemedi ({e}), sıfırdan başlanıyor"),
        }
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
            // her mikro-batch için rastgele konum
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
                Ok(()) => println!("  💾 checkpoint → {ckpt} (adım {})", step + 1),
                Err(e) => println!("  ⚠️  checkpoint yazılamadı: {e}"),
            }
        }
    }

    let _ = model.save_state(&format!("{}.final.state", a.out_prefix));
    println!(
        "\n✅ Eğitim tamam: {} adım, {:.1} dk, ortalama {:.0} tok/s",
        total_steps - start_step,
        run_t0.elapsed().as_secs_f64() / 60.0,
        (tokens_done as f64) / run_t0.elapsed().as_secs_f64()
    );
}

pub fn lr_at(step: usize, total: usize, max_lr: f32, warmup: usize, min_ratio: f32) -> f32 {
    if step < warmup {
        return max_lr * (step + 1) as f32 / warmup.max(1) as f32;
    }
    let span = total.saturating_sub(warmup).max(1) as f32;
    let t = ((step - warmup) as f32 / span).min(1.0);
    let cos = 0.5 * (1.0 + (std::f32::consts::PI * t).cos());
    max_lr * (min_ratio + (1.0 - min_ratio) * cos)
}

fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}
