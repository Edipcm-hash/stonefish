// src/checkpoint.rs
// GPT checkpoint formatu — sürüm 2.
//
// v1 (eski `save_gpt`) sadece ham tensörleri yazıyordu: config yok, vocab yok,
// magic yok. Sonuç: korpus dosyası bir karakter değişse ID'ler kayıyor ve model
// sessizce saçmalıyordu. v2 her şeyi tek dosyaya gömer ve yüklemede doğrular.
//
// Düzen (little-endian):
//   magic        : 8 bayt  b"SFGPT002"
//   version      : u32
//   config       : vocab_size, max_seq_len, d_model, n_layers, n_heads  (5 × u32)
//   vocab        : n_chars u32 + n_chars × u32 (char kodu)
//   n_tensors    : u32
//   tensörler    : n × (rows u32, cols u32, rows*cols × f32)
//
// Tensör sırası `GPT::parameters()` ile birebir aynıdır; yüklemede hem sayı hem
// her tensörün şekli doğrulanır.

use crate::tokenizer::Tokenizer;
use crate::transformer::{GptConfig, GPT};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};

const MAGIC: &[u8; 8] = b"SFGPT002";
const VERSION: u32 = 2;

fn write_u32<W: Write>(w: &mut W, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

/// Modeli ve tokenizer'ı tek dosyaya yazar.
pub fn save_gpt(path: &str, model: &GPT, tok: &Tokenizer) -> io::Result<()> {
    let cfg = model.config;
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);

    w.write_all(MAGIC)?;
    write_u32(&mut w, VERSION)?;
    write_u32(&mut w, cfg.vocab_size as u32)?;
    write_u32(&mut w, cfg.max_seq_len as u32)?;
    write_u32(&mut w, cfg.d_model as u32)?;
    write_u32(&mut w, cfg.n_layers as u32)?;
    write_u32(&mut w, cfg.n_heads as u32)?;

    // vocab
    if tok.vocab_size != cfg.vocab_size {
        return Err(invalid(format!(
            "tokenizer vocab_size ({}) != model vocab_size ({})",
            tok.vocab_size, cfg.vocab_size
        )));
    }
    write_u32(&mut w, tok.chars.len() as u32)?;
    for &c in &tok.chars {
        write_u32(&mut w, c as u32)?;
    }

    // tensörler
    let params = model.parameters();
    write_u32(&mut w, params.len() as u32)?;
    for p in &params {
        let d = p.data.borrow();
        write_u32(&mut w, d.rows as u32)?;
        write_u32(&mut w, d.cols as u32)?;
        for &v in &d.data {
            w.write_all(&v.to_le_bytes())?;
        }
    }
    w.flush()?;
    Ok(())
}

/// Checkpoint'i okur ve modeli + tokenizer'ı yeniden kurar.
/// `tokenizer::Tokenizer::from_chars` ile vocab birebir geri gelir; korpus
/// dosyasına artık ihtiyaç yoktur.
pub fn load_gpt(path: &str) -> io::Result<(GPT, Tokenizer)> {
    let f = File::open(path)?;
    let mut r = BufReader::new(f);

    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(invalid(
            "magic yanlış — bu bir StonefishAI GPT checkpoint'i değil (ya da v1 formatı)",
        ));
    }
    let version = read_u32(&mut r)?;
    if version != VERSION {
        return Err(invalid(format!(
            "desteklenmeyen checkpoint versiyonu: {version} (beklenen {VERSION})"
        )));
    }

    let config = GptConfig {
        vocab_size: read_u32(&mut r)? as usize,
        max_seq_len: read_u32(&mut r)? as usize,
        d_model: read_u32(&mut r)? as usize,
        n_layers: read_u32(&mut r)? as usize,
        n_heads: read_u32(&mut r)? as usize,
    };
    if config.n_heads == 0 || config.d_model % config.n_heads != 0 {
        return Err(invalid(format!(
            "bozuk config: d_model={} n_heads={}",
            config.d_model, config.n_heads
        )));
    }

    let n_chars = read_u32(&mut r)? as usize;
    let mut chars = Vec::with_capacity(n_chars);
    for _ in 0..n_chars {
        let code = read_u32(&mut r)?;
        let c = char::from_u32(code)
            .ok_or_else(|| invalid(format!("geçersiz karakter kodu: {code}")))?;
        chars.push(c);
    }
    let tok = Tokenizer::from_chars(chars)?;
    if tok.vocab_size != config.vocab_size {
        return Err(invalid(format!(
            "vocab boyutu uyuşmuyor: checkpoint config {} != vocab {}",
            config.vocab_size, tok.vocab_size
        )));
    }

    // Modeli aynı mimariyle kur (ağırlıklar birazdan üzerine yazılacak).
    let mut seed = 0u64;
    let model = GPT::new(config, &mut seed);

    let n_tensors = read_u32(&mut r)? as usize;
    let params = model.parameters();
    if n_tensors != params.len() {
        return Err(invalid(format!(
            "tensör sayısı uyuşmuyor: dosyada {n_tensors}, modelde {}",
            params.len()
        )));
    }

    let mut buf4 = [0u8; 4];
    for (i, p) in params.iter().enumerate() {
        let rows = read_u32(&mut r)? as usize;
        let cols = read_u32(&mut r)? as usize;
        let mut d = p.data.borrow_mut();
        if (d.rows, d.cols) != (rows, cols) {
            return Err(invalid(format!(
                "tensör {i} şekli uyuşmuyor: dosyada {rows}x{cols}, modelde {}x{}",
                d.rows, d.cols
            )));
        }
        for k in 0..d.data.len() {
            r.read_exact(&mut buf4)?;
            d.data[k] = f32::from_le_bytes(buf4);
        }
    }

    Ok((model, tok))
}
