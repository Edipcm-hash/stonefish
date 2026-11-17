// src/checkpoint.rs
// GPT checkpoint format - version 2.
//
// v1 (the old `save_gpt`) wrote raw tensors only: no config, no vocab, no
// magic. Result: if the corpus file changed by a single character the ids
// shifted and the model produced garbage silently. v2 embeds everything in one
//
// Layout (little-endian):
//   magic        : 8 bayt  b"SFGPT002"
//   version      : u32
//   config       : vocab_size, max_seq_len, d_model, n_layers, n_heads  (5 × u32)
//   vocab        : n_chars u32 + n_chars × u32 (char kodu)
//   n_tensors    : u32
//   tensors      : n x (rows u32, cols u32, rows*cols x f32)
//
// Tensor order is identical to `GPT::parameters()`; on load both the count
// and every tensor's shape are validated.

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

/// Writes the model and the tokenizer into a single file.
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

    // tensors
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

/// Reads a checkpoint and rebuilds the model + tokenizer.
/// `tokenizer::Tokenizer::from_chars` restores the vocabulary exactly; the
/// corpus file is no longer needed.
pub fn load_gpt(path: &str) -> io::Result<(GPT, Tokenizer)> {
    let f = File::open(path)?;
    let mut r = BufReader::new(f);

    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(invalid(
            "wrong magic - this is not a StonefishAI GPT checkpoint (or it is v1 format)",
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
            .ok_or_else(|| invalid(format!("invalid character code: {code}")))?;
        chars.push(c);
    }
    let tok = Tokenizer::from_chars(chars)?;
    if tok.vocab_size != config.vocab_size {
        return Err(invalid(format!(
            "vocabulary size mismatch: checkpoint config {} != vocab {}",
            config.vocab_size, tok.vocab_size
        )));
    }

    // Build the model with the same architecture (weights are overwritten next).
    let mut seed = 0u64;
    let model = GPT::new(config, &mut seed);

    let n_tensors = read_u32(&mut r)? as usize;
    let params = model.parameters();
    if n_tensors != params.len() {
        return Err(invalid(format!(
            "tensor count mismatch: file has {n_tensors}, model has {}",
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
                "tensor {i} shape mismatch: file has {rows}x{cols}, model has {}x{}",
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
