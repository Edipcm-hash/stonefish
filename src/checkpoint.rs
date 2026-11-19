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

// ============================================================
// SFEN001 (the GPU trainer's format) -> CPU model
// ============================================================

/// Reads only the header of an SFEN001 file (no weights).
///
/// Layout: magic SFEN001\0, version u32, config (5 x u32), adam_t u64,
/// tensor count u32, then tensors as (rows u32, cols u32, f32...). The tensor
/// order is the same as GPT::parameters(), because the device trainer was
/// built from the CPU model in the first place.
pub fn read_sfen_cfg(path: &str) -> io::Result<(GptConfig, u64)> {
    let f = File::open(path)?;
    let mut r = BufReader::new(f);
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != b"SFEN001\0" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not an SFEN001 weight file",
        ));
    }
    let mut b4 = [0u8; 4];
    r.read_exact(&mut b4)?;
    if u32::from_le_bytes(b4) != 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unsupported SFEN version",
        ));
    }
    let mut cfg_u = [0u32; 5];
    for v in cfg_u.iter_mut() {
        r.read_exact(&mut b4)?;
        *v = u32::from_le_bytes(b4);
    }
    let mut b8 = [0u8; 8];
    r.read_exact(&mut b8)?;
    let step = u64::from_le_bytes(b8);
    let cfg = GptConfig {
        vocab_size: cfg_u[0] as usize,
        max_seq_len: cfg_u[1] as usize,
        d_model: cfg_u[2] as usize,
        n_layers: cfg_u[3] as usize,
        n_heads: cfg_u[4] as usize,
    };
    Ok((cfg, step))
}

/// Loads SFEN001 weights into an existing CPU model. Returns the training step.
///
/// The two models keep their parameters in a different order, so this is not a
/// zip: the device block is (norm1, wq, wk, wv, wo, norm2, gate, up, down)
/// while the CPU block is (norm1, norm2, wq, wk, wv, wo, gate, up, down). Both
/// layouts are spelled out here, and every tensor is shape-checked, so a
/// mismatch is an error instead of a silently scrambled model.
pub fn load_sfen_into(path: &str, model: &mut GPT) -> io::Result<u64> {
    let (cfg, step) = read_sfen_cfg(path)?;
    let have = model.config;
    if cfg.vocab_size != have.vocab_size
        || cfg.max_seq_len != have.max_seq_len
        || cfg.d_model != have.d_model
        || cfg.n_layers != have.n_layers
        || cfg.n_heads != have.n_heads
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "config mismatch between the state file and the model",
        ));
    }

    let f = File::open(path)?;
    let mut r = BufReader::new(f);
    let mut skip = [0u8; 8 + 4 + 20 + 8 + 4];
    r.read_exact(&mut skip)?;

    // Read every tensor in file order.
    let per_block = 9usize;
    let expected = 1 + cfg.n_layers * per_block + 1;
    let mut file_tensors: Vec<(usize, usize, Vec<f32>)> = Vec::with_capacity(expected);
    loop {
        let mut hdr = [0u8; 8];
        match r.read_exact(&mut hdr) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
        let rows = u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]) as usize;
        let cols = u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]) as usize;
        let mut buf = vec![0u8; rows * cols * 4];
        r.read_exact(&mut buf)?;
        let mut data = vec![0f32; rows * cols];
        for (k, v) in data.iter_mut().enumerate() {
            let o = k * 4;
            *v = f32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);
        }
        file_tensors.push((rows, cols, data));
    }
    if file_tensors.len() != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("tensor count {} != expected {expected}", file_tensors.len()),
        ));
    }

    // File order -> CPU order (per block).
    let order = [0usize, 5, 1, 2, 3, 4, 6, 7, 8];
    let mut plan: Vec<usize> = vec![0];
    for b in 0..cfg.n_layers {
        let base = 1 + b * per_block;
        for &o in &order {
            plan.push(base + o);
        }
    }
    plan.push(1 + cfg.n_layers * per_block);

    let params = model.parameters();
    if params.len() != plan.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "model has {} parameters, plan has {}",
                params.len(),
                plan.len()
            ),
        ));
    }
    for (slot, &src) in plan.iter().enumerate() {
        let (rows, cols, ref data) = file_tensors[src];
        let p = &params[slot];
        let mut d = p.data.borrow_mut();
        if d.rows != rows || d.cols != cols {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "slot {slot} <- file tensor {src}: shape {rows}x{cols}, model {}x{}",
                    d.rows, d.cols
                ),
            ));
        }
        d.data.copy_from_slice(data);
    }
    Ok(step)
}

#[cfg(test)]
mod sfen_tests {
    use super::*;
    use crate::transformer::GPT;
    use std::io::Write;

    /// Writes a tiny SFEN001 file whose every tensor is filled with a value that
    /// identifies its ROLE (1 = tok_emb, 2 = norm1, 3 = wq, ... 11 = lnf).
    fn write_tiny_sfen(path: &str, cfg: &GptConfig, model: &GPT) {
        let f = std::fs::File::create(path).unwrap();
        let mut w = std::io::BufWriter::new(f);
        w.write_all(b"SFEN001\0").unwrap();
        w.write_all(&1u32.to_le_bytes()).unwrap();
        for v in [
            cfg.vocab_size,
            cfg.max_seq_len,
            cfg.d_model,
            cfg.n_layers,
            cfg.n_heads,
        ] {
            w.write_all(&(v as u32).to_le_bytes()).unwrap();
        }
        w.write_all(&7u64.to_le_bytes()).unwrap(); // adam_t
        let params = model.parameters();
        // CPU order: tok, [n1, n2, wq, wk, wv, wo, gate, up, down] x L, lnf
        // FILE order: tok, [n1, wq, wk, wv, wo, n2, gate, up, down] x L, lnf
        let roles = [2u32, 3, 4, 5, 6, 7, 8, 9, 10]; // file roles per block
        let cpu_idx = [0usize, 2, 3, 4, 5, 1, 6, 7, 8]; // matching CPU slots
        let mut seq: Vec<(usize, u32)> = vec![(0, 1)];
        for b in 0..cfg.n_layers {
            for k in 0..9 {
                seq.push((1 + b * 9 + cpu_idx[k], roles[k]));
            }
        }
        seq.push((params.len() - 1, 11));

        w.write_all(&(seq.len() as u32).to_le_bytes()).unwrap();
        for (slot, role) in seq {
            let d = params[slot].data.borrow();
            let (rows, cols) = (d.rows, d.cols);
            drop(d);
            w.write_all(&(rows as u32).to_le_bytes()).unwrap();
            w.write_all(&(cols as u32).to_le_bytes()).unwrap();
            for _ in 0..(rows * cols) {
                w.write_all(&(role as f32).to_le_bytes()).unwrap();
            }
        }
        w.flush().unwrap();
    }

    /// Regression test for the bug found while adding the CPU generation path:
    /// the device block order (norm1, wq..wo, norm2) is NOT the CPU order
    /// (norm1, norm2, wq..wo). A zip instead of a permutation silently loads q
    /// weights into the norm2 slot; this test would have caught it.
    #[test]
    fn sfen_block_order_matches_roles() {
        let cfg = GptConfig {
            vocab_size: 11,
            max_seq_len: 4,
            d_model: 8,
            n_layers: 1,
            n_heads: 2,
        };
        let mut seed = 42u64;
        let src = GPT::new(cfg, &mut seed);
        let path = std::env::temp_dir().join("stonefish_tiny_sfen.state");
        let path = path.to_str().unwrap().to_string();
        write_tiny_sfen(&path, &cfg, &src);

        let (got_cfg, step) = read_sfen_cfg(&path).unwrap();
        assert_eq!(got_cfg, cfg);
        assert_eq!(step, 7);

        let mut dst = GPT::new(cfg, &mut seed);
        let loaded = load_sfen_into(&path, &mut dst).unwrap();
        assert_eq!(loaded, 7);

        // CPU order: tok_emb, norm1, norm2, wq, wk, wv, wo, gate, up, down, lnf
        let expect = [1.0f32, 2.0, 7.0, 3.0, 4.0, 5.0, 6.0, 8.0, 9.0, 10.0, 11.0];
        let loaded_params = dst.parameters();
        for (slot, want) in expect.iter().enumerate() {
            let d = loaded_params[slot].data.borrow();
            for (k, v) in d.data.iter().enumerate() {
                assert_eq!(
                    *v, *want,
                    "slot {slot} element {k}: expected role {want}, got {v}"
                );
            }
        }
        let _ = std::fs::remove_file(&path);
    }
}
