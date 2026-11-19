// src/corpus.rs
// Streaming token data pipeline.
//
// Why: keeping 3-6 billion tokens in RAM as `Vec<usize>` is impossible
// (8 bytes per token -> 24-48 GB). This module:
//   * writes the tokenized corpus **to disk** as u16 (2 bytes/token),
//   * reads through a **window** during training (the whole file is not loaded),
//   * keeps a fixed-size memory window -> sequential
//     okumalar disk I/O'su yapmaz.

use crate::bpe::Bpe;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};

/// Token cache file: a raw u16 little-endian array.
pub struct TokenDataset {
    file: File,
    n_tokens: u64,
    /// Read window, in tokens
    buf: Vec<u16>,
    buf_start: u64,
}

/// Window size: this multiple of the requested range (>= 4K tokens, <= 1M).
///
/// Training reads from random positions; if the window is too large every miss
/// becomes tens of MB of disk reads (measured: ~4 GB per step, slowing
/// training by 1.8x).
const WINDOW_MIN: usize = 4096;
const WINDOW_MAX: usize = 1 << 20;
const WINDOW_FACTOR: usize = 8;

impl TokenDataset {
    pub fn open(path: &str) -> io::Result<Self> {
        let file = File::open(path)?;
        let bytes = file.metadata()?.len();
        if bytes % 2 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "token cache size must be even (u16)",
            ));
        }
        let n_tokens = bytes / 2;
        Ok(Self {
            file,
            n_tokens,
            buf: Vec::new(),
            buf_start: u64::MAX,
        })
    }

    pub fn len(&self) -> u64 {
        self.n_tokens
    }

    pub fn is_empty(&self) -> bool {
        self.n_tokens == 0
    }

    /// Load the window covering [start, start+len) into memory.
    fn ensure_window(&mut self, start: u64, len: usize) -> io::Result<usize> {
        let need_end = start + len as u64;
        let have = self.buf_start != u64::MAX
            && start >= self.buf_start
            && need_end <= self.buf_start + self.buf.len() as u64;
        if have {
            return Ok((start - self.buf_start) as usize);
        }

        // New window: the requested range plus some look-ahead (useful for
        // sequential access), but never needlessly large.
        let want = (len.saturating_mul(WINDOW_FACTOR)).clamp(WINDOW_MIN, WINDOW_MAX);
        let remaining = self.n_tokens.saturating_sub(start);
        let take = remaining.min(want as u64).max(len as u64) as usize;
        if self.buf.len() != take {
            self.buf.resize(take, 0);
        }
        self.file.seek(SeekFrom::Start(start * 2))?;
        let mut bytes = vec![0u8; take * 2];
        self.file.read_exact(&mut bytes)?;
        for (i, c) in bytes.chunks(2).enumerate() {
            self.buf[i] = u16::from_le_bytes([c[0], c[1]]);
        }
        self.buf_start = start;
        Ok(0)
    }

    /// Reads [start, start+out.len()) (no shift).
    pub fn read(&mut self, start: u64, out: &mut [u16]) -> io::Result<()> {
        assert!(
            start + out.len() as u64 <= self.n_tokens,
            "read exceeds the corpus"
        );
        let off = self.ensure_window(start, out.len())?;
        out.copy_from_slice(&self.buf[off..off + out.len()]);
        Ok(())
    }

    /// Writes `seq_len` inputs plus 1 shifted target starting at `start`.
    /// `x` and `y` are provided by the caller (no per-step allocation).
    pub fn fill_pair(
        &mut self,
        start: u64,
        seq_len: usize,
        x: &mut [u16],
        y: &mut [u16],
    ) -> io::Result<()> {
        assert_eq!(x.len(), seq_len);
        assert_eq!(y.len(), seq_len);
        assert!(
            start + (seq_len as u64 + 1) <= self.n_tokens,
            "window exceeds the corpus"
        );
        self.read(start, x)?;
        self.read(start + 1, y)
    }
}

/// Tokenizes text in chunks and writes a u16 cache file.
/// UTF-8 character boundaries are preserved (aligned to the last newline).
///
/// Speed: a per-word cache (HashMap) is kept - for corpora with a small
/// vocabulary like TinyStories the hit rate is very high, ~50x faster.
pub fn tokenize_file_to_cache(
    text_path: &str,
    cache_path: &str,
    bpe: &Bpe,
    chunk_bytes: usize,
) -> io::Result<u64> {
    tokenize_impl(text_path, cache_path, bpe, chunk_bytes, false)
}

/// Tokenizes by APPENDING to the same cache file (merging multiple corpora).
pub fn tokenize_append(
    text_path: &str,
    cache_path: &str,
    bpe: &Bpe,
    chunk_bytes: usize,
) -> io::Result<u64> {
    tokenize_impl(text_path, cache_path, bpe, chunk_bytes, true)
}

fn tokenize_impl(
    text_path: &str,
    cache_path: &str,
    bpe: &Bpe,
    chunk_bytes: usize,
    append: bool,
) -> io::Result<u64> {
    let mut input = File::open(text_path)?;
    let out = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(!append)
        .append(append)
        .open(cache_path)?;
    let mut out = BufWriter::with_capacity(8 << 20, out);

    let mut buf = vec![0u8; chunk_bytes];
    let mut carry: Vec<u8> = Vec::new();
    let mut total: u64 = 0;
    let mut cache: HashMap<Vec<u8>, Vec<u16>> = HashMap::with_capacity(1 << 17);
    let mut ids: Vec<u16> = Vec::with_capacity(chunk_bytes / 2);
    let mut chunks = 0usize;
    let mut processed = 0usize;

    loop {
        let got = input.read(&mut buf)?;
        if got == 0 {
            if !carry.is_empty() {
                let text = String::from_utf8_lossy(&carry).into_owned();
                ids.clear();
                encode_into(bpe, &text, &mut cache, &mut ids)?;
                total += write_ids(&mut out, &ids)?;
            }
            break;
        }
        let mut data = std::mem::take(&mut carry);
        data.extend_from_slice(&buf[..got]);

        // Carry everything after the last newline into the next chunk
        let cut = match data.iter().rposition(|&b| b == b'\n') {
            Some(p) => p + 1,
            None => data.len(),
        };
        let head = String::from_utf8_lossy(&data[..cut]).into_owned();
        carry = data[cut..].to_vec();

        ids.clear();
        encode_into(bpe, &head, &mut cache, &mut ids)?;
        total += write_ids(&mut out, &ids)?;
        chunks += 1;
        if chunks % 4 == 0 {
            eprintln!(
                "  [tokenize] {:.0} MB processed, {} tokens, cache {} words",
                (processed + got) as f64 / (1 << 20) as f64,
                total,
                cache.len()
            );
        }
        processed += got;
    }

    out.flush()?;
    Ok(total)
}

/// Word-cached encode: split text into words, solve each word once.
fn encode_into(
    bpe: &Bpe,
    text: &str,
    cache: &mut HashMap<Vec<u8>, Vec<u16>>,
    out: &mut Vec<u16>,
) -> io::Result<()> {
    for w in text.split_inclusive([' ', '\n']) {
        if w.is_empty() {
            continue;
        }
        let key = w.as_bytes();
        if let Some(v) = cache.get(key) {
            out.extend_from_slice(v);
            continue;
        }
        let ids = bpe.encode_word(w);
        let mut v: Vec<u16> = Vec::with_capacity(ids.len());
        for id in ids {
            v.push(u16::try_from(id).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("token id {id} does not fit in u16"),
                )
            })?);
        }
        out.extend_from_slice(&v);
        if cache.len() < 4_000_000 {
            cache.insert(key.to_vec(), v);
        }
    }
    Ok(())
}

fn write_ids<W: Write>(w: &mut W, ids: &[u16]) -> io::Result<u64> {
    let mut bytes = Vec::with_capacity(ids.len() * 2);
    for &v in ids {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    w.write_all(&bytes)?;
    Ok(ids.len() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn dataset_roundtrip_and_windows() {
        let dir = std::env::temp_dir().join(format!("sfish_corpus_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cache = dir.join("t.bin");
        // write 1000 tokens
        let ids: Vec<u16> = (0..1000u16).map(|i| i.wrapping_mul(7)).collect();
        {
            let mut f = File::create(&cache).unwrap();
            for v in &ids {
                f.write_all(&v.to_le_bytes()).unwrap();
            }
        }
        let mut ds = TokenDataset::open(cache.to_str().unwrap()).unwrap();
        assert_eq!(ds.len(), 1000);

        let mut x = vec![0u16; 8];
        let mut y = vec![0u16; 8];
        for start in [0u64, 1, 500, 991] {
            ds.fill_pair(start, 8, &mut x, &mut y).unwrap();
            assert_eq!(x, ids[start as usize..start as usize + 8]);
            assert_eq!(y, ids[start as usize + 1..start as usize + 9]);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tokenize_writes_u16_and_roundtrips() {
        let dir = std::env::temp_dir().join(format!("sfish_tok_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let txt = dir.join("c.txt");
        let cache = dir.join("c.bin");
        let content = "the cat sat on the mat.\nthe dog ran.\nsample unicode 🙂 done\n";
        std::fs::write(&txt, content).unwrap();

        let bpe = crate::bpe::train_bpe(content, 290, 10_000);
        // Tokenize in small chunks: UTF-8 boundaries must be preserved.
        let n = tokenize_file_to_cache(txt.to_str().unwrap(), cache.to_str().unwrap(), &bpe, 16)
            .unwrap();

        // Read the whole cache and decode it -> the text must round-trip
        let mut ds = TokenDataset::open(cache.to_str().unwrap()).unwrap();
        assert_eq!(ds.len(), n);
        let mut raw = vec![0u16; n as usize];
        ds.read(0, &mut raw).unwrap();
        let all: Vec<u32> = raw.iter().map(|&v| v as u32).collect();
        assert_eq!(bpe.decode(&all), content);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
