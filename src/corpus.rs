// src/corpus.rs
// Akışlı token veri hattı.
//
// Neden: 3-6 milyar token'ı `Vec<usize>` olarak RAM'de tutmak imkânsız
// (token başına 8 bayt → 24-48 GB). Bu modül:
//   * tokenize edilmiş korpusu **diske** u16 olarak yazar (2 bayt/token),
//   * eğitim sırasında **pencere** okuyarak erişir (tüm dosya yüklenmez),
//   * sabit boyutlu bir bellek penceresi tutar (varsayılan 64 MB) → ardışık
//     okumalar disk I/O'su yapmaz.

use crate::bpe::Bpe;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};

/// Token cache dosyası: ham u16 little-endian dizisi.
pub struct TokenDataset {
    file: File,
    n_tokens: u64,
    /// Bellek penceresi (token cinsinden)
    buf: Vec<u16>,
    buf_start: u64,
}

/// Pencere boyu: istenen aralığın bu katı (en az 4K token, en çok 1M).
///
/// Eğitim rastgele konumlardan okur; pencere çok büyük olursa her ıskalama
/// onlarca MB disk okumasına dönüşür (ölçüldü: adımda ~4 GB trafik, eğitimi
/// 1,8× yavaşlatıyordu).
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
                "token cache boyutu çift olmalı (u16)",
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

    /// [start, start+len) aralığını içeren pencereyi belleğe al.
    fn ensure_window(&mut self, start: u64, len: usize) -> io::Result<usize> {
        let need_end = start + len as u64;
        let have = self.buf_start != u64::MAX
            && start >= self.buf_start
            && need_end <= self.buf_start + self.buf.len() as u64;
        if have {
            return Ok((start - self.buf_start) as usize);
        }

        // Yeni pencere: istenen aralık + biraz ileriye bak (ardışık erişim
        // için faydalı), ama asla gereksiz büyük değil.
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

    /// [start, start+out.len()) aralığını okur (kaydırma yok).
    pub fn read(&mut self, start: u64, out: &mut [u16]) -> io::Result<()> {
        assert!(
            start + out.len() as u64 <= self.n_tokens,
            "okuma korpusu aşıyor"
        );
        let off = self.ensure_window(start, out.len())?;
        out.copy_from_slice(&self.buf[off..off + out.len()]);
        Ok(())
    }

    /// `start`'tan itibaren `seq_len` girdi + 1 kaydırılmış hedef yazar.
    /// `x` ve `y` çağıran tarafından sağlanır (adım başına tahsis yok).
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
            "pencere korpusu aşıyor"
        );
        self.read(start, x)?;
        self.read(start + 1, y)
    }
}

/// Metni parçalar hâlinde tokenize edip u16 cache dosyasına yazar.
/// UTF-8 karakter sınırları korunur (satır sonuna hizalanarak).
///
/// Hız: kelime başına önbellek (HashMap) tutulur — TinyStories gibi küçük
/// sözlü korpuslarda isabet oranı çok yüksektir ve tokenize ~50× hızlanır.
pub fn tokenize_file_to_cache(
    text_path: &str,
    cache_path: &str,
    bpe: &Bpe,
    chunk_bytes: usize,
) -> io::Result<u64> {
    tokenize_impl(text_path, cache_path, bpe, chunk_bytes, false)
}

/// Aynı cache dosyasına EKLEYEREK tokenize eder (çoklu korpus birleştirme).
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

        // Son satır sonundan sonrasını bir sonraki parçaya taşı
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
                "  [tokenize] {:.0} MB işlendi, {} token, önbellek {} kelime",
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


/// Kelime önbellekli encode: metni kelimelere böl, her kelimeyi bir kez çöz.
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
                    format!("token id {id} u16'ya sığmıyor"),
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
        // 1000 token yaz
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
        let content = "the cat sat on the mat.\nthe dog ran.\nşğüöç 🙂 done\n";
        std::fs::write(&txt, content).unwrap();

        let bpe = crate::bpe::train_bpe(content, 290, 10_000);
        // Küçük parçalarla tokenize et: UTF-8 sınırları korunmalı.
        let n = tokenize_file_to_cache(txt.to_str().unwrap(), cache.to_str().unwrap(), &bpe, 16)
            .unwrap();

        // Cache'in tamamını oku ve decode et → metin aynen dönmeli
        let mut ds = TokenDataset::open(cache.to_str().unwrap()).unwrap();
        assert_eq!(ds.len(), n);
        let mut raw = vec![0u16; n as usize];
        ds.read(0, &mut raw).unwrap();
        let all: Vec<u32> = raw.iter().map(|&v| v as u32).collect();        assert_eq!(bpe.decode(&all), content);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
