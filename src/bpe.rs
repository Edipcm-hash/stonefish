// src/bpe.rs
// Byte-level BPE tokenizer — eğitim (merge öğrenimi) + encode/decode.
//
// Neden BPE: karakter seviyesinde model kapasitesinin büyük kısmını heceleme
// öğrenmeye harcıyor ("once upon a time" = 17 ayrı üretim adımı). Byte-level
// BPE ile aynı bağlam penceresi 4-5 kat daha fazla metin kapsar.
//
// Tasarım:
//  * Taban: 256 byte sembolü (her UTF-8 byte'ı) → hiçbir metin "bilinmeyen
//    karakter" hatası vermez, vocab 256'dan başlar.
//  * Merge öğrenimi: korpusu kelimelere böl, kelime frekansları üzerinde en sık
//    bitişik çifti birleştir (klasik BPE). UTF-8 byte sınırları korunur.
//  * Karmaşıklık: kelime başına O(uzunluk²) tarama, frekansla ağırlıklı.
//  * Deterministik: eşit frekansta düşük indeksli çift kazanır (tekrar üretilebilir).

use std::collections::HashMap;

/// Öğrenilmiş BPE modeli: 256 byte tabanı + sıralı merge listesi.
#[derive(Clone, Debug)]
pub struct Bpe {
    /// merge id → (sol, sağ) sembol çifti. Sembol id'leri: < 256 byte, >= 256 merge.
    pub merges: Vec<(u32, u32)>,
    /// encode sırasında arama için: (sol, sağ) → yeni id
    lookup: HashMap<(u32, u32), u32>,
}

impl Bpe {
    /// Merge listesinden kur (checkpoint yükleme yolu).
    pub fn from_merges(merges: Vec<(u32, u32)>) -> Self {
        let mut lookup = HashMap::with_capacity(merges.len());
        for (i, &(a, b)) in merges.iter().enumerate() {
            lookup.insert((a, b), 256 + i as u32);
        }
        Self { merges, lookup }
    }

    /// Toplam kelime dağarcığı: 256 byte + merge sayısı.
    pub fn vocab(&self) -> usize {
        256 + self.merges.len()
    }

    /// Her byte'ı bir sembol olarak başlat.
    pub fn byte_symbols(text: &str) -> Vec<u32> {
        text.bytes().map(|b| b as u32).collect()
    }

    /// Sembol dizisine greedy BPE uygula: her adımda sözlükte olan en düşük
    /// indeksli (yani en erken öğrenilmiş) çifti birleştir.
    pub fn encode_symbols(&self, symbols: &mut Vec<u32>) {
        if symbols.len() < 2 {
            return;
        }
        loop {
            let mut best: Option<(u32, usize)> = None; // (yeni id, konum)
            for i in 0..symbols.len() - 1 {
                if let Some(&id) = self.lookup.get(&(symbols[i], symbols[i + 1])) {
                    match best {
                        Some((bid, _)) if bid <= id => {}
                        _ => best = Some((id, i)),
                    }
                }
            }
            let (new_id, pos) = match best {
                Some(v) => v,
                None => break,
            };
            symbols[pos] = new_id;
            symbols.remove(pos + 1);
        }
    }

    /// Metni token id'lerine çevir.
    ///
    /// DİKKAT: Uzun metinlerde doğrudan çağırmak O(n²) olur (her merge için tüm
    /// sembol vektörü taranır). Korpus tokenizasyonu `encode_word` + kelime
    /// önbelleği kullanır (bkz. `corpus::tokenize_file_to_cache`).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len() / 3 + 4);
        for w in text.split_inclusive([' ', '\n']) {
            out.extend(self.encode_word(w));
        }
        out
    }

    /// Tek bir "kelime" (boşluk/satır sonu dahil) için BPE. O(L²), L kelime
    /// uzunluğu — pratikte ≤ 12.
    pub fn encode_word(&self, word: &str) -> Vec<u32> {
        let mut syms = Self::byte_symbols(word);
        self.encode_symbols(&mut syms);
        syms
    }

    /// Token id'lerini metne çevir. UTF-8 sınırında kesilirse kayıp byte'ı atar.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::with_capacity(ids.len() * 2);
        for &id in ids {
            self.expand(id, &mut bytes);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn expand(&self, id: u32, out: &mut Vec<u8>) {
        if id < 256 {
            out.push(id as u8);
            return;
        }
        let (a, b) = self.merges[(id - 256) as usize];
        self.expand(a, out);
        self.expand(b, out);
    }
}

// ============================================================
// Merge öğrenimi
// ============================================================

/// Korpus üzerinde BPE merge'lerini öğren.
///
/// `target_vocab`: 256 + merges.len() bu değere ulaşınca durur.
/// `max_words`: en sık kaç ayrı kelime kullanılsın (bellek/hız kontrolü).
///
/// Artımlı çift sayımı kullanılır: her merge'den sonra yalnızca o çifti içeren
/// kelimeler yeniden taranır. Naif yaklaşım (her adımda tüm korpusu taramak)
/// 16k merge'de saatler sürüyordu.
pub fn train_bpe(text: &str, target_vocab: usize, max_words: usize) -> Bpe {
    assert!(target_vocab > 256, "target_vocab > 256 olmalı");

    // 1) Kelime frekansları. Boşluk/satır sonu kelimeye dahil edilir (GPT-2
    //    yaklaşımı: " the" tek token olabilsin).
    let mut freq: HashMap<Vec<u32>, u64> = HashMap::new();
    for word in text.split_inclusive([' ', '\n']) {
        let syms = Bpe::byte_symbols(word);
        if syms.is_empty() {
            continue;
        }
        *freq.entry(syms).or_insert(0) += 1;
    }

    // 2) En sık `max_words` kelime
    let mut words: Vec<(Vec<u32>, u64)> = freq.into_iter().collect();
    words.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    words.truncate(max_words);

    // 3) Artımlı yapılar
    let mut pair_counts: HashMap<(u32, u32), u64> = HashMap::new();
    let mut pair_words: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (wi, (w, c)) in words.iter().enumerate() {
        for i in 0..w.len().saturating_sub(1) {
            let p = (w[i], w[i + 1]);
            *pair_counts.entry(p).or_insert(0) += *c;
            let v = pair_words.entry(p).or_default();
            if v.last() != Some(&wi) {
                v.push(wi);
            }
        }
    }

    let mut merges: Vec<(u32, u32)> = Vec::with_capacity(target_vocab - 256);
    let mut lookup: HashMap<(u32, u32), u32> = HashMap::new();

    while 256 + merges.len() < target_vocab {
        // En sık çift; eşitlikte küçük (a,b) kazanır → deterministik.
        let best = match pair_counts
            .iter()
            .filter(|(_, &c)| c > 0)
            .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
        {
            Some((&p, _)) => p,
            None => break,
        };

        let new_id = 256 + merges.len() as u32;
        merges.push(best);
        lookup.insert(best, new_id);

        // Yalnız bu çifti içeren kelimeleri güncelle
        let affected = pair_words.remove(&best).unwrap_or_default();
        pair_counts.remove(&best);
        for wi in affected {
            let (w, c) = &words[wi];
            let c = *c;
            // eski çiftleri düş
            for i in 0..w.len().saturating_sub(1) {
                let p = (w[i], w[i + 1]);
                if let Some(v) = pair_counts.get_mut(&p) {
                    *v = v.saturating_sub(c);
                }
            }
            // birleştir
            let mut out: Vec<u32> = Vec::with_capacity(w.len());
            let mut i = 0;
            while i < w.len() {
                if i + 1 < w.len() && w[i] == best.0 && w[i + 1] == best.1 {
                    out.push(new_id);
                    i += 2;
                } else {
                    out.push(w[i]);
                    i += 1;
                }
            }
            // yeni çiftleri ekle
            for i in 0..out.len().saturating_sub(1) {
                let p = (out[i], out[i + 1]);
                *pair_counts.entry(p).or_insert(0) += c;
                let v = pair_words.entry(p).or_default();
                if v.last() != Some(&wi) {
                    v.push(wi);
                }
            }
            words[wi].0 = out;
        }
    }

    Bpe { merges, lookup }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_ascii_and_unicode() {
        let corpus = "the cat sat on the mat. the cat ate. bir iki üç şğüöç. 🙂";
        let bpe = train_bpe(corpus, 300, 10_000);
        for s in ["the cat", "şğüöç", "🙂", "", "a", "mat. the"] {
            let ids = bpe.encode(s);
            assert_eq!(bpe.decode(&ids), s, "roundtrip bozuldu: {s:?}");
        }
    }

    #[test]
    fn compression_happens() {
        let corpus = "the cat sat on the mat. ".repeat(200);
        let bpe = train_bpe(&corpus, 320, 10_000);
        let ids = bpe.encode("the cat sat on the mat.");
        assert!(
            ids.len() < 10,
            "BPE sıkıştırma yapmadı: {} token",
            ids.len()
        );
    }

    #[test]
    fn deterministic() {
        let corpus = "abab abab abab cdcd";
        let a = train_bpe(corpus, 280, 1000);
        let b = train_bpe(corpus, 280, 1000);
        assert_eq!(a.merges, b.merges, "merge öğrenimi deterministik değil");
    }

    #[test]
    fn vocab_size_matches() {
        // Hedef 270 = 256 + 14 merge; korpus küçük olduğu için daha az merge
        // öğrenilebilir (birleştirilecek çift kalmazsa erken durur).
        let corpus = "abc abc abc def def";
        let bpe = train_bpe(corpus, 270, 1000);
        assert_eq!(bpe.vocab(), 256 + bpe.merges.len());
        assert!(bpe.merges.len() <= 14);
        assert!(!bpe.merges.is_empty(), "hiç merge öğrenilmedi");
    }
}
