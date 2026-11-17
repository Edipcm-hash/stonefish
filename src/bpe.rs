// src/bpe.rs
// Byte-level BPE tokenizer - training (merge learning) + encode/decode.
//
// Why BPE: at character level the model spends most of its capacity learning
// spelling ("once upon a time" = 17 separate generation steps). With byte-level
// BPE the same context window covers 4-5x more text.
//
// Design:
//  * Base: 256 byte symbols (every UTF-8 byte) -> no text can produce an
//    "unknown character" error, vocabulary starts at 256.
//  * Merge learning: split the corpus into words, merge the most frequent
//    adjacent pair over word frequencies (classic BPE). Byte boundaries kept.
//  * Complexity: O(length^2) scan per word, weighted by frequency.
//  * Deterministic: on ties the lower-index pair wins (reproducible).

use std::collections::HashMap;

/// A learned BPE model: 256 byte base + an ordered merge list.
#[derive(Clone, Debug)]
pub struct Bpe {
    /// merge id -> (left, right) symbol pair. Symbol ids: < 256 bytes, >= 256 merges.
    pub merges: Vec<(u32, u32)>,
    /// for lookup during encode: (left, right) -> new id
    lookup: HashMap<(u32, u32), u32>,
}

impl Bpe {
    /// Build from a merge list (checkpoint loading path).
    pub fn from_merges(merges: Vec<(u32, u32)>) -> Self {
        let mut lookup = HashMap::with_capacity(merges.len());
        for (i, &(a, b)) in merges.iter().enumerate() {
            lookup.insert((a, b), 256 + i as u32);
        }
        Self { merges, lookup }
    }

    /// Total vocabulary: 256 bytes + number of merges.
    pub fn vocab(&self) -> usize {
        256 + self.merges.len()
    }

    /// Initialize every byte as a symbol.
    pub fn byte_symbols(text: &str) -> Vec<u32> {
        text.bytes().map(|b| b as u32).collect()
    }

    /// Apply greedy BPE to a symbol sequence: at each step merge the pair with
    /// the lowest index (that is, learned earliest) present in the dictionary.
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

    /// Convert text to token ids.
    ///
    /// WARNING: calling this directly on long text is O(n^2) (the whole symbol
    /// vector is scanned for every merge). Corpus tokenization uses `encode_word`
    /// plus a word cache (see `corpus::tokenize_file_to_cache`).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len() / 3 + 4);
        for w in text.split_inclusive([' ', '\n']) {
            out.extend(self.encode_word(w));
        }
        out
    }

    /// BPE for a single "word" (including its trailing space/newline). O(L^2)
    /// where L is the word length - in practice <= 12.
    pub fn encode_word(&self, word: &str) -> Vec<u32> {
        let mut syms = Self::byte_symbols(word);
        self.encode_symbols(&mut syms);
        syms
    }

    /// Convert token ids back to text. Drops the partial byte if cut mid-UTF-8.
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
// Merge learning
// ============================================================

/// Learn BPE merges over a corpus.
///
/// `target_vocab`: stops when 256 + merges.len() reaches this value.
/// `max_words`: how many distinct words to use (memory/speed control).
///
/// Incremental pair counting is used: after each merge only the words that
/// contain that pair are rescanned. The naive approach (scanning the whole
/// corpus every step) took hours for 16k merges.
pub fn train_bpe(text: &str, target_vocab: usize, max_words: usize) -> Bpe {
    assert!(target_vocab > 256, "target_vocab must be > 256");

    // 1) Word frequencies. The trailing space/newline belongs to the word
    //    (GPT-2 style: " the" can be a single token).
    let mut freq: HashMap<Vec<u32>, u64> = HashMap::new();
    for word in text.split_inclusive([' ', '\n']) {
        let syms = Bpe::byte_symbols(word);
        if syms.is_empty() {
            continue;
        }
        *freq.entry(syms).or_insert(0) += 1;
    }

    // 2) The `max_words` most frequent words
    let mut words: Vec<(Vec<u32>, u64)> = freq.into_iter().collect();
    words.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    words.truncate(max_words);

    // 3) Incremental structures
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
        // Most frequent pair; on ties the smaller (a,b) wins -> deterministic.
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

        // Update only the words containing this pair
        let affected = pair_words.remove(&best).unwrap_or_default();
        pair_counts.remove(&best);
        for wi in affected {
            let (w, c) = &words[wi];
            let c = *c;
            // drop the old pairs
            for i in 0..w.len().saturating_sub(1) {
                let p = (w[i], w[i + 1]);
                if let Some(v) = pair_counts.get_mut(&p) {
                    *v = v.saturating_sub(c);
                }
            }
            // merge
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
            // add the new pairs
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
        let corpus = "the cat sat on the mat. the cat ate. un deux trois. sample text here. 🙂";
        let bpe = train_bpe(corpus, 300, 10_000);
        for s in ["the cat", "sample text", "🙂", "", "a", "mat. the"] {
            let ids = bpe.encode(s);
            assert_eq!(bpe.decode(&ids), s, "roundtrip bozuldu: {s:?}");
        }
    }

    #[test]
    fn compression_happens() {
        let corpus = "the cat sat on the mat. ".repeat(200);
        let bpe = train_bpe(&corpus, 320, 10_000);
        let ids = bpe.encode("the cat sat on the mat.");
        assert!(ids.len() < 10, "BPE did not compress: {} tokens", ids.len());
    }

    #[test]
    fn deterministic() {
        let corpus = "abab abab abab cdcd";
        let a = train_bpe(corpus, 280, 1000);
        let b = train_bpe(corpus, 280, 1000);
        assert_eq!(a.merges, b.merges, "merge learning is not deterministic");
    }

    #[test]
    fn vocab_size_matches() {
        // Target 270 = 256 + 14 merges; the corpus is small so fewer merges
        // can be learned (it stops early when no pair is left to merge).
        let corpus = "abc abc abc def def";
        let bpe = train_bpe(corpus, 270, 1000);
        assert_eq!(bpe.vocab(), 256 + bpe.merges.len());
        assert!(bpe.merges.len() <= 14);
        assert!(!bpe.merges.is_empty(), "no merges were learned");
    }
}
