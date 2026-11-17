// src/tokenizer.rs
// Karakter seviyesi tokenizer. Metin <-> ID.
//
// Vocab, eğitim korpusunun karakter kümesidir (sıralı). Checkpoint'e gömüldüğü
// için korpus dosyası değişse bile ID'ler kaymaz; `save`/`load` ikilisi ise
// yan dosya olarak vocab'ı taşıyıp drift tespiti yapmak için kullanılır.

use std::collections::{HashMap, HashSet};
use std::fs;

#[derive(Clone)]
pub struct Tokenizer {
    pub chars: Vec<char>,           // ID -> char
    pub stoi: HashMap<char, usize>, // char -> ID
    pub vocab_size: usize,
}

impl Tokenizer {
    /// Verilen metinden karakter setini çıkarır (sıralı → deterministik ID).
    pub fn from_text(text: &str) -> Self {
        let set: HashSet<char> = text.chars().collect();
        let mut chars: Vec<char> = set.into_iter().collect();
        chars.sort_unstable();
        Self::from_chars_unchecked(chars)
    }

    /// Sıralı karakter listesinden kurar (checkpoint yükleme).
    /// Liste sıralı ve tekrarsız değilse hata döner.
    pub fn from_chars(chars: Vec<char>) -> std::io::Result<Self> {
        let mut sorted = chars.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != chars.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "vocab içinde tekrarlanan karakter var",
            ));
        }
        if sorted != chars {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "vocab sıralı değil — ID eşlemesi bozulur",
            ));
        }
        Ok(Self::from_chars_unchecked(chars))
    }

    fn from_chars_unchecked(chars: Vec<char>) -> Self {
        let stoi: HashMap<char, usize> = chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        let vocab_size = chars.len();
        Self {
            chars,
            stoi,
            vocab_size,
        }
    }

    /// Metni ID vektörüne çevir. Bilinmeyen karakter → hata (sessiz bozulma yok).
    pub fn try_encode(&self, text: &str) -> Result<Vec<usize>, char> {
        let mut out = Vec::with_capacity(text.len());
        for c in text.chars() {
            match self.stoi.get(&c) {
                Some(&id) => out.push(id),
                None => return Err(c),
            }
        }
        Ok(out)
    }

    /// `try_encode`'in panikleyen sarmalayıcısı (CLI için pratik).
    pub fn encode(&self, text: &str) -> Vec<usize> {
        self.try_encode(text).unwrap_or_else(|c| {
            panic!(
                "bilinmeyen karakter: {:?} (U+{:04X}) — bu karakter eğitim korpusunda yok",
                c, c as u32
            )
        })
    }

    /// ID vektörünü metne çevir.
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .map(|&i| {
                *self
                    .chars
                    .get(i)
                    .unwrap_or_else(|| panic!("geçersiz token id: {i}"))
            })
            .collect()
    }

    /// Vocab'ı yan dosyaya yaz (ID sırasıyla, tek satır).
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        fs::write(path, self.chars.iter().collect::<String>())
    }

    /// Yan dosyadan vocab oku. `save` ile yazılmış olmalı.
    pub fn load(path: &str) -> std::io::Result<Self> {
        let s = fs::read_to_string(path)?;
        let mut chars: Vec<char> = s.chars().collect();
        chars.sort_unstable();
        chars.dedup();
        if chars.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "vocab dosyası boş",
            ));
        }
        Ok(Self::from_chars_unchecked(chars))
    }

    /// İki tokenizer aynı eşlemeyi mi kullanıyor? (korpus drift kontrolü)
    pub fn same_vocab(&self, other: &Self) -> bool {
        self.chars == other.chars
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let tok = Tokenizer::from_text("abcçdefgğ");
        let text = "çğğabc";
        assert_eq!(tok.decode(&tok.encode(text)), text);
    }

    #[test]
    fn unknown_char_is_error() {
        let tok = Tokenizer::from_text("abc");
        assert_eq!(tok.try_encode("abx"), Err('x'));
    }

    #[test]
    fn from_chars_validates_order() {
        assert!(Tokenizer::from_chars(vec!['b', 'a']).is_err());
        assert!(Tokenizer::from_chars(vec!['a', 'a']).is_err());
        assert!(Tokenizer::from_chars(vec!['a', 'b']).is_ok());
    }

    #[test]
    fn ids_are_deterministic() {
        // Aynı karakter kümesi, farklı sırayla yazılmış metin → aynı ID'ler.
        let a = Tokenizer::from_text("ab");
        let b = Tokenizer::from_text("ba");
        assert!(a.same_vocab(&b));
        assert_eq!(a.encode("a"), b.encode("a"));
    }
}
