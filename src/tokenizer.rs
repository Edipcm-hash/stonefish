// src/tokenizer.rs
// Karakter seviyesi tokenizer. Metin <-> ID.

use std::collections::HashMap;
use std::fs;

pub struct Tokenizer {
    pub chars: Vec<char>,          // ID -> char
    pub stoi: HashMap<char, usize>, // char -> ID
    pub vocab_size: usize,
}

impl Tokenizer {
    /// Verilen metinden karakter setini çıkarır.
    pub fn from_text(text: &str) -> Self {
        let mut set: Vec<char> = text.chars().collect::<std::collections::HashSet<_>>().into_iter().collect();
        set.sort();
        let stoi: HashMap<char, usize> = set.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        let vocab_size = set.len();
        Self { chars: set, stoi, vocab_size }
    }

    /// Metni ID vektörüne çevir.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars().map(|c| *self.stoi.get(&c).expect("bilinmeyen karakter")).collect()
    }

    /// ID vektörünü metne çevir.
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter().map(|&i| self.chars[i]).collect()
    }

    /// Tokenizer'ı dosyaya kaydet (vocab).
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let s: String = self.chars.iter().collect();
        fs::write(path, s)
    }

    /// Dosyadan yükle.
    pub fn load(path: &str) -> std::io::Result<Self> {
        let s = fs::read_to_string(path)?;
        Ok(Self::from_text(&s))
    }
}