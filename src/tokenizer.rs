// src/tokenizer.rs
// Character-level tokenizer: text <-> ID.
//
// The vocab is the character set of the training corpus (sorted). Because it is
// embedded in the checkpoint the ids never shift even if the corpus file
// changes; the `save`/`load` pair carries it as a sidecar to detect drift.

use std::collections::{HashMap, HashSet};
use std::fs;

#[derive(Clone)]
pub struct Tokenizer {
    pub chars: Vec<char>,           // ID -> char
    pub stoi: HashMap<char, usize>, // char -> ID
    pub vocab_size: usize,
}

impl Tokenizer {
    /// Extracts the character set from the text (sorted -> deterministic ids).
    pub fn from_text(text: &str) -> Self {
        let set: HashSet<char> = text.chars().collect();
        let mut chars: Vec<char> = set.into_iter().collect();
        chars.sort_unstable();
        Self::from_chars_unchecked(chars)
    }

    /// Builds from a sorted character list (checkpoint loading).
    /// Returns an error if the list is not sorted and unique.
    pub fn from_chars(chars: Vec<char>) -> std::io::Result<Self> {
        let mut sorted = chars.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != chars.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "the vocabulary contains a repeated character",
            ));
        }
        if sorted != chars {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "the vocabulary is not sorted - id mapping would break",
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

    /// Convert text to an id vector. Unknown character -> error (no silent corruption).
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

    /// Panicking wrapper around `try_encode` (handy for the CLI).
    pub fn encode(&self, text: &str) -> Vec<usize> {
        self.try_encode(text).unwrap_or_else(|c| {
            panic!(
                "unknown character: {:?} (U+{:04X}) - this character is not in the training corpus",
                c, c as u32
            )
        })
    }

    /// Convert an id vector back to text.
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .map(|&i| {
                *self
                    .chars
                    .get(i)
                    .unwrap_or_else(|| panic!("invalid token id: {i}"))
            })
            .collect()
    }

    /// Write the vocabulary to a sidecar file (in id order, one line).
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        fs::write(path, self.chars.iter().collect::<String>())
    }

    /// Read the vocabulary from a sidecar file. It must have been written by `save`.
    pub fn load(path: &str) -> std::io::Result<Self> {
        let s = fs::read_to_string(path)?;
        let mut chars: Vec<char> = s.chars().collect();
        chars.sort_unstable();
        chars.dedup();
        if chars.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "the vocabulary file is empty",
            ));
        }
        Ok(Self::from_chars_unchecked(chars))
    }

    /// Do both tokenizers use the same mapping? (corpus drift check)
    pub fn same_vocab(&self, other: &Self) -> bool {
        self.chars == other.chars
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let tok = Tokenizer::from_text("abcdefghij");
        let text = "ijjabc";
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
        // Same character set in a differently ordered text -> the same ids.
        let a = Tokenizer::from_text("ab");
        let b = Tokenizer::from_text("ba");
        assert!(a.same_vocab(&b));
        assert_eq!(a.encode("a"), b.encode("a"));
    }
}
