// src/probe.rs
// Evaluate a trained model on text it has never seen.
//
// The point is honesty about contamination: the training corpus is Public Domain
// books and TinyStories, so a held-out split of that corpus can only tell you so
// much. This module tokenizes an arbitrary UTF-8 file with an EXISTING BPE
// vocabulary and writes the u16 cache `en-eval` expects, so you can point the
// evaluation at a news article, your own writing, or anything written after the
// model was trained.
//
// Usage:
//   stonefish_ai en-tokenize data/en/probe.arwikibooks   (reads .txt, .bpe)
//   stonefish_ai en-eval data/en/probe stonefish_en.final.state 8

use std::path::Path;

use crate::bpe::Bpe;
use crate::corpus;
use crate::en_train::load_bpe;

/// Tokenizes `{prefix}.txt` with `{prefix}.bpe` into `{prefix}.valid.bin`.
///
/// Returns the token count. The BPE must already exist: deriving a new
/// vocabulary here would make the numbers incomparable with the trained model.
pub fn tokenize(prefix: &str) -> std::io::Result<u64> {
    let txt = format!("{prefix}.txt");
    let bpe_path = format!("{prefix}.bpe");
    let out = format!("{prefix}.valid.bin");

    if !Path::new(&txt).exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("missing {txt}"),
        ));
    }
    if !Path::new(&bpe_path).exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "missing {bpe_path} - copy the vocabulary the model was trained with,\n\
                 e.g. `copy data\\en\\tiny.bpe {bpe_path}`"
            ),
        ));
    }

    let bpe: Bpe = load_bpe(&bpe_path)?;
    println!("BPE         : {bpe_path} (vocab {})", bpe.vocab());
    println!("Text        : {txt}");
    println!("Cache       : {out}");

    let n = corpus::tokenize_file_to_cache(&txt, &out, &bpe, 1 << 20)?;

    let bytes = std::fs::metadata(&txt).map(|m| m.len()).unwrap_or(0);
    println!("Tokens      : {n}");
    if n > 0 {
        println!("Chars/token : {:.3}", bytes as f64 / n as f64);
        println!("\nEvaluate with:\n  stonefish_ai en-eval {prefix} stonefish_en.final.state 8");
    }
    Ok(n)
}
