// tests/checkpoint.rs
//
// The checkpoint v2 contract:
//  * write -> read -> same config, same vocab, identical weights, same forward
//  * a corrupt/wrong file returns an io::Error, not a PANIC
//  * a vocab mismatch refuses to save (no silent corruption)

use std::fs;
use stonefish_ai::checkpoint;
use stonefish_ai::tokenizer::Tokenizer;
use stonefish_ai::transformer::{GptConfig, GPT};

fn tmp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("stonefish_test_{name}_{}.bin", std::process::id()))
}

fn tiny_model(cfg: &GptConfig) -> GPT {
    let mut seed = 0x5EEDu64;
    GPT::new(*cfg, &mut seed)
}

fn vocab(n: usize) -> Tokenizer {
    let chars: Vec<char> = (0..n).map(|i| (b'a' + i as u8) as char).collect();
    Tokenizer::from_chars(chars).expect("the vocabulary must be sorted")
}

#[test]
fn roundtrip_preserves_everything() {
    let cfg = GptConfig {
        vocab_size: 8,
        max_seq_len: 6,
        d_model: 4,
        n_layers: 2,
        n_heads: 2,
    };
    let model = tiny_model(&cfg);
    let tok = vocab(8);
    let path = tmp("roundtrip");
    let p = path.to_str().unwrap();

    checkpoint::save_gpt(p, &model, &tok).expect("kaydetme");

    let (loaded, tok2) = checkpoint::load_gpt(p).expect("load");

    assert_eq!(loaded.config, cfg, "config geri gelmedi");
    assert_eq!(tok2.vocab_size, tok.vocab_size, "vocab boyutu");
    assert!(tok2.same_vocab(&tok), "the vocab mapping broke");

    let a = model.parameters();
    let b = loaded.parameters();
    assert_eq!(a.len(), b.len(), "parameter count");
    for (i, (pa, pb)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            pa.data.borrow().data,
            pb.data.borrow().data,
            "parameter {i} is not identical"
        );
    }

    // Same input -> same output
    let ids: Vec<usize> = vec![1, 3, 0, 2, 4, 7];
    let out_a = model.forward(&ids);
    let out_b = loaded.forward(&ids);
    assert_eq!(
        out_a.data.borrow().data,
        out_b.data.borrow().data,
        "the forward output differs"
    );

    let _ = fs::remove_file(path);
    println!(
        "  ✅ checkpoint roundtrip: config + vocab + {n} tensors identical",
        n = a.len()
    );
}

#[test]
fn rejects_wrong_magic() {
    let path = tmp("magic");
    fs::write(&path, b"NOTACHECKPOINT_AT_ALL_XXXXXXXXXXXXXXXX").unwrap();
    let err = match checkpoint::load_gpt(path.to_str().unwrap()) {
        Ok(_) => panic!("a corrupt file must not load"),
        Err(e) => e,
    };
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    let _ = fs::remove_file(path);
    println!("  ✅ wrong magic -> InvalidData (no panic)");
}

#[test]
fn rejects_truncated_file() {
    let cfg = GptConfig {
        vocab_size: 6,
        max_seq_len: 4,
        d_model: 4,
        n_layers: 1,
        n_heads: 1,
    };
    let model = tiny_model(&cfg);
    let tok = vocab(6);
    let path = tmp("truncated");
    let p = path.to_str().unwrap();
    checkpoint::save_gpt(p, &model, &tok).unwrap();

    let bytes = fs::read(p).unwrap();
    fs::write(p, &bytes[..bytes.len() / 2]).unwrap();
    assert!(
        checkpoint::load_gpt(p).is_err(),
        "a truncated file must fail"
    );
    let _ = fs::remove_file(path);
    println!("  ✅ truncated file -> error (no panic)");
}

#[test]
fn rejects_vocab_mismatch_on_save() {
    let cfg = GptConfig {
        vocab_size: 6,
        max_seq_len: 4,
        d_model: 4,
        n_layers: 1,
        n_heads: 1,
    };
    let model = tiny_model(&cfg);
    let wrong_tok = vocab(5); // 5 != 6
    let path = tmp("vocab_mismatch");
    let err = match checkpoint::save_gpt(path.to_str().unwrap(), &model, &wrong_tok) {
        Ok(_) => panic!("must not save on a vocab mismatch"),
        Err(e) => e,
    };
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    let _ = fs::remove_file(path);
    println!("  ✅ a vocab mismatch refuses to save");
}

#[test]
fn tokenizer_sidecar_detects_drift() {
    let tok = vocab(6);
    let path = tmp("vocab_sidecar");
    let p = path.to_str().unwrap();
    tok.save(p).expect("vocab yazma");

    let loaded = Tokenizer::load(p).expect("vocab okuma");
    assert!(
        loaded.same_vocab(&tok),
        "the sidecar must give the same vocab"
    );

    // A different corpus -> a different vocab -> drift must be detected
    let drifted = Tokenizer::from_text("abcdefg");
    assert!(!loaded.same_vocab(&drifted), "drift tespit edilmedi");

    let _ = fs::remove_file(path);
    println!("  ✅ the vocab sidecar detects drift");
}
