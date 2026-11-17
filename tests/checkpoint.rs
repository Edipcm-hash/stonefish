// tests/checkpoint.rs
//
// Checkpoint v2 sözleşmesi:
//  * yaz → oku → aynı config, aynı vocab, birebir aynı ağırlıklar, aynı forward
//  * bozuk/yanlış dosya PANİK değil, io::Error döndürür
//  * vocab uyuşmazlığı kaydetmeyi reddeder (sessiz bozulma yok)

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
    Tokenizer::from_chars(chars).expect("vocab sıralı olmalı")
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

    let (loaded, tok2) = checkpoint::load_gpt(p).expect("yükleme");

    assert_eq!(loaded.config, cfg, "config geri gelmedi");
    assert_eq!(tok2.vocab_size, tok.vocab_size, "vocab boyutu");
    assert!(tok2.same_vocab(&tok), "vocab eşlemesi bozuldu");

    let a = model.parameters();
    let b = loaded.parameters();
    assert_eq!(a.len(), b.len(), "parametre sayısı");
    for (i, (pa, pb)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            pa.data.borrow().data,
            pb.data.borrow().data,
            "parametre {i} birebir aynı değil"
        );
    }

    // Aynı girdi → aynı çıktı
    let ids: Vec<usize> = vec![1, 3, 0, 2, 4, 7];
    let out_a = model.forward(&ids);
    let out_b = loaded.forward(&ids);
    assert_eq!(
        out_a.data.borrow().data,
        out_b.data.borrow().data,
        "forward çıktısı farklı"
    );

    let _ = fs::remove_file(path);
    println!(
        "  ✅ checkpoint roundtrip: config + vocab + {n} tensör birebir",
        n = a.len()
    );
}

#[test]
fn rejects_wrong_magic() {
    let path = tmp("magic");
    fs::write(&path, b"NOTACHECKPOINT_AT_ALL_XXXXXXXXXXXXXXXX").unwrap();
    let err = match checkpoint::load_gpt(path.to_str().unwrap()) {
        Ok(_) => panic!("bozuk dosya yüklenmemeliydi"),
        Err(e) => e,
    };
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    let _ = fs::remove_file(path);
    println!("  ✅ yanlış magic → InvalidData (panik yok)");
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
        "yarım dosya hata vermeliydi"
    );
    let _ = fs::remove_file(path);
    println!("  ✅ yarım dosya → hata (panik yok)");
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
        Ok(_) => panic!("vocab uyuşmazlığında kaydetmemeliydi"),
        Err(e) => e,
    };
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    let _ = fs::remove_file(path);
    println!("  ✅ vocab uyuşmazlığı kaydetmeyi reddediyor");
}

#[test]
fn tokenizer_sidecar_detects_drift() {
    let tok = vocab(6);
    let path = tmp("vocab_sidecar");
    let p = path.to_str().unwrap();
    tok.save(p).expect("vocab yazma");

    let loaded = Tokenizer::load(p).expect("vocab okuma");
    assert!(loaded.same_vocab(&tok), "yan dosya aynı vocab'ı vermeli");

    // Farklı bir korpus → farklı vocab → drift tespit edilmeli
    let drifted = Tokenizer::from_text("abcdefg");
    assert!(!loaded.same_vocab(&drifted), "drift tespit edilmedi");

    let _ = fs::remove_file(path);
    println!("  ✅ vocab yan dosyası drift tespit ediyor");
}
