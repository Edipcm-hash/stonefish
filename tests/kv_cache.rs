// tests/kv_cache.rs
//
// KV-cache'in TAM ileri geçişle aynı sonucu verdiğini doğrular.
// En kritik test: cache yolu ayrı bir kod yolu olduğu için (tek token, m=1
// GEMM'ler, kademeli K/V) sessizce sapabilir. Referans, doğrulanmış tam
// ileri geçiştir (`forward_logits_at`).

#![cfg(feature = "cuda")]

use stonefish_ai::gpu_train::GpuGpt;
use stonefish_ai::kv_cache::CachedModel;
use stonefish_ai::transformer::GptConfig;

fn rnd(n: usize, seed: &mut u64, vocab: u32) -> Vec<u32> {
    (0..n)
        .map(|_| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            (*seed % vocab as u64) as u32
        })
        .collect()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn kv_cache_matches_full_forward() {
    let cfg = GptConfig {
        vocab_size: 97,
        max_seq_len: 32,
        d_model: 64,
        n_layers: 2,
        n_heads: 4,
    };
    let seq = cfg.max_seq_len;
    let mut model = GpuGpt::new_random(cfg, 1, seq, 0xC0FFEE);
    model.training = false;
    model.dropout_p = 0.0;

    let mut s = 7u64;
    let tokens = rnd(20, &mut s, cfg.vocab_size as u32);

    // --- referans: TAM ileri geçiş (sağdan dolgu ile) ---
    // Son token'ın logitleri, prefix'in tamamı görülerek hesaplanır.
    let mut padded: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    while padded.len() < seq {
        padded.push(*padded.last().unwrap());
    }
    let full = model.forward_logits_at(&padded, tokens.len() - 1);

    // --- KV-cache: token token ---
    let mut cached = CachedModel::new(&mut model, seq);
    let mut last = Vec::new();
    for (pos, &t) in tokens.iter().enumerate() {
        last = cached.step(t, pos);
    }

    let d = max_diff(&full, &last);
    println!("  ✅ kv-cache ↔ tam ileri geçiş: max fark {d:.2e}");
    assert!(
        d < 3e-4,
        "KV-cache tam ileri geçişle uyuşmuyor: max fark {d} (ölçek {})",
        full.iter().map(|v| v.abs()).fold(0.0f32, f32::max)
    );
}

#[test]
fn kv_cache_prefix_logits_are_consistent() {
    // Her adımda cache'ten gelen logitler, aynı prefix'in tam ileri geçişiyle
    // uyuşmalı — yalnız son adım değil, tüm adımlar.
    let cfg = GptConfig {
        vocab_size: 61,
        max_seq_len: 16,
        d_model: 32,
        n_layers: 1,
        n_heads: 2,
    };
    let seq = cfg.max_seq_len;
    let mut model = GpuGpt::new_random(cfg, 1, seq, 0xBADF00D);
    model.training = false;

    let mut s = 11u64;
    let tokens = rnd(12, &mut s, cfg.vocab_size as u32);

    // Referansları ÖNCE hesapla (model ödünç alınmadan)
    let mut refs: Vec<Vec<f32>> = Vec::new();
    for pos in 0..tokens.len() {
        let mut padded: Vec<i32> = tokens[..=pos].iter().map(|&x| x as i32).collect();
        while padded.len() < seq {
            padded.push(*padded.last().unwrap());
        }
        refs.push(model.forward_logits_at(&padded, pos));
    }

    let mut cached = CachedModel::new(&mut model, seq);
    let mut worst = 0.0f32;
    for (pos, &t) in tokens.iter().enumerate() {
        let logits = cached.step(t, pos);
        worst = worst.max(max_diff(&refs[pos], &logits));
    }
    println!("  ✅ tüm prefix'lerde en kötü fark {worst:.2e}");
    assert!(worst < 3e-4, "prefix tutarsızlığı: {worst}");
}
