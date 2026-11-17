// tests/kv_cache.rs
//
// Verifies that the KV-cache gives the same result as the FULL forward pass.
// The most critical test: the cache path is a separate code path (single
// token, m=1 GEMMs, incremental K/V) that can drift silently. The reference
// is the verified full forward pass (`forward_logits_at`).

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

    // --- reference: FULL forward pass (with right padding) ---
    // The last token's logits are computed seeing the whole prefix.
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
    println!("  ✅ kv-cache <-> full forward pass: max diff {d:.2e}");
    assert!(
        d < 3e-4,
        "the KV-cache does not match the full forward pass: max diff {d} (scale {})",
        full.iter().map(|v| v.abs()).fold(0.0f32, f32::max)
    );
}

#[test]
fn kv_cache_prefix_logits_are_consistent() {
    // The logits from the cache must match the full forward pass of the same
    // prefix at every step - not only the last one.
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

    // Compute the references FIRST (before the model is borrowed)
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
    println!("  ✅ worst diff over all prefixes {worst:.2e}");
    assert!(worst < 3e-4, "prefix inconsistency: {worst}");
}
