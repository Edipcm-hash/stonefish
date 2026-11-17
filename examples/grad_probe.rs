// examples/grad_probe.rs
// Diagnostic tool: prints the analytic and numeric gradient of a single
// parameter of a small GPT side by side. (For hunting gradient bugs.)
#![allow(clippy::needless_range_loop)]

use stonefish_ai::transformer::{GptConfig, GPT};

const EPS: f32 = 1e-3;

fn loss_of(model: &GPT, ids: &[usize], tgt: &[usize], batch: usize, seq: usize) -> f32 {
    model
        .forward_batch(ids, batch, seq)
        .softmax_cross_entropy_ids(tgt)
        .item()
}

fn probe(label: &str, batch: usize, seq: usize, skip_attention: bool, emb_scale: f32) {
    let mut seed = 31u64;
    let config = GptConfig {
        vocab_size: 5,
        max_seq_len: 4,
        d_model: 4,
        n_layers: if skip_attention { 0 } else { 1 },
        n_heads: 2,
    };
    let model = GPT::new(config, &mut seed);
    if emb_scale != 1.0 {
        let p = &model.embedding.token_emb;
        let d = p.data.borrow().scale(emb_scale);
        *p.data.borrow_mut() = d;
    }
    let params = model.parameters();

    let mut ids: Vec<usize> = vec![1, 3, 0, 2, 4, 1];
    let mut tgt: Vec<usize> = vec![2, 4, 1, 0, 3, 2];
    ids.truncate(batch * seq);
    tgt.truncate(batch * seq);

    // analitik
    for p in &params {
        p.zero_grad();
    }
    let l = model
        .forward_batch(&ids, batch, seq)
        .softmax_cross_entropy_ids(&tgt);
    println!(
        "\n=== {label} (B={batch}, S={seq}) loss={:.6} ===",
        l.item()
    );
    l.backward();

    // v2 parameter order: embedding -> each block (norm1, wq/wk/wv/wo, norm2,
    // ff_gate/ff_up/ff_down) -> final norm. Same order as `GPT::parameters()`.
    let names = [
        "token_emb",
        "norm1_g",
        "w_q",
        "w_k",
        "w_v",
        "w_o",
        "norm2_g",
        "ff_gate",
        "ff_up",
        "ff_down",
        "lnf_gamma",
    ];

    for (pi, p) in params.iter().enumerate() {
        let name = names.get(pi).copied().unwrap_or("?");
        let analytic = p.grad.borrow().data.clone();
        let n = p.data.borrow().data.len();
        let mut worst = 0.0f32;
        let mut worst_detail = String::new();
        for k in 0..n {
            let orig = p.data.borrow().data[k];
            p.data.borrow_mut().data[k] = orig + EPS;
            let lp = loss_of(&model, &ids, &tgt, batch, seq);
            p.data.borrow_mut().data[k] = orig - EPS;
            let lm = loss_of(&model, &ids, &tgt, batch, seq);
            p.data.borrow_mut().data[k] = orig;
            let numeric = (lp - lm) / (2.0 * EPS);
            let rel = (analytic[k] - numeric).abs() / analytic[k].abs().max(numeric.abs()).max(1.0);
            if rel > worst {
                worst = rel;
                worst_detail = format!("k={k} analitik={:.6} numerik={:.6}", analytic[k], numeric);
            }
        }
        let flag = if worst > 2e-2 { "❌" } else { "✅" };
        println!(
            "{flag} {name:<12} ({:>2}x{:<2}) worst rel={:.2e}  {worst_detail}",
            p.data.borrow().rows,
            p.data.borrow().cols,
            worst
        );
    }
}

fn main() {
    probe("tek sequence", 1, 3, false, 1.0);
    probe("batch=2", 2, 3, false, 1.0);
    probe("batch=2, emb×50", 2, 3, false, 50.0);
    probe("batch=2, emb x50, no attn", 2, 3, true, 50.0);
}
