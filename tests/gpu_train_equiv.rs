// tests/gpu_train_equiv.rs
//
// Device-resident GPU eğiticisinin, doğrulanmış CPU autograd yoluna karşı
// eşdeğerliği:
//   1) aynı ağırlıklar, aynı batch → aynı loss
//   2) aynı gradyanlar (tüm parametreler)
//   3) N optimizer adımı sonrası aynı loss eğrisi
//
// CUDA çekirdekleri `use_fast_math` ile derlendiği için birebir bit eşitliği
// beklenmiyor; toleranslar ölçülen değerlere göre belgelenmiştir.

#![cfg(feature = "cuda")]

use stonefish_ai::gpu_train::GpuGpt;
use stonefish_ai::transformer::{GptConfig, GPT};

const LOSS_TOL: f32 = 2e-3;
const GRAD_TOL: f32 = 3e-3;

struct Case {
    model: GPT,
    ids: Vec<i32>,
    targets: Vec<i32>,
    batch: usize,
    seq: usize,
}

fn case() -> Case {
    let mut seed = 0xABCD_1234u64;
    let config = GptConfig {
        vocab_size: 7,
        max_seq_len: 4,
        d_model: 8,
        n_layers: 2,
        n_heads: 2,
    };
    let model = GPT::new(config, &mut seed);
    let (batch, seq) = (2usize, 4usize);
    let ids: Vec<usize> = vec![1, 3, 5, 0, 6, 2, 4, 1];
    let targets: Vec<usize> = vec![3, 5, 0, 6, 2, 4, 1, 0];
    assert_eq!(ids.len(), batch * seq);
    Case {
        model,
        ids: ids.iter().map(|&x| x as i32).collect(),
        targets: targets.iter().map(|&x| x as i32).collect(),
        batch,
        seq,
    }
}

/// CPU referansı: (loss, isim → gradyan)
fn cpu_forward_backward(c: &Case) -> (f32, Vec<(String, Vec<f32>)>) {
    use stonefish_ai::autograd::Tensor;

    let ids: Vec<usize> = c.ids.iter().map(|&x| x as usize).collect();
    let targets: Vec<usize> = c.targets.iter().map(|&x| x as usize).collect();
    let logits = c.model.forward_batch(&ids, c.batch, c.seq);
    let loss = logits.softmax_cross_entropy_ids(&targets);

    for p in c.model.parameters() {
        p.zero_grad();
    }
    loss.backward();
    let l = loss.item();

    let mut grads: Vec<(String, Vec<f32>)> = Vec::new();
    let g = |name: &str, t: &Tensor, out: &mut Vec<(String, Vec<f32>)>| {
        out.push((name.to_string(), t.grad.borrow().data.clone()));
    };
    g("tok_emb", &c.model.embedding.token_emb, &mut grads);
    for (i, b) in c.model.blocks.iter().enumerate() {
        g(&format!("norm1_g[{i}]"), &b.norm1_gamma, &mut grads);
        g(&format!("wq[{i}]"), &b.attn.w_q, &mut grads);
        g(&format!("wk[{i}]"), &b.attn.w_k, &mut grads);
        g(&format!("wv[{i}]"), &b.attn.w_v, &mut grads);
        g(&format!("wo[{i}]"), &b.attn.w_o, &mut grads);
        g(&format!("norm2_g[{i}]"), &b.norm2_gamma, &mut grads);
        g(&format!("ff_gate[{i}]"), &b.ff_gate, &mut grads);
        g(&format!("ff_up[{i}]"), &b.ff_up, &mut grads);
        g(&format!("ff_down[{i}]"), &b.ff_down, &mut grads);
    }
    g("lnf_g", &c.model.ln_f_gamma, &mut grads);
    (l, grads)
}

fn gpu_grad(g: &mut GpuGpt, name: &str) -> Vec<f32> {
    // "wq[2]" → ("wq", 2)
    if let Some((base, idx)) = name.split_once('[') {
        let i: usize = idx.trim_end_matches(']').parse().unwrap();
        g.grad_of(base, i).expect(name)
    } else {
        g.grad_of(name, usize::MAX).expect(name)
    }
}

#[test]
fn gpu_trainer_matches_cpu_gradients() {
    let c = case();
    let (loss_cpu, grads_cpu) = cpu_forward_backward(&c);

    let mut gpu = GpuGpt::from_cpu(&c.model, c.batch, c.seq);
    let loss_gpu = gpu.forward_backward(&c.ids, &c.targets);

    let d_loss = (loss_gpu - loss_cpu).abs();
    println!("  loss: cpu={loss_cpu:.6} gpu={loss_gpu:.6} fark={d_loss:.2e}");
    assert!(d_loss < LOSS_TOL, "loss farkı {d_loss}");

    let mut worst = 0.0f32;
    let mut worst_name = String::new();
    for (name, gc) in &grads_cpu {
        let gg = gpu_grad(&mut gpu, name);
        assert_eq!(gg.len(), gc.len(), "grad boyutu: {name}");
        for i in 0..gc.len() {
            let d = (gg[i] - gc[i]).abs();
            if d > worst {
                worst = d;
                worst_name = format!("{name}[{i}] cpu={} gpu={}", gc[i], gg[i]);
            }
        }
    }
    println!("  gradyanlar: en kötü |fark| = {worst:.2e}  ({worst_name})");
    assert!(worst < GRAD_TOL, "gradyan farkı {worst}: {worst_name}");
}

#[test]
fn gpu_trainer_matches_cpu_loss_curve() {
    // Beş Adam adımı: iki eğitici aynı loss eğrisini izlemeli.
    use stonefish_ai::optim::Adam;

    let c = case();
    let mut gpu = GpuGpt::from_cpu(&c.model, c.batch, c.seq);
    let cpu_model = c.model;
    let lr = 1e-3f32;
    let mut cpu_opt = Adam::new(cpu_model.parameters(), lr);

    let ids: Vec<usize> = c.ids.iter().map(|&x| x as usize).collect();
    let targets: Vec<usize> = c.targets.iter().map(|&x| x as usize).collect();

    for step in 0..5 {
        // CPU adımı (Adam)
        let logits = cpu_model.forward_batch(&ids, c.batch, c.seq);
        let loss_t = logits.softmax_cross_entropy_ids(&targets);
        let loss_cpu = loss_t.item();
        cpu_opt.zero_grad();
        loss_t.backward();
        cpu_opt.step();

        // GPU adımı (Adam, kırpma kapalı)
        let loss_gpu = gpu.forward_backward(&c.ids, &c.targets);
        gpu.step(lr, 0.0);

        let d = (loss_gpu - loss_cpu).abs();
        println!("  step {step}: cpu={loss_cpu:.6} gpu={loss_gpu:.6} fark={d:.2e}");
        assert!(
            d < LOSS_TOL * (1.0 + step as f32),
            "adım {step} loss farkı {d} (birikmiş olabilir)"
        );
    }
}
