// src/trainer.rs
// One training interface over two implementations.
//
// The GPU path is device-resident (`gpu_train::GpuGpt`: parameters, gradients
// and Adam state never leave the card). The CPU path uses the verified autograd
// engine. Both expose `train_step`, so the epoch loop - and the demo - are
// written once.
//
// This lives in the library rather than the binary so `demo` works on a machine
// without CUDA: the demo picks whichever trainer is available at runtime.

#[cfg(feature = "cuda")]
use crate::gpu::HAS_CUDA;
use crate::optim::{self, Adam};
use crate::transformer::{GptConfig, GPT};

/// Both trainers expose the same interface, so the loop is written once.
/// The weights are also kept in an "outer" model because the checkpoint format
/// is defined in terms of the CPU model.
pub enum Trainer {
    Cpu {
        model: Box<GPT>,
        opt: Box<Adam>,
    },
    #[cfg(feature = "cuda")]
    Gpu {
        g: Box<crate::gpu_train::GpuGpt>,
    },
}

impl Trainer {
    /// Picks the device-resident trainer when CUDA is present, the CPU one
    /// otherwise. `batch`/`seq` are ignored by the CPU variant.
    pub fn new(config: GptConfig, seed: u64, batch: usize, seq: usize) -> Self {
        #[cfg(feature = "cuda")]
        {
            if HAS_CUDA {
                let mut s = seed;
                let probe = GPT::new(config, &mut s);
                return Trainer::Gpu {
                    g: Box::new(crate::gpu_train::GpuGpt::from_cpu(&probe, batch, seq)),
                };
            }
        }
        let mut s = seed;
        let model = GPT::new(config, &mut s);
        let opt = Adam::new(model.parameters(), 1e-3);
        let _ = (batch, seq);
        Trainer::Cpu {
            model: Box::new(model),
            opt: Box::new(opt),
        }
    }

    /// True when the device-resident path is in use.
    pub fn is_gpu(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            matches!(self, Trainer::Gpu { .. })
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    /// A single training step. Returns (loss, global gradient norm before clipping).
    pub fn train_step(
        &mut self,
        ids: &[usize],
        targets: &[usize],
        bs: usize,
        seq: usize,
        lr: f32,
        clip: f32,
    ) -> (f32, f32) {
        match self {
            #[cfg(feature = "cuda")]
            Trainer::Gpu { g } => {
                let ids_i: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
                let tgt_i: Vec<i32> = targets.iter().map(|&x| x as i32).collect();
                let loss = g.forward_backward(&ids_i, &tgt_i);
                let norm = g.step(lr, clip);
                (loss, norm)
            }
            Trainer::Cpu { model, opt } => {
                opt.lr = lr;
                let logits = model.forward_batch(ids, bs, seq);
                let loss = logits.softmax_cross_entropy_ids(targets);
                opt.zero_grad();
                loss.backward();
                let norm = optim::clip_grad_norm(&opt.params, clip);
                opt.step();
                (loss.item(), norm)
            }
        }
    }

    /// Mean loss over `seqs`, without touching the gradients of the live model.
    pub fn eval_loss(
        &mut self,
        seqs: &[(Vec<usize>, Vec<usize>)],
        batch_size: usize,
        seq_len: usize,
    ) -> f64 {
        let mut sum = 0.0f64;
        let mut n = 0usize;
        let mut start = 0;
        while start < seqs.len() {
            let end = (start + batch_size).min(seqs.len());
            let bs = end - start;
            let mut ids = Vec::with_capacity(bs * seq_len);
            let mut tgt = Vec::with_capacity(bs * seq_len);
            for k in start..end {
                let (a, b) = &seqs[k];
                ids.extend_from_slice(a);
                tgt.extend_from_slice(b);
            }
            let l = match self {
                #[cfg(feature = "cuda")]
                Trainer::Gpu { g } => {
                    let ids_i: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
                    let tgt_i: Vec<i32> = tgt.iter().map(|&x| x as i32).collect();
                    g.forward_loss(&ids_i, &tgt_i) as f64
                }
                Trainer::Cpu { model, .. } => {
                    let logits = model.forward_batch(&ids, bs, seq_len);
                    logits.softmax_cross_entropy_ids(&tgt).item() as f64
                }
            };
            sum += l;
            n += 1;
            start = end;
        }
        sum / n.max(1) as f64
    }

    /// Copies the trained weights into the outer (checkpoint) model.
    pub fn sync_to_cpu(&mut self, out: &mut GPT) {
        match self {
            #[cfg(feature = "cuda")]
            Trainer::Gpu { g } => g.copy_to_cpu(out),
            Trainer::Cpu { model, .. } => {
                for (src, dst) in model.parameters().iter().zip(out.parameters().iter()) {
                    let data = src.data.borrow().data.clone();
                    dst.data.borrow_mut().data.copy_from_slice(&data);
                }
            }
        }
    }
}

/// Linear warmup then cosine decay - the schedule used by the demo and by the
/// English trainer. Lives here so the demo can use it without CUDA.
pub fn lr_at(step: usize, total: usize, max_lr: f32, warmup: usize, min_ratio: f32) -> f32 {
    if step < warmup {
        return max_lr * (step + 1) as f32 / warmup.max(1) as f32;
    }
    let span = total.saturating_sub(warmup).max(1) as f32;
    let t = ((step - warmup) as f32 / span).min(1.0);
    let cos = 0.5 * (1.0 + (std::f32::consts::PI * t).cos());
    max_lr * (min_ratio + (1.0 - min_ratio) * cos)
}

#[cfg(test)]
mod tests {
    use super::lr_at;

    const MAX: f32 = 4e-4;
    const WARMUP: usize = 100;
    const TOTAL: usize = 1100;
    const MIN_RATIO: f32 = 0.1;

    /// The warmup is linear and reaches max_lr exactly at the last warmup step.
    #[test]
    fn warmup_is_linear_and_ends_at_max() {
        assert!(lr_at(0, TOTAL, MAX, WARMUP, MIN_RATIO) < MAX * 0.02);
        let mid = lr_at(WARMUP / 2, TOTAL, MAX, WARMUP, MIN_RATIO);
        let first = lr_at(0, TOTAL, MAX, WARMUP, MIN_RATIO);
        assert!(mid > first, "warmup must increase");
        // step+1 scaling: the last warmup step is exactly max_lr
        let last = lr_at(WARMUP - 1, TOTAL, MAX, WARMUP, MIN_RATIO);
        assert!(
            (last - MAX).abs() < MAX * 1e-5,
            "last warmup step is {last}"
        );
    }

    /// The decay is monotone after warmup and never drops below min_ratio*max_lr.
    #[test]
    fn cosine_decays_monotonically_to_the_floor() {
        let mut prev = lr_at(WARMUP, TOTAL, MAX, WARMUP, MIN_RATIO);
        for step in (WARMUP + 1)..TOTAL {
            let lr = lr_at(step, TOTAL, MAX, WARMUP, MIN_RATIO);
            assert!(
                lr <= prev + 1e-9,
                "not monotone at step {step}: {lr} > {prev}"
            );
            assert!(
                lr >= MAX * MIN_RATIO - 1e-9,
                "below the floor at step {step}"
            );
            prev = lr;
        }
        let last = lr_at(TOTAL - 1, TOTAL, MAX, WARMUP, MIN_RATIO);
        assert!((last - MAX * MIN_RATIO).abs() < MAX * 0.01, "end is {last}");
    }

    /// Past the last step the schedule must stay at the floor, not go negative
    /// (the cosine formula would otherwise wrap around).
    #[test]
    fn past_the_end_it_holds_the_floor() {
        let floor = MAX * MIN_RATIO;
        for step in [TOTAL, TOTAL + 500, TOTAL * 3] {
            let lr = lr_at(step, TOTAL, MAX, WARMUP, MIN_RATIO);
            assert!((lr - floor).abs() < MAX * 1e-4, "step {step} gave {lr}");
        }
    }
}
