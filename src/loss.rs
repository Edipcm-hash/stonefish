// src/loss.rs
use crate::autograd::Tensor;

/// MSE: mean((pred - target)^2)
pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    let diff = pred.sub(target);
    let sq = diff.mul_elem(&diff);
    let n = sq.data.borrow().data.len() as f32;
    sq.sum().scale(1.0 / n)
}

/// Softmax + Cross-Entropy
pub fn softmax_cross_entropy(logits: &Tensor, onehot: &Tensor) -> Tensor {
    logits.softmax_cross_entropy(onehot)
}