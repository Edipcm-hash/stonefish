// src/loss.rs
// Loss functions. (MSE was removed: it was never used - YAGNI.)

use crate::autograd::Tensor;

/// Softmax + cross-entropy (numerically stable, via log-sum-exp).
pub fn softmax_cross_entropy(logits: &Tensor, onehot: &Tensor) -> Tensor {
    logits.softmax_cross_entropy(onehot)
}
