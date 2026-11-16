// src/loss.rs
// Kayıp fonksiyonları. (MSE kaldırıldı: hiçbir yerde kullanılmıyordu — YAGNI.)

use crate::autograd::Tensor;

/// Softmax + Cross-Entropy (sayısal olarak kararlı, log-sum-exp ile).
pub fn softmax_cross_entropy(logits: &Tensor, onehot: &Tensor) -> Tensor {
    logits.softmax_cross_entropy(onehot)
}
