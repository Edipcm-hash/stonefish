// src/optim.rs
use crate::autograd::Tensor;
use crate::tensor::Matrix;

// ============================================================
// SGD
// ============================================================

pub struct SGD {
    pub params: Vec<Tensor>,
    pub lr: f32,
}

impl SGD {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        Self { params, lr }
    }

    pub fn step(&self) {
        for p in &self.params {
            p.apply_grad(self.lr);
        }
    }

    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }
}

// ============================================================
// Gradyan kırpma
// ============================================================

/// Global L2 norm kırpma: tüm parametrelerin gradyanlarını tek bir vektör gibi
/// düşünüp normu `max_norm` ile sınırlar. Norm `max_norm`'u aşmıyorsa hiçbir
/// şeye dokunmaz. Kırpma **öncesi** normu döndürür (eğitim loglarında izlenir;
/// patlayan gradyanı kırpmadan önce görmek istiyoruz).
///
/// Toplam f64'te biriktirilir: 842k parametrede f32 toplamı taşma/incelik
/// kaybına açıktır.
pub fn clip_grad_norm(params: &[Tensor], max_norm: f32) -> f32 {
    let mut sq = 0.0f64;
    for p in params {
        let g = p.grad.borrow();
        for &v in &g.data {
            sq += (v as f64) * (v as f64);
        }
    }
    let norm = sq.sqrt() as f32;
    if norm > max_norm && norm > 0.0 {
        let scale = max_norm / norm;
        for p in params {
            let mut g = p.grad.borrow_mut();
            for v in g.data.iter_mut() {
                *v *= scale;
            }
        }
    }
    norm
}

// ============================================================
// Adam (düzeltilmiş)
// ============================================================

pub struct Adam {
    pub params: Vec<Tensor>,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub t: usize,
    m: Vec<Matrix>,
    v: Vec<Matrix>,
}

impl Adam {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        let mut m = Vec::with_capacity(params.len());
        let mut v = Vec::with_capacity(params.len());
        for p in &params {
            let d = p.data.borrow();
            m.push(Matrix::zeros(d.rows, d.cols));
            v.push(Matrix::zeros(d.rows, d.cols));
        }
        Self {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-6, // 1e-8 yerine 1e-6 (daha güvenli)
            t: 0,
            m,
            v,
        }
    }

    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }

    pub fn step(&mut self) {
        let prof_t = crate::profile::start();
        self.t += 1;
        let t = self.t as f32;

        // Bias correction (her adımda güncellenir)
        let bc1 = 1.0 - self.beta1.powf(t);
        let bc2 = 1.0 - self.beta2.powf(t);

        for (i, p) in self.params.iter().enumerate() {
            let g = p.grad.borrow();
            let mut d = p.data.borrow_mut();

            for k in 0..d.data.len() {
                let gk = g.data[k];
                // m = beta1*m + (1-beta1)*g
                self.m[i].data[k] = self.beta1 * self.m[i].data[k] + (1.0 - self.beta1) * gk;
                // v = beta2*v + (1-beta2)*g^2
                self.v[i].data[k] = self.beta2 * self.v[i].data[k] + (1.0 - self.beta2) * gk * gk;

                // Bias-corrected momentler
                let m_hat = self.m[i].data[k] / bc1;
                let v_hat = self.v[i].data[k] / bc2;

                // param -= lr * m_hat / (sqrt(v_hat) + eps)
                d.data[k] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }
        }
        crate::profile::stop(prof_t, &crate::profile::ADAM_NANOS);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::Tensor;

    fn leaf_with_grad(rows: usize, cols: usize, grads: Vec<f32>) -> Tensor {
        let t = Tensor::leaf(Matrix::zeros(rows, cols), true);
        *t.grad.borrow_mut() = Matrix::from_vec(rows, cols, grads);
        t
    }

    #[test]
    fn clip_leaves_small_grads_untouched() {
        // norm = sqrt(3² + 4²) = 5 < max_norm=10 → dokunulmamalı
        let t = leaf_with_grad(1, 2, vec![3.0, 4.0]);
        let before = t.grad.borrow().data.clone();
        let norm = clip_grad_norm(std::slice::from_ref(&t), 10.0);
        assert!((norm - 5.0).abs() < 1e-6, "norm {norm}");
        assert_eq!(t.grad.borrow().data, before);
    }

    #[test]
    fn clip_scales_to_max_norm() {
        // norm = 5 → max_norm=1 ile ölçek 0.2; yön korunur.
        let t = leaf_with_grad(1, 2, vec![3.0, 4.0]);
        let norm = clip_grad_norm(std::slice::from_ref(&t), 1.0);
        assert!(
            (norm - 5.0).abs() < 1e-6,
            "kırpma öncesi norm dönmeli: {norm}"
        );
        let g = t.grad.borrow();
        assert!((g.get(0, 0) - 0.6).abs() < 1e-6, "g0={}", g.get(0, 0));
        assert!((g.get(0, 1) - 0.8).abs() < 1e-6, "g1={}", g.get(0, 1));
        let new_norm = (0.6f32 * 0.6 + 0.8 * 0.8).sqrt();
        assert!((new_norm - 1.0).abs() < 1e-5, "yeni norm {new_norm}");
    }

    #[test]
    fn clip_uses_global_norm_across_params() {
        // İki parametre birlikte norm 5 eder; tek başına hiçbiri sınırı aşmaz.
        let a = leaf_with_grad(1, 1, vec![3.0]);
        let b = leaf_with_grad(1, 1, vec![4.0]);
        let norm = clip_grad_norm(&[a.clone(), b.clone()], 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((a.grad.borrow().get(0, 0) - 1.5).abs() < 1e-6);
        assert!((b.grad.borrow().get(0, 0) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn clip_handles_zero_grads() {
        let t = leaf_with_grad(1, 3, vec![0.0, 0.0, 0.0]);
        let norm = clip_grad_norm(std::slice::from_ref(&t), 1.0);
        assert_eq!(norm, 0.0);
        assert_eq!(t.grad.borrow().data, vec![0.0, 0.0, 0.0]);
    }
}
