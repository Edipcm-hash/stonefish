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
            eps: 1e-6,   // 1e-8 yerine 1e-6 (daha güvenli)
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
    }
}