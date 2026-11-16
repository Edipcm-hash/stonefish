// src/autograd.rs
// StonefishAI — autograd + Softmax + GELU + LayerNorm + Embedding + MaskCausal.
// Düzeltmeler: SoftmaxCE'de 1/batch, LayerNorm'da eps, HashSet topo, cache.

use crate::tensor::Matrix;
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

#[derive(Clone, Copy, Debug)]
pub enum Op {
    Leaf,
    Add,
    Sub,
    MulElem,
    MatMul,
    Scale(f32),
    ReLU,
    Sigmoid,
    Transpose,
    Sum,
    SoftmaxCE,
    Softmax,
    GELU,
    LayerNorm(f32),   // eps
    Embedding,
    MaskCausal,
}

pub struct Node {
    pub op: Op,
    pub inputs: Vec<Tensor>,
    /// Forward sırasında hesaplanan ara değerler (backward için cache)
    pub cache: Vec<Matrix>,
}

#[derive(Clone)]
pub struct Tensor {
    pub data: Rc<RefCell<Matrix>>,
    pub grad: Rc<RefCell<Matrix>>,
    pub requires_grad: bool,
    pub node: Option<Rc<Node>>,
}

impl Tensor {
    pub fn leaf(data: Matrix, requires_grad: bool) -> Self {
        let grad = Rc::new(RefCell::new(Matrix::zeros(data.rows, data.cols)));
        Self {
            data: Rc::new(RefCell::new(data)),
            grad,
            requires_grad,
            node: None,
        }
    }

    pub fn from_op(data: Matrix, op: Op, inputs: Vec<Tensor>) -> Self {
        Self::from_op_cached(data, op, inputs, Vec::new())
    }

    pub fn from_op_cached(
        data: Matrix,
        op: Op,
        inputs: Vec<Tensor>,
        cache: Vec<Matrix>,
    ) -> Self {
        let requires_grad = inputs.iter().any(|t| t.requires_grad);
        let grad = Rc::new(RefCell::new(Matrix::zeros(data.rows, data.cols)));
        let node = if requires_grad {
            Some(Rc::new(Node { op, inputs, cache }))
        } else {
            None
        };
        Self {
            data: Rc::new(RefCell::new(data)),
            grad,
            requires_grad,
            node,
        }
    }

    pub fn zero_grad(&self) {
        let d = self.data.borrow();
        *self.grad.borrow_mut() = Matrix::zeros(d.rows, d.cols);
    }

    pub fn apply_grad(&self, lr: f32) {
        let g = self.grad.borrow();
        let mut d = self.data.borrow_mut();
        assert_eq!((d.rows, d.cols), (g.rows, g.cols), "apply_grad boyut");
        for i in 0..d.data.len() {
            d.data[i] -= lr * g.data[i];
        }
    }

    // ---- İleri ----

    pub fn add(&self, other: &Tensor) -> Tensor {
        let data = self.data.borrow().add(&other.data.borrow());
        Tensor::from_op(data, Op::Add, vec![self.clone(), other.clone()])
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        let data = self.data.borrow().sub(&other.data.borrow());
        Tensor::from_op(data, Op::Sub, vec![self.clone(), other.clone()])
    }

    pub fn mul_elem(&self, other: &Tensor) -> Tensor {
        let data = self.data.borrow().mul_elem(&other.data.borrow());
        Tensor::from_op(data, Op::MulElem, vec![self.clone(), other.clone()])
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let a = self.data.borrow();
        let b = other.data.borrow();
        let data = crate::gpu::matmul_gpu(&a, &b);
        drop(a);
        drop(b);
        Tensor::from_op(data, Op::MatMul, vec![self.clone(), other.clone()])
    }

    pub fn scale(&self, s: f32) -> Tensor {
        let data = self.data.borrow().scale(s);
        Tensor::from_op(data, Op::Scale(s), vec![self.clone()])
    }

    pub fn relu(&self) -> Tensor {
        let data = self.data.borrow().map(|x| if x > 0.0 { x } else { 0.0 });
        Tensor::from_op(data, Op::ReLU, vec![self.clone()])
    }

    pub fn sigmoid(&self) -> Tensor {
        let d = self.data.borrow();
        let s = d.map(|x| 1.0 / (1.0 + (-x).exp()));
        drop(d);
        Tensor::from_op_cached(
            s.clone(),
            Op::Sigmoid,
            vec![self.clone()],
            vec![s],  // cache: sigmoid(x)
        )
    }

    pub fn transpose(&self) -> Tensor {
        let data = self.data.borrow().transpose();
        Tensor::from_op(data, Op::Transpose, vec![self.clone()])
    }

    pub fn sum(&self) -> Tensor {
        let s: f32 = self.data.borrow().data.iter().sum();
        let data = Matrix::from_vec(1, 1, vec![s]);
        Tensor::from_op(data, Op::Sum, vec![self.clone()])
    }

    pub fn softmax_cross_entropy(&self, onehot: &Tensor) -> Tensor {
        let zd = self.data.borrow();
        let td = onehot.data.borrow();
        assert_eq!((zd.rows, zd.cols), (td.rows, td.cols), "CE boyut");

        let batch = zd.rows;
        let classes = zd.cols;
        let mut total_loss = 0.0f32;
        let mut probs = Matrix::zeros(batch, classes);  // cache: softmax

        for i in 0..batch {
            let mut max = f32::NEG_INFINITY;
            for j in 0..classes {
                let v = zd.get(i, j);
                if v > max { max = v; }
            }
            let mut sum_exp = 0.0f32;
            for j in 0..classes {
                sum_exp += (zd.get(i, j) - max).exp();
            }
            let log_sum_exp = max + sum_exp.ln();
            for j in 0..classes {
                let p = (zd.get(i, j) - max).exp() / sum_exp;
                probs.set(i, j, p);
                if td.get(i, j) > 0.5 {
                    total_loss += -zd.get(i, j) + log_sum_exp;
                }
            }
        }

        let mean_loss = total_loss / batch as f32;
        let data = Matrix::from_vec(1, 1, vec![mean_loss]);
        drop(zd);
        drop(td);
        Tensor::from_op_cached(
            data,
            Op::SoftmaxCE,
            vec![self.clone(), onehot.clone()],
            vec![probs],  // cache: softmax probs
        )
    }

    pub fn softmax(&self) -> Tensor {
        let zd = self.data.borrow();
        let (rows, cols) = (zd.rows, zd.cols);
        let mut data = Matrix::zeros(rows, cols);
        for i in 0..rows {
            let mut max = f32::NEG_INFINITY;
            for j in 0..cols {
                let v = zd.get(i, j);
                if v > max { max = v; }
            }
            let mut sum = 0.0f32;
            for j in 0..cols {
                sum += (zd.get(i, j) - max).exp();
            }
            for j in 0..cols {
                data.set(i, j, (zd.get(i, j) - max).exp() / sum);
            }
        }
        drop(zd);
        Tensor::from_op_cached(
            data.clone(),
            Op::Softmax,
            vec![self.clone()],
            vec![data],  // cache: softmax
        )
    }

    pub fn gelu(&self) -> Tensor {
        let data = self.data.borrow().map(|x| {
            let c = (2.0f32 / std::f32::consts::PI).sqrt();
            0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
        });
        Tensor::from_op(data, Op::GELU, vec![self.clone()])
    }

    pub fn layer_norm(&self, gamma: &Tensor, beta: &Tensor, eps: f32) -> Tensor {
        let xd = self.data.borrow();
        let (rows, cols) = (xd.rows, xd.cols);
        let gd = gamma.data.borrow();
        let bd = beta.data.borrow();
        assert_eq!(gd.cols, cols);
        assert_eq!(bd.cols, cols);

        let mut data = Matrix::zeros(rows, cols);
        let mut x_hat_cache = Matrix::zeros(rows, cols);
        let mut inv_cache = Matrix::zeros(rows, 1);

        for i in 0..rows {
            let mut mean = 0.0f32;
            for j in 0..cols { mean += xd.get(i, j); }
            mean /= cols as f32;

            let mut var = 0.0f32;
            for j in 0..cols {
                let d = xd.get(i, j) - mean;
                var += d * d;
            }
            var /= cols as f32;

            let inv = 1.0 / (var + eps).sqrt();
            inv_cache.set(i, 0, inv);
            for j in 0..cols {
                let xn = (xd.get(i, j) - mean) * inv;
                x_hat_cache.set(i, j, xn);
                data.set(i, j, xn * gd.get(0, j) + bd.get(0, j));
            }
        }
        drop(xd);
        drop(gd);
        drop(bd);

        Tensor::from_op_cached(
            data,
            Op::LayerNorm(eps),
            vec![self.clone(), gamma.clone(), beta.clone()],
            vec![x_hat_cache, inv_cache],  // cache: x_hat, inv_std
        )
    }

    pub fn mask_causal(&self) -> Tensor {
        let d = self.data.borrow();
        let (rows, cols) = (d.rows, d.cols);
        assert_eq!(rows, cols, "causal mask kare matris ister");
        let mut out = d.clone();
        for i in 0..rows {
            for j in 0..cols {
                if j > i {
                    out.set(i, j, -1e9);
                }
            }
        }
        drop(d);
        Tensor::from_op(out, Op::MaskCausal, vec![self.clone()])
    }

    // ---- Backward ----

    pub fn backward(&self) {
        let mut topo: Vec<Tensor> = Vec::new();
        let mut visited: HashSet<*const Node> = HashSet::new();
        self.build_topo(&mut topo, &mut visited);

        {
            let d = self.data.borrow();
            *self.grad.borrow_mut() = Matrix::filled(d.rows, d.cols, 1.0);
        }

        for t in topo.iter().rev() {
            let node = match &t.node {
                Some(n) => n.clone(),
                None => continue,
            };
            let upstream = t.grad.borrow().clone();
            apply_backward(&node, &upstream);
        }
    }

    fn build_topo(&self, topo: &mut Vec<Tensor>, visited: &mut HashSet<*const Node>) {
        if let Some(node) = &self.node {
            let ptr = Rc::as_ptr(node);
            if !visited.insert(ptr) { return; }
            for inp in &node.inputs {
                inp.build_topo(topo, visited);
            }
        }
        topo.push(self.clone());
    }
}

fn apply_backward(node: &Node, upstream: &Matrix) {
    match node.op {
        Op::Leaf => {}

        Op::Add => {
            let a = &node.inputs[0];
            let b = &node.inputs[1];
            let ga = reduce_to(upstream, a.data.borrow().rows, a.data.borrow().cols);
            let gb = reduce_to(upstream, b.data.borrow().rows, b.data.borrow().cols);
            accumulate(a, &ga);
            accumulate(b, &gb);
        }

        Op::Sub => {
            let a = &node.inputs[0];
            let b = &node.inputs[1];
            let ga = reduce_to(upstream, a.data.borrow().rows, a.data.borrow().cols);
            let gb = reduce_to(upstream, b.data.borrow().rows, b.data.borrow().cols).scale(-1.0);
            accumulate(a, &ga);
            accumulate(b, &gb);
        }

        Op::MulElem => {
            let a = &node.inputs[0];
            let b = &node.inputs[1];
            let ga = upstream.mul_elem(&b.data.borrow());
            let gb = upstream.mul_elem(&a.data.borrow());
            accumulate(a, &ga);
            accumulate(b, &gb);
        }

        Op::MatMul => {
            let a = &node.inputs[0];
            let b = &node.inputs[1];
            let ga = upstream.matmul(&b.data.borrow().transpose());
            let gb = a.data.borrow().transpose().matmul(upstream);
            accumulate(a, &ga);
            accumulate(b, &gb);
        }

        Op::Scale(s) => {
            let ga = upstream.scale(s);
            accumulate(&node.inputs[0], &ga);
        }

        Op::ReLU => {
            let x = &node.inputs[0];
            let mask = x.data.borrow().map(|v| if v > 0.0 { 1.0 } else { 0.0 });
            let g = upstream.mul_elem(&mask);
            accumulate(x, &g);
        }

        Op::Sigmoid => {
            // cache[0] = sigmoid(x)
            let x = &node.inputs[0];
            let s = &node.cache[0];
            let one_minus = s.map(|v| 1.0 - v);
            let deriv = s.mul_elem(&one_minus);
            let g = upstream.mul_elem(&deriv);
            accumulate(x, &g);
        }

        Op::Transpose => {
            let g = upstream.transpose();
            accumulate(&node.inputs[0], &g);
        }

        Op::Sum => {
            let scalar = upstream.get(0, 0);
            let x = &node.inputs[0];
            let d = x.data.borrow();
            let g = Matrix::filled(d.rows, d.cols, scalar);
            drop(d);
            accumulate(x, &g);
        }

        Op::SoftmaxCE => {
            let z = &node.inputs[0];
            let t = &node.inputs[1];
            let td = t.data.borrow();
            let batch = td.rows;
            // cache[0] = softmax probs
            let probs = &node.cache[0];

            // dL/dlogits = (softmax - onehot) / batch
            let mut diff = Matrix::zeros(batch, td.cols);
            for i in 0..batch {
                for j in 0..td.cols {
                    diff.set(i, j, probs.get(i, j) - td.get(i, j));
                }
            }
            drop(td);

            // Kritik düzeltme: 1/batch
            let scalar = upstream.get(0, 0) / batch as f32;
            let mut g_logits = Matrix::zeros(batch, diff.cols);
            for i in 0..batch {
                for j in 0..diff.cols {
                    g_logits.set(i, j, diff.get(i, j) * scalar);
                }
            }
            accumulate(z, &g_logits);
        }

        Op::Softmax => {
            // cache[0] = softmax çıktısı
            let x = &node.inputs[0];
            let sm = &node.cache[0];
            let (rows, cols) = (sm.rows, sm.cols);
            let up = upstream;

            let mut g = Matrix::zeros(rows, cols);
            for i in 0..rows {
                let mut dot = 0.0f32;
                for j in 0..cols {
                    dot += up.get(i, j) * sm.get(i, j);
                }
                for j in 0..cols {
                    g.set(i, j, sm.get(i, j) * (up.get(i, j) - dot));
                }
            }
            accumulate(x, &g);
        }

        Op::GELU => {
            let x = &node.inputs[0];
            let xd = x.data.borrow();
            let (rows, cols) = (xd.rows, xd.cols);
            let c = (2.0f32 / std::f32::consts::PI).sqrt();
            let mut g = Matrix::zeros(rows, cols);
            for i in 0..rows {
                for j in 0..cols {
                    let v = xd.get(i, j);
                    let inner = c * (v + 0.044715 * v * v * v);
                    let t = inner.tanh();
                    let dinner = c * (1.0 + 3.0 * 0.044715 * v * v);
                    let deriv = 0.5 * (1.0 + t) + 0.5 * v * (1.0 - t * t) * dinner;
                    g.set(i, j, upstream.get(i, j) * deriv);
                }
            }
            drop(xd);
            accumulate(x, &g);
        }

        Op::LayerNorm(eps) => {
            let x = &node.inputs[0];
            let gamma = &node.inputs[1];
            let beta = &node.inputs[2];
            // cache[0] = x_hat, cache[1] = inv_std
            let x_hat = &node.cache[0];
            let inv = &node.cache[1];

            let xd = x.data.borrow();
            let gd = gamma.data.borrow();
            let bd = beta.data.borrow();

            let (rows, cols) = (xd.rows, xd.cols);
            let mut g_x = Matrix::zeros(rows, cols);
            let mut g_gamma = Matrix::zeros(1, cols);
            let mut g_beta = Matrix::zeros(1, cols);

            let _ = eps; // cache'te var, gerekmez ama enum'da

            for i in 0..rows {
                let inv_i = inv.get(i, 0);
                let n = cols as f32;

                // g_beta += upstream
                for j in 0..cols {
                    g_beta.set(0, j, g_beta.get(0, j) + upstream.get(i, j));
                }

                // g_gamma += upstream * x_hat
                for j in 0..cols {
                    g_gamma.set(0, j, g_gamma.get(0, j) + upstream.get(i, j) * x_hat.get(i, j));
                }

                // dx_hat = upstream * gamma
                let mut dx_hat = vec![0.0f32; cols];
                for j in 0..cols {
                    dx_hat[j] = upstream.get(i, j) * gd.get(0, j);
                }

                let mut sum_dxhat = 0.0f32;
                let mut sum_dxhat_xhat = 0.0f32;
                for j in 0..cols {
                    sum_dxhat += dx_hat[j];
                    sum_dxhat_xhat += dx_hat[j] * x_hat.get(i, j);
                }

                for j in 0..cols {
                    let gx = (1.0 / n) * inv_i
                        * (n * dx_hat[j] - sum_dxhat - x_hat.get(i, j) * sum_dxhat_xhat);
                    g_x.set(i, j, gx);
                }
            }

            drop(xd);
            drop(gd);
            drop(bd);

            accumulate(x, &g_x);
            accumulate(gamma, &g_gamma);
            accumulate(beta, &g_beta);
        }

        Op::Embedding => {
            let tok_emb = &node.inputs[0];
            let pos_emb = &node.inputs[1];
            let ids = &node.inputs[2];

            let idd = ids.data.borrow();
            let seq_len = idd.rows;
            let d_model = upstream.cols;

            let vocab = tok_emb.data.borrow().rows;
            let pos_rows = pos_emb.data.borrow().rows;
            let mut g_tok = Matrix::zeros(vocab, d_model);
            let mut g_pos = Matrix::zeros(pos_rows, d_model);

            for s in 0..seq_len {
                let id = idd.get(s, 0) as usize;
                for d in 0..d_model {
                    let g = upstream.get(s, d);
                    g_tok.set(id, d, g_tok.get(id, d) + g);
                    g_pos.set(s, d, g_pos.get(s, d) + g);
                }
            }
            drop(idd);

            accumulate(tok_emb, &g_tok);
            accumulate(pos_emb, &g_pos);
        }

        Op::MaskCausal => {
            let x = &node.inputs[0];
            let d = x.data.borrow();
            let (rows, cols) = (d.rows, d.cols);
            let mut g = Matrix::zeros(rows, cols);
            for i in 0..rows {
                for j in 0..cols {
                    if j <= i {
                        g.set(i, j, upstream.get(i, j));
                    }
                }
            }
            drop(d);
            accumulate(x, &g);
        }
    }
}

fn accumulate(t: &Tensor, g: &Matrix) {
    if !t.requires_grad { return; }
    let mut tg = t.grad.borrow_mut();
    assert_eq!((tg.rows, tg.cols), (g.rows, g.cols), "grad boyut");
    for i in 0..tg.data.len() {
        tg.data[i] += g.data[i];
    }
}

fn reduce_to(upstream: &Matrix, rows: usize, cols: usize) -> Matrix {
    if upstream.rows == rows && upstream.cols == cols {
        return upstream.clone();
    }
    if rows == 1 && upstream.cols == cols {
        let mut out = Matrix::zeros(1, cols);
        for i in 0..upstream.rows {
            for j in 0..cols {
                out.set(0, j, out.get(0, j) + upstream.get(i, j));
            }
        }
        return out;
    }
    panic!(
        "reduce_to: {}x{} -> {}x{} yapılamıyor",
        upstream.rows, upstream.cols, rows, cols
    );
}