// src/autograd.rs
// StonefishAI — autograd motoru.
//
// Design note (batching):
// Batched work is not done with a separate 3D tensor type but by **row packing**:
// The (B, S, D) tensor is stored as a (B*S, D) matrix. LayerNorm, Linear, GELU,
// bias-add and softmax are already row-wise; the only exception is attention,
// resolved inside `Op::CausalAttention` with batch/seq/head metadata.
// So a single autograd engine carries both the MLP and the GPT.

use crate::tensor::Matrix;
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

#[derive(Clone, Copy, Debug)]
pub enum Op {
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
    /// Seyrek (index) hedefli cross-entropy: one-hot matrisi tahsis etmez.
    /// produces the same gradient as dense `SoftmaxCE`.
    SoftmaxCEIds,
    Softmax,
    GELU,
    LayerNorm(f32), // eps
    /// Row-packed embedding: `seq_len` is needed to recover the position index.
    Embedding {
        seq_len: usize,
    },
    MaskCausal,
    /// Fused causal multi-head self-attention.
    /// q, k, v: (batch*seq, d_model). Output: (batch*seq, d_model).
    /// Each head softmaxes its own d_model/n_heads slice.
    CausalAttention {
        batch: usize,
        seq: usize,
        n_heads: usize,
    },
    /// RMSNorm - LayerNorm without the mean (LLaMA/PaLM).
    /// `y = x / sqrt(mean(x^2)+eps) * gamma`. No beta.
    RmsNorm(f32),
    /// Rotary position encoding (RoPE). Input: (batch*seq, n_heads*d_head).
    /// The (2i, 2i+1) pair inside each head slice is rotated by token position.
    Rope {
        batch: usize,
        seq: usize,
        n_heads: usize,
    },
}

pub struct Node {
    pub op: Op,
    pub inputs: Vec<Tensor>,
    /// Intermediate values computed during the forward pass (cache for backward)
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

    pub fn from_op_cached(data: Matrix, op: Op, inputs: Vec<Tensor>, cache: Vec<Matrix>) -> Self {
        let t = crate::profile::start();
        let requires_grad = inputs.iter().any(|t| t.requires_grad);
        let grad = Rc::new(RefCell::new(Matrix::zeros(data.rows, data.cols)));
        let node = if requires_grad {
            Some(Rc::new(Node { op, inputs, cache }))
        } else {
            None
        };
        let out = Self {
            data: Rc::new(RefCell::new(data)),
            grad,
            requires_grad,
            node,
        };
        crate::profile::stop(t, &crate::profile::ALLOC_NANOS);
        out
    }

    pub fn zero_grad(&self) {
        let t = crate::profile::start();
        let d = self.data.borrow();
        *self.grad.borrow_mut() = Matrix::zeros(d.rows, d.cols);
        drop(d);
        crate::profile::stop(t, &crate::profile::ZEROGRAD_NANOS);
    }

    pub fn apply_grad(&self, lr: f32) {
        let g = self.grad.borrow();
        let mut d = self.data.borrow_mut();
        assert_eq!((d.rows, d.cols), (g.rows, g.cols), "apply_grad size");
        for i in 0..d.data.len() {
            d.data[i] -= lr * g.data[i];
        }
    }

    /// The value of a 1x1 tensor.
    pub fn item(&self) -> f32 {
        self.data.borrow().scalar()
    }

    // ---- Forward ----

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
        let data = {
            let a = self.data.borrow();
            let b = other.data.borrow();
            crate::gpu::matmul_gpu(&a, &b)
        };
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
        let s = self.data.borrow().map(|x| 1.0 / (1.0 + (-x).exp()));
        Tensor::from_op_cached(s.clone(), Op::Sigmoid, vec![self.clone()], vec![s])
    }

    pub fn transpose(&self) -> Tensor {
        let data = self.data.borrow().transpose();
        Tensor::from_op(data, Op::Transpose, vec![self.clone()])
    }

    pub fn sum(&self) -> Tensor {
        let s: f32 = self.data.borrow().data.iter().sum();
        Tensor::from_op(Matrix::from_vec(1, 1, vec![s]), Op::Sum, vec![self.clone()])
    }

    pub fn softmax_cross_entropy(&self, onehot: &Tensor) -> Tensor {
        let zd = self.data.borrow();
        let td = onehot.data.borrow();
        assert_eq!((zd.rows, zd.cols), (td.rows, td.cols), "CE size");

        let batch = zd.rows;
        let classes = zd.cols;
        let mut total_loss = 0.0f32;
        let mut probs = Matrix::zeros(batch, classes);

        for i in 0..batch {
            let mut max = f32::NEG_INFINITY;
            for j in 0..classes {
                let v = zd.get(i, j);
                if v > max {
                    max = v;
                }
            }
            let mut sum_exp = 0.0f32;
            for j in 0..classes {
                sum_exp += (zd.get(i, j) - max).exp();
            }
            let log_sum_exp = max + sum_exp.ln();
            for j in 0..classes {
                probs.set(i, j, (zd.get(i, j) - max).exp() / sum_exp);
                if td.get(i, j) > 0.5 {
                    total_loss += -zd.get(i, j) + log_sum_exp;
                }
            }
        }

        let mean_loss = total_loss / batch as f32;
        drop(zd);
        drop(td);
        Tensor::from_op_cached(
            Matrix::from_vec(1, 1, vec![mean_loss]),
            Op::SoftmaxCE,
            vec![self.clone(), onehot.clone()],
            vec![probs],
        )
    }

    /// Sparse-target cross-entropy: `targets[i]` is the correct class of row i.
    /// It does not allocate a one-hot matrix - during GPT training it saves a
    /// (B*S x vocab) matrix. Same gradient as dense `softmax_cross_entropy`.
    pub fn softmax_cross_entropy_ids(&self, targets: &[usize]) -> Tensor {
        let t = crate::profile::start();
        let zd = self.data.borrow();
        let (rows, classes) = (zd.rows, zd.cols);
        assert_eq!(
            targets.len(),
            rows,
            "CE: number of targets must equal the number of rows"
        );

        let mut total_loss = 0.0f32;
        let mut probs = Matrix::zeros(rows, classes);
        let mut tgt = Matrix::zeros(rows, 1);

        for i in 0..rows {
            let t = targets[i];
            assert!(t < classes, "CE: target {t} >= class count {classes}");
            tgt.set(i, 0, t as f32);

            let mut max = f32::NEG_INFINITY;
            for j in 0..classes {
                let v = zd.get(i, j);
                if v > max {
                    max = v;
                }
            }
            let mut sum_exp = 0.0f32;
            for j in 0..classes {
                sum_exp += (zd.get(i, j) - max).exp();
            }
            let log_sum_exp = max + sum_exp.ln();
            let inv_sum = 1.0 / sum_exp;
            for j in 0..classes {
                probs.set(i, j, (zd.get(i, j) - max).exp() * inv_sum);
            }
            total_loss += -zd.get(i, t) + log_sum_exp;
        }

        let mean_loss = total_loss / rows as f32;
        drop(zd);
        crate::profile::stop(t, &crate::profile::CE_NANOS);
        Tensor::from_op_cached(
            Matrix::from_vec(1, 1, vec![mean_loss]),
            Op::SoftmaxCEIds,
            vec![self.clone()],
            vec![probs, tgt],
        )
    }

    pub fn softmax(&self) -> Tensor {
        let t = crate::profile::start();
        let zd = self.data.borrow();
        let (rows, cols) = (zd.rows, zd.cols);
        let mut data = Matrix::zeros(rows, cols);
        for i in 0..rows {
            let mut max = f32::NEG_INFINITY;
            for j in 0..cols {
                let v = zd.get(i, j);
                if v > max {
                    max = v;
                }
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
        crate::profile::stop(t, &crate::profile::SOFTMAX_NANOS);
        Tensor::from_op_cached(data.clone(), Op::Softmax, vec![self.clone()], vec![data])
    }

    pub fn gelu(&self) -> Tensor {
        let t = crate::profile::start();
        let data = self.data.borrow().map(|x| {
            let c = (2.0f32 / std::f32::consts::PI).sqrt();
            0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
        });
        crate::profile::stop(t, &crate::profile::GELU_NANOS);
        Tensor::from_op(data, Op::GELU, vec![self.clone()])
    }

    pub fn layer_norm(&self, gamma: &Tensor, beta: &Tensor, eps: f32) -> Tensor {
        let t = crate::profile::start();
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
            for j in 0..cols {
                mean += xd.get(i, j);
            }
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
        crate::profile::stop(t, &crate::profile::LN_NANOS);

        Tensor::from_op_cached(
            data,
            Op::LayerNorm(eps),
            vec![self.clone(), gamma.clone(), beta.clone()],
            vec![x_hat_cache, inv_cache],
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

    /// RMSNorm: `y = x / sqrt(mean(x^2)+eps) * gamma` (no mean subtraction).
    ///
    /// Unlike LayerNorm: no mean is computed and there is no beta. Used by
    /// LLaMA/PaLM; the same quality for less work.
    pub fn rms_norm(&self, gamma: &Tensor, eps: f32) -> Tensor {
        let xd = self.data.borrow();
        let (rows, cols) = (xd.rows, xd.cols);
        let gd = gamma.data.borrow();
        assert_eq!(gd.cols, cols, "rms_norm: gamma boyutu");

        let mut data = Matrix::zeros(rows, cols);
        let mut x_hat = Matrix::zeros(rows, cols);
        let mut inv = Matrix::zeros(rows, 1);
        for i in 0..rows {
            let mut ss = 0.0f32;
            for j in 0..cols {
                let v = xd.get(i, j);
                ss += v * v;
            }
            let r = 1.0 / (ss / cols as f32 + eps).sqrt();
            inv.set(i, 0, r);
            for j in 0..cols {
                let h = xd.get(i, j) * r;
                x_hat.set(i, j, h);
                data.set(i, j, h * gd.get(0, j));
            }
        }
        drop(xd);
        drop(gd);
        Self::from_op_cached(
            data,
            Op::RmsNorm(eps),
            vec![self.clone(), gamma.clone()],
            vec![x_hat, inv],
        )
    }

    /// RoPE: rotates a (batch*seq, n_heads*d_head) tensor by position.
    pub fn rope(&self, batch: usize, seq: usize, n_heads: usize) -> Tensor {
        let xd = self.data.borrow();
        let (rows, cols) = (xd.rows, xd.cols);
        assert_eq!(rows, batch * seq, "rope: rows must equal batch*seq");
        assert_eq!(cols % n_heads, 0, "rope: columns must be divisible by n_heads");
        let d_head = cols / n_heads;
        assert!(d_head % 2 == 0, "rope: d_head must be even");

        let mut data = Matrix::zeros(rows, cols);
        for r in 0..rows {
            let s = r % seq;
            for h in 0..n_heads {
                for j in 0..d_head / 2 {
                    let freq = 10000f32.powf(-2.0 * j as f32 / d_head as f32);
                    let ang = s as f32 * freq;
                    let (c, sn) = (ang.cos(), ang.sin());
                    let base = h * d_head + 2 * j;
                    let x0 = xd.get(r, base);
                    let x1 = xd.get(r, base + 1);
                    data.set(r, base, x0 * c - x1 * sn);
                    data.set(r, base + 1, x0 * sn + x1 * c);
                }
            }
        }
        drop(xd);
        Self::from_op(
            data,
            Op::Rope {
                batch,
                seq,
                n_heads,
            },
            vec![self.clone()],
        )
    }

    /// Fused causal multi-head self-attention.
    ///
    /// * `self` = Q, `k` = K, `v` = V; all three are `(batch*seq, d_model)`.
    /// * `d_model` must be divisible by `n_heads`; each head softmaxes inside its
    ///   own `d_head` slice and the slices are merged on output.
    /// * The `1/sqrt(d_head)` scale and the causal mask live inside the op.
    pub fn causal_attention(
        &self,
        k: &Tensor,
        v: &Tensor,
        batch: usize,
        seq: usize,
        n_heads: usize,
    ) -> Tensor {
        let prof = crate::profile::enabled();
        let t0 = if prof {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let (out_data, weights) = {
            let qd = self.data.borrow();
            let kd = k.data.borrow();
            let vd = v.data.borrow();

            let (rows, d_model) = (qd.rows, qd.cols);
            assert_eq!(rows, batch * seq, "attention: rows must equal batch*seq");
            assert_eq!(kd.rows, rows, "attention: K row count");
            assert_eq!(vd.rows, rows, "attention: V row count");
            assert_eq!(kd.cols, d_model, "attention: K boyutu");
            assert_eq!(vd.cols, d_model, "attention: V boyutu");
            assert_eq!(d_model % n_heads, 0, "attention: d_model % n_heads");
            let d_head = d_model / n_heads;
            let scale = 1.0 / (d_head as f32).sqrt();

            // cache: (batch*n_heads*seq, seq) attention weights
            let mut w_cache = Matrix::zeros(batch * n_heads * seq, seq);
            let mut out = Matrix::zeros(rows, d_model);
            let mut scores = vec![0.0f32; seq];

            // Inner loops are slice-based: no bounds checks, open to vectorization.
            for b in 0..batch {
                for h in 0..n_heads {
                    let off = h * d_head;
                    let w_row0 = (b * n_heads + h) * seq;
                    for i in 0..seq {
                        let q_row = b * seq + i;
                        let q_s = &qd.data[q_row * d_model + off..q_row * d_model + off + d_head];

                        // scores (only j <= i)
                        let mut max = f32::NEG_INFINITY;
                        for j in 0..=i {
                            let k_row = b * seq + j;
                            let k_s =
                                &kd.data[k_row * d_model + off..k_row * d_model + off + d_head];
                            let mut acc = 0.0f32;
                            for d in 0..d_head {
                                acc += q_s[d] * k_s[d];
                            }
                            scores[j] = acc * scale;
                            if scores[j] > max {
                                max = scores[j];
                            }
                        }

                        // nedensel softmax (j <= i)
                        let mut sum = 0.0f32;
                        for j in 0..=i {
                            scores[j] = (scores[j] - max).exp();
                            sum += scores[j];
                        }
                        let inv_sum = 1.0 / sum;
                        for j in 0..=i {
                            scores[j] *= inv_sum;
                            w_cache.set(w_row0 + i, j, scores[j]);
                        }

                        // weighted sum of V
                        let out_off = q_row * d_model + off;
                        let o_s = &mut out.data[out_off..out_off + d_head];
                        for d in 0..d_head {
                            o_s[d] = 0.0;
                        }
                        for j in 0..=i {
                            let wj = scores[j];
                            let v_row = b * seq + j;
                            let v_s =
                                &vd.data[v_row * d_model + off..v_row * d_model + off + d_head];
                            for d in 0..d_head {
                                o_s[d] += wj * v_s[d];
                            }
                        }
                    }
                }
            }
            (out, w_cache)
        };

        if let Some(t0) = t0 {
            crate::profile::add_attn_fwd(t0.elapsed().as_nanos() as u64);
        }

        Tensor::from_op_cached(
            out_data,
            Op::CausalAttention {
                batch,
                seq,
                n_heads,
            },
            vec![self.clone(), k.clone(), v.clone()],
            vec![weights],
        )
    }

    // ---- Backward ----

    /// Seeds the root tensor's gradient with 1.0 and accumulates all intermediate
    /// gradients in reverse topological order.
    ///
    /// Note: gradients are not zeroed - consecutive `backward()` calls accumulate.
    /// The training loop manages this via `optimizer::zero_grad()`.
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
            // IMPORTANT: we borrow the upstream instead of cloning it. Previously a full
            // copy of the gradient matrix was taken per node; in a 4-layer model that
            // meant dozens of MB of useless copying per step. Since a node's own
            // gradient and its inputs' gradients are different RefCells
            // (a DAG), there is no borrow conflict.
            let upstream = t.grad.borrow();
            apply_backward(&node, &upstream);
            drop(upstream);
        }
    }

    fn build_topo(&self, topo: &mut Vec<Tensor>, visited: &mut HashSet<*const Node>) {
        if let Some(node) = &self.node {
            let ptr = Rc::as_ptr(node);
            if !visited.insert(ptr) {
                return;
            }
            for inp in &node.inputs {
                inp.build_topo(topo, visited);
            }
        }
        topo.push(self.clone());
    }
}

fn apply_backward(node: &Node, upstream: &Matrix) {
    match node.op {
        Op::Add => {
            let a = &node.inputs[0];
            let b = &node.inputs[1];
            accumulate_reduced(a, upstream, false);
            accumulate_reduced(b, upstream, false);
        }

        Op::Sub => {
            let a = &node.inputs[0];
            let b = &node.inputs[1];
            accumulate_reduced(a, upstream, false);
            accumulate_reduced(b, upstream, true);
        }

        Op::MulElem => {
            let a = &node.inputs[0];
            let b = &node.inputs[1];
            let ga = {
                let raw = upstream.mul_elem(&b.data.borrow());
                reduce_to(&raw, a.data.borrow().rows, a.data.borrow().cols)
            };
            let gb = {
                let raw = upstream.mul_elem(&a.data.borrow());
                reduce_to(&raw, b.data.borrow().rows, b.data.borrow().cols)
            };
            accumulate(a, &ga);
            accumulate(b, &gb);
        }

        Op::MatMul => {
            let a = &node.inputs[0];
            let b = &node.inputs[1];
            // We let cuBLAS do the transposes instead of materializing them: strided
            // dA = dOut·Bᵀ, dB = Aᵀ·dOut. Eskiden her backward'da (1024×128)'lik
            // transpose copies used to be taken - a significant part of the "other"
            // bucket in the profile.
            let ga = crate::gpu::matmul_bt_gpu(upstream, &b.data.borrow());
            let gb = crate::gpu::matmul_ta_gpu(&a.data.borrow(), upstream);
            accumulate(a, &ga);
            accumulate(b, &gb);
        }

        Op::Scale(s) => {
            accumulate(&node.inputs[0], &upstream.scale(s));
        }

        Op::ReLU => {
            let x = &node.inputs[0];
            let mask = x.data.borrow().map(|v| if v > 0.0 { 1.0 } else { 0.0 });
            accumulate(x, &upstream.mul_elem(&mask));
        }

        Op::Sigmoid => {
            let x = &node.inputs[0];
            let s = &node.cache[0];
            let deriv = s.mul_elem(&s.map(|v| 1.0 - v));
            accumulate(x, &upstream.mul_elem(&deriv));
        }

        Op::Transpose => {
            accumulate(&node.inputs[0], &upstream.transpose());
        }

        Op::Sum => {
            let x = &node.inputs[0];
            let d = x.data.borrow();
            let g = Matrix::filled(d.rows, d.cols, upstream.scalar());
            drop(d);
            accumulate(x, &g);
        }

        Op::SoftmaxCE => {
            let z = &node.inputs[0];
            let t = &node.inputs[1];
            let td = t.data.borrow();
            let batch = td.rows;
            let probs = &node.cache[0];

            // dL/dlogits = (softmax - onehot) / batch
            let mut g_logits = Matrix::zeros(batch, td.cols);
            let scalar = upstream.scalar() / batch as f32;
            for i in 0..batch {
                for j in 0..td.cols {
                    g_logits.set(i, j, (probs.get(i, j) - td.get(i, j)) * scalar);
                }
            }
            drop(td);
            accumulate(z, &g_logits);
        }

        Op::SoftmaxCEIds => {
            let t = crate::profile::start();
            let z = &node.inputs[0];
            let probs = &node.cache[0];
            let tgt = &node.cache[1];
            let zd = z.data.borrow();
            let (rows, classes) = (zd.rows, zd.cols);
            let scalar = upstream.scalar() / rows as f32;

            let mut g_logits = Matrix::zeros(rows, classes);
            for i in 0..rows {
                let t = tgt.get(i, 0) as usize;
                for j in 0..classes {
                    let mut g = probs.get(i, j);
                    if j == t {
                        g -= 1.0;
                    }
                    g_logits.set(i, j, g * scalar);
                }
            }
            drop(zd);
            crate::profile::stop(t, &crate::profile::CE_NANOS);
            accumulate(z, &g_logits);
        }

        Op::Softmax => {
            let x = &node.inputs[0];
            let sm = &node.cache[0];
            let (rows, cols) = (sm.rows, sm.cols);
            let mut g = Matrix::zeros(rows, cols);
            for i in 0..rows {
                let mut dot = 0.0f32;
                for j in 0..cols {
                    dot += upstream.get(i, j) * sm.get(i, j);
                }
                for j in 0..cols {
                    g.set(i, j, sm.get(i, j) * (upstream.get(i, j) - dot));
                }
            }
            accumulate(x, &g);
        }

        Op::GELU => {
            let t = crate::profile::start();
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
            crate::profile::stop(t, &crate::profile::GELU_NANOS);
            accumulate(x, &g);
        }

        Op::LayerNorm(_eps) => {
            let t = crate::profile::start();
            let x = &node.inputs[0];
            let gamma = &node.inputs[1];
            let beta = &node.inputs[2];
            let x_hat = &node.cache[0];
            let inv = &node.cache[1];

            let xd = x.data.borrow();
            let gd = gamma.data.borrow();
            let (rows, cols) = (xd.rows, xd.cols);

            let mut g_x = Matrix::zeros(rows, cols);
            let mut g_gamma = Matrix::zeros(1, cols);
            let mut g_beta = Matrix::zeros(1, cols);

            for i in 0..rows {
                let inv_i = inv.get(i, 0);
                let n = cols as f32;

                for j in 0..cols {
                    let up = upstream.get(i, j);
                    g_beta.set(0, j, g_beta.get(0, j) + up);
                    g_gamma.set(0, j, g_gamma.get(0, j) + up * x_hat.get(i, j));
                }

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
                    g_x.set(
                        i,
                        j,
                        (1.0 / n)
                            * inv_i
                            * (n * dx_hat[j] - sum_dxhat - x_hat.get(i, j) * sum_dxhat_xhat),
                    );
                }
            }

            drop(xd);
            drop(gd);
            crate::profile::stop(t, &crate::profile::LN_NANOS);
            accumulate(x, &g_x);
            accumulate(gamma, &g_gamma);
            accumulate(beta, &g_beta);
        }

        Op::RmsNorm(_eps) => {
            let x = &node.inputs[0];
            let gamma = &node.inputs[1];
            let x_hat = &node.cache[0];
            let inv = &node.cache[1];

            let (rows, cols) = {
                let xd = x.data.borrow();
                (xd.rows, xd.cols)
            };
            let gdata = gamma.data.borrow();
            let mut g_x = Matrix::zeros(rows, cols);
            let mut g_gamma = Matrix::zeros(1, cols);

            for i in 0..rows {
                // dgamma contribution: dy * xhat
                for j in 0..cols {
                    g_gamma.set(
                        0,
                        j,
                        g_gamma.get(0, j) + upstream.get(i, j) * x_hat.get(i, j),
                    );
                }
                // dx = inv·(dy·γ − x̂·mean(dy·γ·x̂))
                let mut mean = 0.0f32;
                for j in 0..cols {
                    mean += upstream.get(i, j) * gdata.get(0, j) * x_hat.get(i, j);
                }
                mean /= cols as f32;
                let r_inv = inv.get(i, 0);
                for j in 0..cols {
                    let g = upstream.get(i, j) * gdata.get(0, j);
                    g_x.set(i, j, r_inv * (g - x_hat.get(i, j) * mean));
                }
            }
            drop(gdata);
            accumulate(x, &g_x);
            accumulate(gamma, &g_gamma);
        }

        Op::Rope {
            batch: _batch,
            seq,
            n_heads,
        } => {
            let x = &node.inputs[0];
            let (rows, cols) = {
                let xd = x.data.borrow();
                (xd.rows, xd.cols)
            };
            let d_head = cols / n_heads;
            let mut g_x = Matrix::zeros(rows, cols);
            // Inverse rotation: R(-theta) = R(theta)^T
            for r in 0..rows {
                let s = r % seq;
                for h in 0..n_heads {
                    for j in 0..d_head / 2 {
                        let freq = 10000f32.powf(-2.0 * j as f32 / d_head as f32);
                        let ang = s as f32 * freq;
                        let (c, sn) = (ang.cos(), ang.sin());
                        let base = h * d_head + 2 * j;
                        let y0 = upstream.get(r, base);
                        let y1 = upstream.get(r, base + 1);
                        g_x.set(r, base, y0 * c + y1 * sn);
                        g_x.set(r, base + 1, -y0 * sn + y1 * c);
                    }
                }
            }
            accumulate(x, &g_x);
        }

        Op::MaskCausal => {
            // Causal mask: the upper triangle (j > i) is masked in the forward pass,
            // so no gradient flows into that region.
            let x = &node.inputs[0];
            let (rows, cols) = {
                let d = x.data.borrow();
                (d.rows, d.cols)
            };
            let mut g = Matrix::zeros(rows, cols);
            for i in 0..rows {
                for j in 0..cols {
                    if j <= i {
                        g.set(i, j, upstream.get(i, j));
                    }
                }
            }
            accumulate(x, &g);
        }

        Op::Embedding { seq_len } => {
            // v2: girdiler (tok_emb, ids). Konum bilgisi RoPE'den gelir,
            // therefore there is no positional embedding gradient.
            let _ = seq_len;
            let tok_emb = &node.inputs[0];
            let ids = &node.inputs[1];

            let idd = ids.data.borrow();
            let rows = idd.rows;
            let d_model = upstream.cols;
            let vocab = tok_emb.data.borrow().rows;
            let mut g_tok = Matrix::zeros(vocab, d_model);
            for r in 0..rows {
                let id = idd.get(r, 0) as usize;
                for j in 0..d_model {
                    g_tok.set(id, j, g_tok.get(id, j) + upstream.get(r, j));
                }
            }
            drop(idd);
            accumulate(tok_emb, &g_tok);
        }
        Op::CausalAttention {
            batch,
            seq,
            n_heads,
        } => {
            let q = &node.inputs[0];
            let k = &node.inputs[1];
            let v = &node.inputs[2];
            let weights = &node.cache[0]; // (batch*n_heads*seq, seq)

            let qd = q.data.borrow();
            let kd = k.data.borrow();
            let vd = v.data.borrow();
            let d_model = qd.cols;
            let d_head = d_model / n_heads;
            let scale = 1.0 / (d_head as f32).sqrt();
            let up = &upstream.data;
            let prof = crate::profile::enabled();
            let t0 = if prof {
                Some(std::time::Instant::now())
            } else {
                None
            };

            // Forward: out_i = Σ_{j<=i} w_ij v_j,  w = softmax(scale·q_i·k_j)
            // Backward:
            //   dV_j += w_ij · dOut_i
            //   dS_ij  = w_ij · ( dW_ij − Σ_j' w_ij' dW_ij' ),  dW_ij = <dOut_i, v_j>
            //   dQ_i  += scale · Σ_j dS_ij · k_j
            //   dK_j  += scale · dS_ij · q_i
            let total = batch * seq * d_model;
            let mut g_q = vec![0.0f32; total];
            let mut g_k = vec![0.0f32; total];
            let mut g_v = vec![0.0f32; total];

            let mut dw = vec![0.0f32; seq];
            let mut ds = vec![0.0f32; seq];

            for b in 0..batch {
                for h in 0..n_heads {
                    let off = h * d_head;
                    let w_row0 = (b * n_heads + h) * seq;
                    for i in 0..seq {
                        let i1 = i + 1;
                        let q_row = b * seq + i;
                        let q_off = q_row * d_model + off;
                        // Take the slices once: no bounds checks in the inner loops,
                        // the zips are open to vectorization.
                        let q_s = &qd.data[q_off..q_off + d_head];
                        let up_s = &up[q_off..q_off + d_head];
                        let w_row = &weights.data[(w_row0 + i) * seq..(w_row0 + i) * seq + i1];

                        // dW_ij = <dOut_i, V_j> ; dsum = Σ_j w_ij·dW_ij
                        let mut dsum = 0.0f32;
                        for j in 0..i1 {
                            let v_off = (b * seq + j) * d_model + off;
                            let v_s = &vd.data[v_off..v_off + d_head];
                            let mut acc = 0.0f32;
                            for (x, y) in up_s.iter().zip(v_s) {
                                acc += x * y;
                            }
                            dw[j] = acc;
                            dsum += w_row[j] * acc;
                        }

                        // softmax backward (including the scale) + dV
                        for j in 0..i1 {
                            let wj = w_row[j];
                            ds[j] = wj * (dw[j] - dsum) * scale;

                            // dV_j += w_ij · dOut_i
                            let v_off = (b * seq + j) * d_model + off;
                            let v_dst = &mut g_v[v_off..v_off + d_head];
                            for (o, &u) in v_dst.iter_mut().zip(up_s) {
                                *o += wj * u;
                            }
                        }

                        // dQ_i += Σ_j ds_ij·k_j ; dK_j += ds_ij·q_i
                        let q_dst = &mut g_q[q_off..q_off + d_head];
                        for j in 0..i1 {
                            let dsj = ds[j];
                            let k_off = (b * seq + j) * d_model + off;
                            let k_s = &kd.data[k_off..k_off + d_head];
                            for (o, &kj) in q_dst.iter_mut().zip(k_s) {
                                *o += dsj * kj;
                            }
                            let k_dst = &mut g_k[k_off..k_off + d_head];
                            for (o, &qi) in k_dst.iter_mut().zip(q_s) {
                                *o += dsj * qi;
                            }
                        }
                    }
                }
            }

            accumulate(q, &Matrix::from_vec(batch * seq, d_model, g_q));
            accumulate(k, &Matrix::from_vec(batch * seq, d_model, g_k));
            accumulate(v, &Matrix::from_vec(batch * seq, d_model, g_v));

            if let Some(t0) = t0 {
                crate::profile::add_attn_bwd(t0.elapsed().as_nanos() as u64);
            }
        }
    }
}

fn accumulate(t: &Tensor, g: &Matrix) {
    if !t.requires_grad {
        return;
    }
    let mut tg = t.grad.borrow_mut();
    assert_eq!((tg.rows, tg.cols), (g.rows, g.cols), "grad size");
    for i in 0..tg.data.len() {
        tg.data[i] += g.data[i];
    }
}

/// A clone-free combination of `accumulate` + `reduce_to`: adds directly when
/// shapes match, or by summing rows for a (1, cols) broadcast. `negate` flips
/// the sign for subtraction. Used by ops like Add/Sub that do not deserve a copy.
fn accumulate_reduced(t: &Tensor, upstream: &Matrix, negate: bool) {
    if !t.requires_grad {
        return;
    }
    let (rows, cols) = {
        let d = t.data.borrow();
        (d.rows, d.cols)
    };
    let sign = if negate { -1.0f32 } else { 1.0f32 };
    let mut tg = t.grad.borrow_mut();

    if upstream.rows == rows && upstream.cols == cols {
        for i in 0..tg.data.len() {
            tg.data[i] += sign * upstream.data[i];
        }
    } else if rows == 1 && upstream.cols == cols {
        for i in 0..upstream.rows {
            let row = &upstream.data[i * cols..(i + 1) * cols];
            for j in 0..cols {
                tg.data[j] += sign * row[j];
            }
        }
    } else {
        panic!(
            "accumulate_reduced: cannot {}x{} -> {}x{}",
            upstream.rows, upstream.cols, rows, cols
        );
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
        "reduce_to: cannot {}x{} -> {}x{}",
        upstream.rows, upstream.cols, rows, cols
    );
}
