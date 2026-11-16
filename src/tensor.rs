// src/tensor.rs
// StonefishAI — sıfırdan Matrix (2D) + Tensor3D (batch).

use std::fmt;

// ============================================================
// Matrix (2D)
// ============================================================

#[derive(Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>, // row-major
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![0.0; rows * cols] }
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(rows * cols, data.len(), "boyut uyuşmuyor");
        Self { rows, cols, data }
    }

    pub fn filled(rows: usize, cols: usize, v: f32) -> Self {
        Self { rows, cols, data: vec![v; rows * cols] }
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: f32) {
        self.data[r * self.cols + c] = v;
    }

    pub fn random(rows: usize, cols: usize, scale: f32, seed: &mut u64) -> Self {
        let mut m = Self::zeros(rows, cols);
        for v in m.data.iter_mut() {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            let r = ((*seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5;
            *v = (r as f32) * scale;
        }
        m
    }

    pub fn add(&self, other: &Matrix) -> Matrix {
        if other.rows == 1 && other.cols == self.cols {
            let mut out = self.clone();
            for i in 0..self.rows {
                for j in 0..self.cols {
                    out.set(i, j, self.get(i, j) + other.get(0, j));
                }
            }
            return out;
        }
        assert_eq!((self.rows, self.cols), (other.rows, other.cols), "add boyut");
        let data = self.data.iter().zip(other.data.iter()).map(|(a, b)| a + b).collect();
        Matrix::from_vec(self.rows, self.cols, data)
    }

    pub fn sub(&self, other: &Matrix) -> Matrix {
        if other.rows == 1 && other.cols == self.cols {
            let mut out = self.clone();
            for i in 0..self.rows {
                for j in 0..self.cols {
                    out.set(i, j, self.get(i, j) - other.get(0, j));
                }
            }
            return out;
        }
        assert_eq!((self.rows, self.cols), (other.rows, other.cols), "sub boyut");
        let data = self.data.iter().zip(other.data.iter()).map(|(a, b)| a - b).collect();
        Matrix::from_vec(self.rows, self.cols, data)
    }

    pub fn mul_elem(&self, other: &Matrix) -> Matrix {
        assert_eq!((self.rows, self.cols), (other.rows, other.cols));
        let data = self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).collect();
        Matrix::from_vec(self.rows, self.cols, data)
    }

    pub fn scale(&self, s: f32) -> Matrix {
        let data = self.data.iter().map(|x| x * s).collect();
        Matrix::from_vec(self.rows, self.cols, data)
    }

    pub fn matmul(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows, "matmul boyut");
        let mut out = Matrix::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut acc = 0.0f32;
                for k in 0..self.cols {
                    acc += self.get(i, k) * other.get(k, j);
                }
                out.set(i, j, acc);
            }
        }
        out
    }

    pub fn transpose(&self) -> Matrix {
        let mut out = Matrix::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                out.set(j, i, self.get(i, j));
            }
        }
        out
    }

    pub fn map<F: Fn(f32) -> f32>(&self, f: F) -> Matrix {
        let data = self.data.iter().map(|&x| f(x)).collect();
        Matrix::from_vec(self.rows, self.cols, data)
    }

    pub fn sum_rows(&self) -> Matrix {
        let mut out = Matrix::zeros(self.rows, 1);
        for i in 0..self.rows {
            let mut s = 0.0;
            for j in 0..self.cols {
                s += self.get(i, j);
            }
            out.set(i, 0, s);
        }
        out
    }
}

impl fmt::Debug for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Matrix {}x{}:", self.rows, self.cols)?;
        for i in 0..self.rows {
            for j in 0..self.cols {
                write!(f, "{:8.4} ", self.get(i, j))?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

// ============================================================
// Tensor3D — (batch, seq_len, d_model) 3D tensor
// Tüm Matrix operasyonlarını batch üzerinden destekler.
// ============================================================

#[derive(Clone)]
pub struct Tensor3D {
    pub batch: usize,
    pub seq_len: usize,
    pub d_model: usize,
    pub data: Vec<f32>,
}

impl Tensor3D {
    pub fn zeros(batch: usize, seq_len: usize, d_model: usize) -> Self {
        Self {
            batch,
            seq_len,
            d_model,
            data: vec![0.0; batch * seq_len * d_model],
        }
    }

    pub fn filled(batch: usize, seq_len: usize, d_model: usize, v: f32) -> Self {
        Self {
            batch,
            seq_len,
            d_model,
            data: vec![v; batch * seq_len * d_model],
        }
    }

    pub fn from_vec(batch: usize, seq_len: usize, d_model: usize, data: Vec<f32>) -> Self {
        assert_eq!(batch * seq_len * d_model, data.len(), "Tensor3D boyut");
        Self { batch, seq_len, d_model, data }
    }

    #[inline]
    pub fn get(&self, b: usize, s: usize, d: usize) -> f32 {
        self.data[(b * self.seq_len + s) * self.d_model + d]
    }

    #[inline]
    pub fn set(&mut self, b: usize, s: usize, d: usize, v: f32) {
        self.data[(b * self.seq_len + s) * self.d_model + d] = v;
    }

    pub fn slice_batch(&self, b: usize) -> Matrix {
        let start = b * self.seq_len * self.d_model;
        let end = start + self.seq_len * self.d_model;
        Matrix::from_vec(self.seq_len, self.d_model, self.data[start..end].to_vec())
    }

    pub fn set_batch(&mut self, b: usize, m: &Matrix) {
        assert_eq!(m.rows, self.seq_len);
        assert_eq!(m.cols, self.d_model);
        let start = b * self.seq_len * self.d_model;
        self.data[start..start + self.seq_len * self.d_model].copy_from_slice(&m.data);
    }

    // ---- Element-wise ----

    pub fn add(&self, other: &Tensor3D) -> Tensor3D {
        assert_eq!((self.batch, self.seq_len, self.d_model), (other.batch, other.seq_len, other.d_model));
        let data = self.data.iter().zip(other.data.iter()).map(|(a, b)| a + b).collect();
        Tensor3D::from_vec(self.batch, self.seq_len, self.d_model, data)
    }

    pub fn sub(&self, other: &Tensor3D) -> Tensor3D {
        assert_eq!((self.batch, self.seq_len, self.d_model), (other.batch, other.seq_len, other.d_model));
        let data = self.data.iter().zip(other.data.iter()).map(|(a, b)| a - b).collect();
        Tensor3D::from_vec(self.batch, self.seq_len, self.d_model, data)
    }

    pub fn mul_elem(&self, other: &Tensor3D) -> Tensor3D {
        assert_eq!((self.batch, self.seq_len, self.d_model), (other.batch, other.seq_len, other.d_model));
        let data = self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).collect();
        Tensor3D::from_vec(self.batch, self.seq_len, self.d_model, data)
    }

    pub fn scale(&self, s: f32) -> Tensor3D {
        let data = self.data.iter().map(|x| x * s).collect();
        Tensor3D::from_vec(self.batch, self.seq_len, self.d_model, data)
    }

    pub fn map<F: Fn(f32) -> f32>(&self, f: F) -> Tensor3D {
        let data = self.data.iter().map(|&x| f(x)).collect();
        Tensor3D::from_vec(self.batch, self.seq_len, self.d_model, data)
    }

    // ---- Matmul ----

    /// (batch, seq_len, d_model) * (d_model, out) → (batch, seq_len, out)
    pub fn matmul(&self, w: &Matrix) -> Tensor3D {
        assert_eq!(self.d_model, w.rows, "matmul boyut");
        let mut out = Tensor3D::zeros(self.batch, self.seq_len, w.cols);
        for b in 0..self.batch {
            for s in 0..self.seq_len {
                for o in 0..w.cols {
                    let mut acc = 0.0f32;
                    for d in 0..self.d_model {
                        acc += self.get(b, s, d) * w.get(d, o);
                    }
                    out.set(b, s, o, acc);
                }
            }
        }
        out
    }

    /// (batch, seq_len, d_model) * (batch, d_model, out) → (batch, seq_len, out)
    pub fn matmul_batched(&self, w: &Tensor3D) -> Tensor3D {
        assert_eq!(self.batch, w.batch);
        assert_eq!(self.d_model, w.seq_len);
        let mut out = Tensor3D::zeros(self.batch, self.seq_len, w.d_model);
        for b in 0..self.batch {
            for s in 0..self.seq_len {
                for o in 0..w.d_model {
                    let mut acc = 0.0f32;
                    for d in 0..self.d_model {
                        acc += self.get(b, s, d) * w.get(b, d, o);
                    }
                    out.set(b, s, o, acc);
                }
            }
        }
        out
    }

    /// (batch, seq_len, d_model) * (batch, seq_len, d_model)^T → (batch, seq_len, seq_len)
    pub fn matmul_transpose(&self, other: &Tensor3D) -> Tensor3D {
        assert_eq!(self.batch, other.batch);
        assert_eq!(self.seq_len, other.seq_len);
        assert_eq!(self.d_model, other.d_model);
        let mut out = Tensor3D::zeros(self.batch, self.seq_len, other.seq_len);
        for b in 0..self.batch {
            for i in 0..self.seq_len {
                for j in 0..other.seq_len {
                    let mut acc = 0.0f32;
                    for d in 0..self.d_model {
                        acc += self.get(b, i, d) * other.get(b, j, d);
                    }
                    out.set(b, i, j, acc);
                }
            }
        }
        out
    }

    /// Bias ekle: (batch, seq_len, d_model) + (1, d_model) broadcast
    pub fn add_bias(&self, bias: &Matrix) -> Tensor3D {
        assert_eq!(bias.rows, 1);
        assert_eq!(bias.cols, self.d_model);
        let mut out = self.clone();
        for b in 0..self.batch {
            for s in 0..self.seq_len {
                for d in 0..self.d_model {
                    out.set(b, s, d, out.get(b, s, d) + bias.get(0, d));
                }
            }
        }
        out
    }

    // ---- Transpose ----

    /// (batch, seq_len, d_model) → (batch, d_model, seq_len)
    pub fn transpose(&self) -> Tensor3D {
        let mut out = Tensor3D::zeros(self.batch, self.d_model, self.seq_len);
        for b in 0..self.batch {
            for s in 0..self.seq_len {
                for d in 0..self.d_model {
                    out.set(b, d, s, self.get(b, s, d));
                }
            }
        }
        out
    }

    // ---- Sum ----

    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    pub fn sum_rows(&self) -> Tensor3D {
        let mut out = Tensor3D::zeros(self.batch, self.seq_len, 1);
        for b in 0..self.batch {
            for s in 0..self.seq_len {
                let mut acc = 0.0f32;
                for d in 0..self.d_model {
                    acc += self.get(b, s, d);
                }
                out.set(b, s, 0, acc);
            }
        }
        out
    }

    // ---- Softmax (satır bazında) ----

    pub fn softmax(&self) -> Tensor3D {
        let mut out = Tensor3D::zeros(self.batch, self.seq_len, self.d_model);
        for b in 0..self.batch {
            for s in 0..self.seq_len {
                let mut max = f32::NEG_INFINITY;
                for d in 0..self.d_model {
                    let v = self.get(b, s, d);
                    if v > max { max = v; }
                }
                let mut sum = 0.0f32;
                for d in 0..self.d_model {
                    sum += (self.get(b, s, d) - max).exp();
                }
                for d in 0..self.d_model {
                    out.set(b, s, d, (self.get(b, s, d) - max).exp() / sum);
                }
            }
        }
        out
    }

    // ---- LayerNorm ----

    pub fn layer_norm(&self, gamma: &Matrix, beta: &Matrix, eps: f32) -> Tensor3D {
        assert_eq!(gamma.cols, self.d_model);
        assert_eq!(beta.cols, self.d_model);
        let mut out = Tensor3D::zeros(self.batch, self.seq_len, self.d_model);
        for b in 0..self.batch {
            for s in 0..self.seq_len {
                let mut mean = 0.0f32;
                for d in 0..self.d_model { mean += self.get(b, s, d); }
                mean /= self.d_model as f32;

                let mut var = 0.0f32;
                for d in 0..self.d_model {
                    let diff = self.get(b, s, d) - mean;
                    var += diff * diff;
                }
                var /= self.d_model as f32;

                let inv = 1.0 / (var + eps).sqrt();
                for d in 0..self.d_model {
                    let xn = (self.get(b, s, d) - mean) * inv;
                    out.set(b, s, d, xn * gamma.get(0, d) + beta.get(0, d));
                }
            }
        }
        out
    }

    // ---- GELU ----

    pub fn gelu(&self) -> Tensor3D {
        let c = (2.0f32 / std::f32::consts::PI).sqrt();
        self.map(|x| {
            0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
        })
    }

    // ---- Causal Mask ----

    pub fn mask_causal(&self) -> Tensor3D {
        assert_eq!(self.seq_len, self.d_model, "causal mask kare olmalı");
        let mut out = self.clone();
        for b in 0..self.batch {
            for i in 0..self.seq_len {
                for j in 0..self.d_model {
                    if j > i {
                        out.set(b, i, j, -1e9);
                    }
                }
            }
        }
        out
    }
}