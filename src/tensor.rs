// src/tensor.rs
// StonefishAI — sıfırdan yazılmış row-major 2D matris.
//
// Batch'li işler (GPT eğitimi) ayrı bir 3D tipiyle değil, **satır paketleme**
// ile yapılır: (B, S, D) yerine (B*S, D) matrisi kullanılır. LayerNorm, Linear,
// GELU, bias-add ve softmax zaten satır-bazlı olduğu için bu dönüşüm bedava;
// yalnızca attention gerçek 3D yapı ister ve o tek yer `Op::CausalAttention`
// içinde batch/seq bilgisiyle çözülür. Böylece tek bir autograd motoru yeter.

use std::fmt;

#[derive(Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>, // row-major
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(rows * cols, data.len(), "boyut uyuşmuyor");
        Self { rows, cols, data }
    }

    pub fn filled(rows: usize, cols: usize, v: f32) -> Self {
        Self {
            rows,
            cols,
            data: vec![v; rows * cols],
        }
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

    /// `other` ya aynı şekildedir ya da (1, cols) — satır bazlı broadcast.
    pub fn add(&self, other: &Matrix) -> Matrix {
        if other.rows == 1 && other.cols == self.cols {
            // Bias ekleme her katmanda çalışır; dilim tabanlı zip sınır kontrolü
            // yapmaz ve vektörleşir (eski get/set döngüsü ölçümde pahalıydı).
            let bias = &other.data[..self.cols];
            let mut out = self.clone();
            for row in out.data.chunks_exact_mut(self.cols) {
                for (o, &b) in row.iter_mut().zip(bias) {
                    *o += b;
                }
            }
            return out;
        }
        assert_eq!(
            (self.rows, self.cols),
            (other.rows, other.cols),
            "add boyut"
        );
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Matrix::from_vec(self.rows, self.cols, data)
    }

    /// `other` ya aynı şekildedir ya da (1, cols) — satır bazlı broadcast.
    pub fn sub(&self, other: &Matrix) -> Matrix {
        if other.rows == 1 && other.cols == self.cols {
            let bias = &other.data[..self.cols];
            let mut out = self.clone();
            for row in out.data.chunks_exact_mut(self.cols) {
                for (o, &b) in row.iter_mut().zip(bias) {
                    *o -= b;
                }
            }
            return out;
        }
        assert_eq!(
            (self.rows, self.cols),
            (other.rows, other.cols),
            "sub boyut"
        );
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Matrix::from_vec(self.rows, self.cols, data)
    }

    pub fn mul_elem(&self, other: &Matrix) -> Matrix {
        assert_eq!(
            (self.rows, self.cols),
            (other.rows, other.cols),
            "mul_elem boyut"
        );
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Matrix::from_vec(self.rows, self.cols, data)
    }

    pub fn scale(&self, s: f32) -> Matrix {
        let data = self.data.iter().map(|x| x * s).collect();
        Matrix::from_vec(self.rows, self.cols, data)
    }

    /// Naif CPU matmul (referans / yedek yol).
    /// Performans kritik yerlerde `gpu::matmul_gpu` kullanılır.
    pub fn matmul(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows, "matmul boyut");
        let (m, k, n) = (self.rows, self.cols, other.cols);
        let mut out = Matrix::zeros(m, n);
        for i in 0..m {
            let a_row = &self.data[i * k..(i + 1) * k];
            let o_row = &mut out.data[i * n..(i + 1) * n];
            for (kk, &a) in a_row.iter().enumerate() {
                let b_row = &other.data[kk * n..(kk + 1) * n];
                for j in 0..n {
                    o_row[j] += a * b_row[j];
                }
            }
        }
        out
    }

    pub fn transpose(&self) -> Matrix {
        let (r, c) = (self.rows, self.cols);
        let mut out = Matrix::zeros(c, r);
        for i in 0..r {
            for j in 0..c {
                out.data[j * r + i] = self.data[i * c + j];
            }
        }
        out
    }

    pub fn map<F: Fn(f32) -> f32>(&self, f: F) -> Matrix {
        let data = self.data.iter().map(|&x| f(x)).collect();
        Matrix::from_vec(self.rows, self.cols, data)
    }

    /// Bir skaler değeri okurken kullanılan kısayol (yalnızca 1x1 matrislerde anlamlı).
    #[inline]
    pub fn scalar(&self) -> f32 {
        self.data[0]
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
