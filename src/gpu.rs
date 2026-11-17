// src/gpu.rs
// GPU matmul (cuBLAS) + CPU yedeği.
//
// `cuda` feature'ı kapalıyken (--no-default-features) her şey CPU'da çalışır;
// bu, CUDA'sı olmayan makinelerde projenin derlenip test edilebilmesini sağlar.
//
// Performans notu: bu motor CPU'yu "gerçeğin kaynağı" olarak tutar ve her matmul
// için CPU→GPU→CPU turu yapar. Bu ancak GEMM büyüdükçe kârlı olur; satır
// paketleme sayesinde tipik şekil (batch*seq × d_model) olduğundan (ör. 1024×128)
// transfer maliyeti hesaplamaya göre küçülür. Ölçüm: `test-matmul-bench`.

use crate::tensor::Matrix;

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::Matrix;
    use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
    use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
    use std::sync::{Arc, Mutex, OnceLock};

    static CTX: OnceLock<Arc<CudaContext>> = OnceLock::new();
    static BLAS: OnceLock<Mutex<CudaBlas>> = OnceLock::new();
    /// Aynı boyutlu GEMM çıktıları için geri dönüşüm havuzu.
    /// Model içinde şekiller az sayıda olduğundan isabet oranı yüksek.
    static POOL: OnceLock<Mutex<Vec<CudaSlice<f32>>>> = OnceLock::new();

    /// Küresel cuBLAS handle'ı (cihaz GEMM'leri de bunu paylaşır).
    pub fn blas() -> &'static Mutex<CudaBlas> {
        let stream = get_ctx().default_stream();
        BLAS.get_or_init(|| Mutex::new(CudaBlas::new(stream).expect("CudaBlas handle")))
    }

    pub fn get_ctx() -> &'static Arc<CudaContext> {
        CTX.get_or_init(|| CudaContext::new(0).expect("CUDA context açılamadı"))
    }

    fn alloc_pooled(stream: &Arc<CudaStream>, len: usize) -> CudaSlice<f32> {
        let pool = POOL.get_or_init(|| Mutex::new(Vec::new()));
        {
            let mut p = pool.lock().unwrap();
            if let Some(i) = p.iter().position(|b| b.len() == len) {
                return p.swap_remove(i);
            }
        }
        stream.alloc_zeros::<f32>(len).expect("GPU alloc")
    }

    fn recycle(buf: CudaSlice<f32>) {
        let pool = POOL.get_or_init(|| Mutex::new(Vec::new()));
        let mut p = pool.lock().unwrap();
        // Üst sınır: model başına birkaç farklı şekil var; 16 tampon fazlasıyla
        // yeter ve belleği (en büyük tampon × 16) ile sınırlar.
        if p.len() < 16 {
            p.push(buf);
        }
    }

    pub struct GpuMatrix {
        pub rows: usize,
        pub cols: usize,
        pub data: CudaSlice<f32>,
        pub ctx: Arc<CudaContext>,
    }

    impl GpuMatrix {
        pub fn from_cpu(ctx: &Arc<CudaContext>, m: &Matrix) -> Self {
            let stream = ctx.default_stream();
            let data = stream.clone_htod(&m.data).expect("CPU->GPU transfer");
            Self {
                rows: m.rows,
                cols: m.cols,
                data,
                ctx: ctx.clone(),
            }
        }

        pub fn to_cpu(&self) -> Matrix {
            let stream = self.ctx.default_stream();
            let data = stream.clone_dtoh(&self.data).expect("GPU->CPU transfer");
            Matrix::from_vec(self.rows, self.cols, data)
        }

        /// C = A·B. Row-major girdiler için cuBLAS'ın kolon-major dünyasında
        /// C^T = B^T·A^T yazımı kullanılır (operandlar yer değiştirir).
        pub fn matmul(a: &GpuMatrix, b: &GpuMatrix) -> GpuMatrix {
            assert_eq!(a.cols, b.rows, "matmul boyut");
            let (m, n, k) = (a.rows, b.cols, a.cols);

            let stream = a.ctx.default_stream();
            let blas_mutex = BLAS.get_or_init(|| {
                Mutex::new(CudaBlas::new(stream.clone()).expect("CudaBlas handle"))
            });
            let blas = blas_mutex.lock().unwrap();

            let mut c_dev = alloc_pooled(&stream, m * n);

            let cfg = GemmConfig {
                transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                m: n as i32,
                n: m as i32,
                k: k as i32,
                alpha: 1.0,
                beta: 0.0,
                lda: n as i32,
                ldb: k as i32,
                ldc: n as i32,
            };

            unsafe {
                blas.gemm(cfg, &b.data, &a.data, &mut c_dev).expect("gemm");
            }

            GpuMatrix {
                rows: m,
                cols: n,
                data: c_dev,
                ctx: a.ctx.clone(),
            }
        }
        /// C = A·Bᵀ  (A: m×k, B: n×k → C: m×n)
        /// Transpose'u malzemeleştirmeden yapar: autograd'ın `dA = dOut·Bᵀ` yolu.
        pub fn matmul_bt(a: &GpuMatrix, b: &GpuMatrix) -> GpuMatrix {
            assert_eq!(a.cols, b.cols, "matmul_bt boyut");
            // a: m×k, b: n×k → C: m×n
            let (m, n, k) = (a.rows, b.rows, a.cols);
            let stream = a.ctx.default_stream();
            let blas_mutex = BLAS.get_or_init(|| {
                Mutex::new(CudaBlas::new(stream.clone()).expect("CudaBlas handle"))
            });
            let blas = blas_mutex.lock().unwrap();
            let mut c_dev = alloc_pooled(&stream, m * n);

            // Row-major C(m,n) = A(m,k)·B(n,k)ᵀ; kolon-major dünyada
            // C_c(n,m) = B(n,k)·A(k,m)ᵀ → op(A_c)=Bᵀ (stored k×n), op(B_c)=A (stored k×m).
            let cfg = GemmConfig {
                transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
                transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                m: n as i32,
                n: m as i32,
                k: k as i32,
                alpha: 1.0,
                beta: 0.0,
                lda: k as i32,
                ldb: k as i32,
                ldc: n as i32,
            };
            unsafe {
                blas.gemm(cfg, &b.data, &a.data, &mut c_dev).expect("gemm_bt");
            }
            GpuMatrix { rows: m, cols: n, data: c_dev, ctx: a.ctx.clone() }
        }

        /// C = Aᵀ·B  (A: k×m, B: k×n → C: m×n)
        /// autograd'ın `dB = Aᵀ·dOut` yolu.
        pub fn matmul_ta(a: &GpuMatrix, b: &GpuMatrix) -> GpuMatrix {
            assert_eq!(a.rows, b.rows, "matmul_ta boyut");
            let (m, n, k) = (a.cols, b.cols, a.rows);
            let stream = a.ctx.default_stream();
            let blas_mutex = BLAS.get_or_init(|| {
                Mutex::new(CudaBlas::new(stream.clone()).expect("CudaBlas handle"))
            });
            let blas = blas_mutex.lock().unwrap();
            let mut c_dev = alloc_pooled(&stream, m * n);

            // Row-major C(m,n) = A(k,m)ᵀ·B(k,n); kolon-major dünyada
            // C_c(n,m) = B(n,k)ᵀ·A(k,m) → op(A_c)=B (stored n×k), op(B_c)=Aᵀ.
            let cfg = GemmConfig {
                transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
                m: n as i32,
                n: m as i32,
                k: k as i32,
                alpha: 1.0,
                beta: 0.0,
                lda: n as i32,
                ldb: m as i32,
                ldc: n as i32,
            };
            unsafe {
                blas.gemm(cfg, &b.data, &a.data, &mut c_dev).expect("gemm_ta");
            }
            GpuMatrix { rows: m, cols: n, data: c_dev, ctx: a.ctx.clone() }
        }
    }
    pub fn matmul_gpu(a: &Matrix, b: &Matrix) -> Matrix {
        let prof = crate::profile::enabled();
        let t0 = if prof { Some(std::time::Instant::now()) } else { None };
        let mut transfer = 0u64;

        let ctx = get_ctx();
        let t_up0 = std::time::Instant::now();
        let a_gpu = GpuMatrix::from_cpu(ctx, a);
        let b_gpu = GpuMatrix::from_cpu(ctx, b);
        if prof {
            transfer += t_up0.elapsed().as_nanos() as u64;
        }

        let c_gpu = GpuMatrix::matmul(&a_gpu, &b_gpu);

        let stream = ctx.default_stream();
        let t_down0 = std::time::Instant::now();
        let out = stream
            .clone_dtoh(&c_gpu.data)
            .expect("GPU->CPU transfer");
        if prof {
            transfer += t_down0.elapsed().as_nanos() as u64;
        }

        let (rows, cols) = (c_gpu.rows, c_gpu.cols);
        let GpuMatrix { data, .. } = c_gpu;
        recycle(data);

        if let Some(t0) = t0 {
            crate::profile::add_matmul(t0.elapsed().as_nanos() as u64, transfer);
        }
        Matrix::from_vec(rows, cols, out)
    }
    /// C = A·Bᵀ. Transpose malzemeleştirmeden (cuBLAS `transa`).
    pub fn matmul_bt_gpu(a: &Matrix, b: &Matrix) -> Matrix {
        let prof = crate::profile::enabled();
        let t0 = if prof { Some(std::time::Instant::now()) } else { None };
        let ctx = get_ctx();
        let t_up = std::time::Instant::now();
        let a_gpu = GpuMatrix::from_cpu(ctx, a);
        let b_gpu = GpuMatrix::from_cpu(ctx, b);
        let up_ns = t_up.elapsed().as_nanos() as u64;
        let c_gpu = GpuMatrix::matmul_bt(&a_gpu, &b_gpu);
        let t_down = std::time::Instant::now();
        let out = ctx
            .default_stream()
            .clone_dtoh(&c_gpu.data)
            .expect("GPU->CPU transfer");
        let transfer = up_ns + t_down.elapsed().as_nanos() as u64;
        let (rows, cols) = (c_gpu.rows, c_gpu.cols);
        let GpuMatrix { data, .. } = c_gpu;
        recycle(data);
        if let Some(t0) = t0 {
            crate::profile::add_matmul(t0.elapsed().as_nanos() as u64, transfer);
        }
        Matrix::from_vec(rows, cols, out)
    }

    /// C = Aᵀ·B. Transpose malzemeleştirmeden (cuBLAS `transb`).
    pub fn matmul_ta_gpu(a: &Matrix, b: &Matrix) -> Matrix {
        let prof = crate::profile::enabled();
        let t0 = if prof { Some(std::time::Instant::now()) } else { None };
        let ctx = get_ctx();
        let t_up = std::time::Instant::now();
        let a_gpu = GpuMatrix::from_cpu(ctx, a);
        let b_gpu = GpuMatrix::from_cpu(ctx, b);
        let up_ns = t_up.elapsed().as_nanos() as u64;
        let c_gpu = GpuMatrix::matmul_ta(&a_gpu, &b_gpu);
        let t_down = std::time::Instant::now();
        let out = ctx
            .default_stream()
            .clone_dtoh(&c_gpu.data)
            .expect("GPU->CPU transfer");
        let transfer = up_ns + t_down.elapsed().as_nanos() as u64;
        let (rows, cols) = (c_gpu.rows, c_gpu.cols);
        let GpuMatrix { data, .. } = c_gpu;
        recycle(data);
        if let Some(t0) = t0 {
            crate::profile::add_matmul(t0.elapsed().as_nanos() as u64, transfer);
        }
        Matrix::from_vec(rows, cols, out)
    }
}

#[cfg(feature = "cuda")]
pub use cuda_impl::{get_ctx, matmul_bt_gpu, matmul_gpu, matmul_ta_gpu, GpuMatrix};

// ============================================================
// Cihaz-üstü (transfer yapmayan) GEMM'ler
// ============================================================
// Bunlar GPT eğitim döngüsünün omurgası: tüm ara sonuçlar cihazda kalır.
// Şekiller row-major; "row stride" satırlar arası mesafe (view'lar için
// genişlikten büyük olabilir), "bstride" batch'ler arası mesafe.

/// Row-major GEMM modu (C = op(A)·op(B)).
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, PartialEq)]
pub enum GemmMode {
    /// C = A·B     (A: m×k, B: k×n)
    Nn,
    /// C = A·Bᵀ    (A: m×k, B: n×k)
    Nt,
    /// C = Aᵀ·B    (A: k×m, B: k×n)
    Tn,
}

/// Tek (batch'siz) cihaz GEMM'i.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn dev_gemm<A, B, C>(
    c: &mut C,
    a: &A,
    b: &B,
    m: usize,
    n: usize,
    k: usize,
    a_row: usize,
    b_row: usize,
    c_row: usize,
    mode: GemmMode,
) where
    A: cudarc::driver::DevicePtr<f32>,
    B: cudarc::driver::DevicePtr<f32>,
    C: cudarc::driver::DevicePtrMut<f32>,
{
    dev_gemm_batched(c, a, b, m, n, k, a_row, b_row, c_row, 0, 0, 0, 1, 0.0, mode);
}

/// Strided-batched cihaz GEMM'i: C_i = op(A_i)·op(B_i).
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn dev_gemm_batched<A, B, C>(
    c: &mut C,
    a: &A,
    b: &B,
    m: usize,
    n: usize,
    k: usize,
    a_row: usize,
    b_row: usize,
    c_row: usize,
    a_bstride: usize,
    b_bstride: usize,
    c_bstride: usize,
    batch: usize,
    beta: f32,
    mode: GemmMode,
) where
    A: cudarc::driver::DevicePtr<f32>,
    B: cudarc::driver::DevicePtr<f32>,
    C: cudarc::driver::DevicePtrMut<f32>,
{
    use cudarc::cublas::{Gemm, GemmConfig, StridedBatchedConfig};
    use cudarc::cublas::sys::cublasOperation_t as Ops;

    let ctx = get_ctx();
    let stream = ctx.default_stream();
    let blas_mutex = cuda_impl::blas();
    let blas = blas_mutex.lock().unwrap();

    // Kolon-major dünyada C(m,n) = op(A)·op(B) yazımı:
    //   Cᵀ(n,m) = op(B)ᵀ · op(A)ᵀ  → operandlar yer değiştirir.
    let (transa, transb) = match mode {
        GemmMode::Nn => (Ops::CUBLAS_OP_N, Ops::CUBLAS_OP_N),
        GemmMode::Nt => (Ops::CUBLAS_OP_T, Ops::CUBLAS_OP_N),
        GemmMode::Tn => (Ops::CUBLAS_OP_N, Ops::CUBLAS_OP_T),
    };
    let cfg = StridedBatchedConfig {
        gemm: GemmConfig {
            transa,
            transb,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: 1.0,
            lda: b_row as i32,
            ldb: a_row as i32,
            beta,
            ldc: c_row as i32,
        },
        batch_size: batch as i32,
        stride_a: b_bstride as i64,
        stride_b: a_bstride as i64,
        stride_c: c_bstride as i64,
    };
    unsafe { blas.gemm_strided_batched(cfg, b, a, c) }.expect("dev_gemm_batched");
    let _ = stream;
}

/// CUDA kapalıyken: aynı arayüz, CPU yolu.
#[cfg(not(feature = "cuda"))]
pub fn matmul_gpu(a: &Matrix, b: &Matrix) -> Matrix {
    let prof = crate::profile::enabled();
    let t0 = if prof { Some(std::time::Instant::now()) } else { None };
    let out = a.matmul(b);
    if let Some(t0) = t0 {
        crate::profile::add_matmul(t0.elapsed().as_nanos() as u64, 0);
    }
    out
}

#[cfg(not(feature = "cuda"))]
pub fn matmul_bt_gpu(a: &Matrix, b: &Matrix) -> Matrix {
    a.matmul(&b.transpose())
}

#[cfg(not(feature = "cuda"))]
pub fn matmul_ta_gpu(a: &Matrix, b: &Matrix) -> Matrix {
    a.transpose().matmul(b)
}

/// Bu derlemede GPU yolu var mı?
pub const HAS_CUDA: bool = cfg!(feature = "cuda");
