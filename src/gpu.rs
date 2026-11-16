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
    }

    /// CPU'daki iki matrisi GPU'da çarp, sonucu CPU'ya döndür.
    pub fn matmul_gpu(a: &Matrix, b: &Matrix) -> Matrix {
        let ctx = get_ctx();
        let a_gpu = GpuMatrix::from_cpu(ctx, a);
        let b_gpu = GpuMatrix::from_cpu(ctx, b);
        let c_gpu = GpuMatrix::matmul(&a_gpu, &b_gpu);
        let stream = ctx.default_stream();
        let out = stream
            .clone_dtoh(&c_gpu.data)
            .expect("GPU->CPU transfer");
        let (rows, cols) = (c_gpu.rows, c_gpu.cols);
        let GpuMatrix { data, .. } = c_gpu;
        recycle(data);
        Matrix::from_vec(rows, cols, out)
    }
}

#[cfg(feature = "cuda")]
pub use cuda_impl::{get_ctx, matmul_gpu, GpuMatrix};

/// CUDA kapalıyken: aynı arayüz, CPU yolu.
#[cfg(not(feature = "cuda"))]
pub fn matmul_gpu(a: &Matrix, b: &Matrix) -> Matrix {
    a.matmul(b)
}

/// Bu derlemede GPU yolu var mı?
pub const HAS_CUDA: bool = cfg!(feature = "cuda");
