// src/gpu.rs
// GPU matmul — cublas ile, global context ve handle cache'li.

use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{CudaContext, CudaSlice};
use std::sync::{Arc, Mutex, OnceLock};
use crate::tensor::Matrix;

static CTX: OnceLock<Arc<CudaContext>> = OnceLock::new();
static BLAS: OnceLock<Mutex<CudaBlas>> = OnceLock::new();

pub fn get_ctx() -> &'static Arc<CudaContext> {
    CTX.get_or_init(|| CudaContext::new(0).expect("CUDA context"))
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
        Self { rows: m.rows, cols: m.cols, data, ctx: ctx.clone() }
    }

    pub fn to_cpu(&self) -> Matrix {
        let stream = self.ctx.default_stream();
        let data = stream.clone_dtoh(&self.data).expect("GPU->CPU transfer");
        Matrix::from_vec(self.rows, self.cols, data)
    }

    pub fn matmul(a: &GpuMatrix, b: &GpuMatrix) -> GpuMatrix {
        assert_eq!(a.cols, b.rows, "matmul boyut");
        let m = a.rows;
        let n = b.cols;
        let k = a.cols;

        let stream = a.ctx.default_stream();
        let blas_mutex = BLAS.get_or_init(|| {
            Mutex::new(CudaBlas::new(stream.clone()).expect("CudaBlas handle"))
        });
        let blas = blas_mutex.lock().unwrap();

        let mut c_dev = stream.alloc_zeros::<f32>(m * n).expect("alloc C");

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
    c_gpu.to_cpu()
}