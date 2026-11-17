// src/gpu.rs
// GPU matmul — cuBLAS üzerinden, HARİCİ CRATE YOK (kendi FFI'miz, src/sys).
//
// `cuda` feature'ı kapalıyken her şey CPU'da çalışır; bu, CUDA'sı olmayan
// makinelerde projenin derlenip test edilebilmesini sağlar.
//
// Row-major ↔ kolon-major köprüsü: cuBLAS kolon-major olduğu için
//   C(m,n) = A·B  ⇒  Cᵀ = Bᵀ·Aᵀ  (operandlar yer değişir)
// ve transa/transb ile ifade edilir. Böylece transpose malzemeleştirilmez.

use crate::tensor::Matrix;

/// Bu derlemede GPU yolu var mı?
pub const HAS_CUDA: bool = cfg!(feature = "cuda");

#[cfg(feature = "cuda")]
mod imp {
    use super::Matrix;
    use crate::sys::cublas;
    use crate::sys::dev::{self, DevBuf, DevPtr, Function, Kargs, Module};
    use std::sync::OnceLock;

    /// Row-major GEMM modu (C = op(A)·op(B)).
    #[derive(Clone, Copy, PartialEq)]
    pub enum GemmMode {
        /// C = A·B     (A: m×k, B: k×n)
        Nn,
        /// C = A·Bᵀ    (A: m×k, B: n×k)
        Nt,
        /// C = Aᵀ·B    (A: k×m, B: k×n)
        Tn,
    }

    // ---- cuBLAS tutamacı ----
    struct Blas(cublas::cublasHandle_t);
    unsafe impl Send for Blas {}
    unsafe impl Sync for Blas {}
    static BLAS: OnceLock<Blas> = OnceLock::new();

    fn blas() -> cublas::cublasHandle_t {
        BLAS.get_or_init(|| unsafe {
            let mut h: cublas::cublasHandle_t = std::ptr::null_mut();
            cublas::check(
                cublas::cublasCreate_v2().expect("cublasCreate")(&mut h),
                "cublasCreate",
            );
            Blas(h)
        })
        .0
    }

    /// Tek (batch'siz) cihaz GEMM'i.
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
        A: DevPtr,
        B: DevPtr,
        C: DevPtr,
    {
        dev_gemm_batched(c, a, b, m, n, k, a_row, b_row, c_row, 0, 0, 0, 1, 0.0, mode);
    }

    /// Strided-batched cihaz GEMM'i: C_i = op(A_i)·op(B_i).
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
        A: DevPtr,
        B: DevPtr,
        C: DevPtr,
    {
        // Kolon-major dünyada operandlar yer değiştirir.
        let (transa, transb) = match mode {
            GemmMode::Nn => (cublas::CUBLAS_OP_N, cublas::CUBLAS_OP_N),
            GemmMode::Nt => (cublas::CUBLAS_OP_T, cublas::CUBLAS_OP_N),
            GemmMode::Tn => (cublas::CUBLAS_OP_N, cublas::CUBLAS_OP_T),
        };
        let alpha = 1.0f32;
        let h = blas();

        unsafe {
            if batch <= 1 {
                cublas::check(
                    cublas::cublasSgemm_v2().expect("cublasSgemm")(
                        h,
                        transa,
                        transb,
                        n as i32,
                        m as i32,
                        k as i32,
                        &alpha,
                        b.dptr() as *const f32,
                        b_row as i32,
                        a.dptr() as *const f32,
                        a_row as i32,
                        &beta,
                        c.dptr() as *mut f32,
                        c_row as i32,
                    ),
                    "cublasSgemm",
                );
            } else {
                cublas::check(
                    cublas::cublasSgemmStridedBatched().expect("cublasSgemmStridedBatched")(
                        h,
                        transa,
                        transb,
                        n as i32,
                        m as i32,
                        k as i32,
                        &alpha,
                        b.dptr() as *const f32,
                        b_row as i32,
                        b_bstride as i64,
                        a.dptr() as *const f32,
                        a_row as i32,
                        a_bstride as i64,
                        &beta,
                        c.dptr() as *mut f32,
                        c_row as i32,
                        c_bstride as i64,
                        batch as i32,
                    ),
                    "cublasSgemmStridedBatched",
                );
            }
        }
    }

    /// CPU'daki iki matrisi GPU'da çarp (basit yol).
    pub fn matmul_gpu(a: &Matrix, b: &Matrix) -> Matrix {
        let mut c = DevBuf::zeros(a.rows * b.cols);
        let da = DevBuf::from_host(&a.data);
        let db = DevBuf::from_host(&b.data);
        dev_gemm(
            &mut c,
            &da,
            &db,
            a.rows,
            b.cols,
            a.cols,
            a.cols,
            b.cols,
            b.cols,
            GemmMode::Nn,
        );
        let out = c.to_host();
        Matrix::from_vec(a.rows, b.cols, out)
    }

    /// C = A·Bᵀ (transpose malzemeleştirmeden).
    pub fn matmul_bt_gpu(a: &Matrix, b: &Matrix) -> Matrix {
        let mut c = DevBuf::zeros(a.rows * b.rows);
        let da = DevBuf::from_host(&a.data);
        let db = DevBuf::from_host(&b.data);
        dev_gemm(
            &mut c,
            &da,
            &db,
            a.rows,
            b.rows,
            a.cols,
            a.cols,
            b.cols,
            b.rows,
            GemmMode::Nt,
        );
        Matrix::from_vec(a.rows, b.rows, c.to_host())
    }

    /// C = Aᵀ·B (transpose malzemeleştirmeden).
    pub fn matmul_ta_gpu(a: &Matrix, b: &Matrix) -> Matrix {
        let mut c = DevBuf::zeros(a.cols * b.cols);
        let da = DevBuf::from_host(&a.data);
        let db = DevBuf::from_host(&b.data);
        // Tn: A (k×m) gibi davranır → m = a.cols, n = b.cols, k = a.rows
        dev_gemm(
            &mut c,
            &da,
            &db,
            a.cols,
            b.cols,
            a.rows,
            a.cols,
            b.cols,
            b.cols,
            GemmMode::Tn,
        );
        Matrix::from_vec(a.cols, b.cols, c.to_host())
    }

    // ---- CUDA çekirdek modülü (NVRTC ile çalışma zamanında) ----

    static MODULE: OnceLock<Module> = OnceLock::new();

    fn module() -> &'static Module {
        MODULE.get_or_init(|| {
            let src = crate::gpu_kernels::SRC;
            let ptx = compile_cached(src);
            Module::from_ptx(&ptx).unwrap_or_else(|e| panic!("GPU modülü yüklenemedi: {e}"))
        })
    }

    /// Çekirdek fonksiyonunu al (paniklersiz).
    pub fn kernel(name: &str) -> Function {
        module()
            .func(name)
            .unwrap_or_else(|e| panic!("çekirdek yüklenemedi: {name}: {e}"))
    }

    fn compile_cached(src: &str) -> String {
        use crate::sys::nvrtc;
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let (maj, min) = dev::compute_capability();
        let arch: &'static str = match (maj, min) {
            (12, _) => "compute_120",
            (10, _) => "compute_100",
            (9, _) => "compute_90",
            (8, 9) => "compute_89",
            (8, 6) => "compute_86",
            (8, 0) => "compute_80",
            _ => "compute_90",
        };

        let mut h = DefaultHasher::new();
        src.hash(&mut h);
        arch.hash(&mut h);
        crate::VERSION.hash(&mut h);
        let key = format!("{:016x}", h.finish());

        let dir = std::path::Path::new("target").join("gpu-cache");
        let path = dir.join(format!("kernels-{key}.ptx"));
        if let Ok(s) = std::fs::read_to_string(&path) {
            return s;
        }

        let opt = format!("--gpu-architecture={arch}");
        let ptx = nvrtc::compile(src, &[&opt, "--fmad=true", "--use_fast_math"])
            .unwrap_or_else(|e| panic!("CUDA çekirdekleri derlenemedi (NVRTC): {e}"));
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(&path, &ptx);
        ptx
    }

    /// 1D grid: 256 thread/blok.
    pub fn cfg_1d(n: usize) -> ((u32, u32, u32), (u32, u32, u32)) {
        let threads = 256u32;
        let blocks = ((n as u64).div_ceil(threads as u64)).max(1) as u32;
        ((blocks, 1, 1), (threads, 1, 1))
    }

    /// Satır başına bir blok.
    pub fn cfg_rows(rows: usize, threads: u32) -> ((u32, u32, u32), (u32, u32, u32)) {
        ((rows as u32, 1, 1), (threads, 1, 1))
    }

    /// Çekirdek başlatma kısayolu.
    pub fn launch(name: &str, grid: ((u32, u32, u32), (u32, u32, u32)), shared: u32, args: &mut Kargs) {
        dev::launch(kernel(name), grid.0, grid.1, shared, args);
    }
}

#[cfg(feature = "cuda")]
pub use imp::{
    cfg_1d, cfg_rows, dev_gemm, dev_gemm_batched, kernel, launch, matmul_bt_gpu, matmul_gpu,
    matmul_ta_gpu, GemmMode,
};

/// CUDA kapalıyken: aynı arayüz, CPU yolu.
#[cfg(not(feature = "cuda"))]
pub fn matmul_gpu(a: &Matrix, b: &Matrix) -> Matrix {
    a.matmul(b)
}

#[cfg(not(feature = "cuda"))]
pub fn matmul_bt_gpu(a: &Matrix, b: &Matrix) -> Matrix {
    a.matmul(&b.transpose())
}

#[cfg(not(feature = "cuda"))]
pub fn matmul_ta_gpu(a: &Matrix, b: &Matrix) -> Matrix {
    a.transpose().matmul(b)
}
