// src/sys/cublas.rs
// cuBLAS'ın ihtiyacımız olan alt kümesi (SGEMM + strided-batched).

#![allow(non_camel_case_types, dead_code)]

use super::{LazyFn, CUBLAS_MODULES};
use std::ffi::c_void;

pub type cublasHandle_t = *mut c_void;
pub type cublasStatus_t = i32;
pub type cublasOperation_t = i32;

pub const CUBLAS_STATUS_SUCCESS: cublasStatus_t = 0;
pub const CUBLAS_OP_N: cublasOperation_t = 0;
pub const CUBLAS_OP_T: cublasOperation_t = 1;

macro_rules! blas_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name() -> Option<$ty> {
            static F: LazyFn<$ty> = LazyFn::new(CUBLAS_MODULES, stringify!($name));
            F.get()
        }
    };
}

blas_fn!(cublasCreate_v2, unsafe extern "system" fn(*mut cublasHandle_t) -> cublasStatus_t);
blas_fn!(cublasDestroy_v2, unsafe extern "system" fn(cublasHandle_t) -> cublasStatus_t);
blas_fn!(
    cublasSetStream_v2,
    unsafe extern "system" fn(cublasHandle_t, *mut c_void) -> cublasStatus_t
);
blas_fn!(
    cublasSgemm_v2,
    unsafe extern "system" fn(
        cublasHandle_t,
        cublasOperation_t,
        cublasOperation_t,
        i32,
        i32,
        i32,
        *const f32,
        *const f32,
        i32,
        *const f32,
        i32,
        *const f32,
        *mut f32,
        i32,
    ) -> cublasStatus_t
);
blas_fn!(
    cublasSgemmStridedBatched,
    unsafe extern "system" fn(
        cublasHandle_t,
        cublasOperation_t,
        cublasOperation_t,
        i32,
        i32,
        i32,
        *const f32,
        *const f32,
        i32,
        i64,
        *const f32,
        i32,
        i64,
        *const f32,
        *mut f32,
        i32,
        i64,
        i32,
    ) -> cublasStatus_t
);

pub fn check(res: cublasStatus_t, what: &str) {
    if res != CUBLAS_STATUS_SUCCESS {
        panic!("cuBLAS hatası ({what}): kod {res}");
    }
}
