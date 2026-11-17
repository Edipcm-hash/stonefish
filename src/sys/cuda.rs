// src/sys/cuda.rs
// The subset of the CUDA Driver API (nvcuda) that we need.
//
// All signatures are taken verbatim from the CUDA driver API docs.
// Functions are loaded at runtime; returns `None` if CUDA is absent.

#![allow(non_camel_case_types, dead_code)]

use super::{LazyFn, CUDA_MODULES};
use std::ffi::c_void;

pub type CUresult = i32;
pub type CUdevice = i32;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUdeviceptr = u64;
pub type CUjit_option = u32;

pub const CUDA_SUCCESS: CUresult = 0;
pub const CU_MEMHOSTALLOC_PORTABLE: u32 = 0x01;

// Device attributes (the ones we need)
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: i32 = 76;

macro_rules! cuda_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name() -> Option<$ty> {
            static F: LazyFn<$ty> = LazyFn::new(CUDA_MODULES, stringify!($name));
            F.get()
        }
    };
}

// ---- Version / device / context ----
cuda_fn!(cuInit, unsafe extern "system" fn(u32) -> CUresult);
cuda_fn!(
    cuDriverGetVersion,
    unsafe extern "system" fn(*mut i32) -> CUresult
);
cuda_fn!(
    cuDeviceGet,
    unsafe extern "system" fn(*mut CUdevice, i32) -> CUresult
);
cuda_fn!(
    cuDeviceGetName,
    unsafe extern "system" fn(*mut i8, i32, CUdevice) -> CUresult
);
cuda_fn!(
    cuDeviceGetAttribute,
    unsafe extern "system" fn(*mut i32, i32, CUdevice) -> CUresult
);
cuda_fn!(
    cuDeviceTotalMem_v2,
    unsafe extern "system" fn(*mut usize, CUdevice) -> CUresult
);
cuda_fn!(
    cuCtxCreate_v2,
    unsafe extern "system" fn(*mut CUcontext, u32, CUdevice) -> CUresult
);
cuda_fn!(
    cuCtxDestroy_v2,
    unsafe extern "system" fn(CUcontext) -> CUresult
);
cuda_fn!(cuCtxSynchronize, unsafe extern "system" fn() -> CUresult);
cuda_fn!(
    cuCtxSetCurrent,
    unsafe extern "system" fn(CUcontext) -> CUresult
);

// ---- Bellek ----
cuda_fn!(
    cuMemAlloc_v2,
    unsafe extern "system" fn(*mut CUdeviceptr, usize) -> CUresult
);
cuda_fn!(
    cuMemFree_v2,
    unsafe extern "system" fn(CUdeviceptr) -> CUresult
);
cuda_fn!(
    cuMemcpyHtoD_v2,
    unsafe extern "system" fn(CUdeviceptr, *const c_void, usize) -> CUresult
);
cuda_fn!(
    cuMemcpyDtoH_v2,
    unsafe extern "system" fn(*mut c_void, CUdeviceptr, usize) -> CUresult
);
cuda_fn!(
    cuMemcpyDtoD_v2,
    unsafe extern "system" fn(CUdeviceptr, CUdeviceptr, usize) -> CUresult
);
cuda_fn!(
    cuMemsetD32_v2,
    unsafe extern "system" fn(CUdeviceptr, u32, usize) -> CUresult
);
cuda_fn!(
    cuMemGetInfo_v2,
    unsafe extern "system" fn(*mut usize, *mut usize) -> CUresult
);

// ---- Module / kernel ----
cuda_fn!(
    cuModuleLoadData,
    unsafe extern "system" fn(*mut CUmodule, *const c_void) -> CUresult
);
cuda_fn!(
    cuModuleLoadDataEx,
    unsafe extern "system" fn(
        *mut CUmodule,
        *const c_void,
        u32,
        *mut CUjit_option,
        *mut *mut c_void,
    ) -> CUresult
);
cuda_fn!(
    cuModuleUnload,
    unsafe extern "system" fn(CUmodule) -> CUresult
);
cuda_fn!(
    cuModuleGetFunction,
    unsafe extern "system" fn(*mut CUfunction, CUmodule, *const i8) -> CUresult
);

// ---- Kernel launch ----
cuda_fn!(
    cuLaunchKernel,
    unsafe extern "system" fn(
        CUfunction,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        CUstream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CUresult
);
cuda_fn!(
    cuGetErrorName,
    unsafe extern "system" fn(CUresult, *mut *const i8) -> CUresult
);

// ============================================================
// Safe wrappers
// ============================================================

/// Panic with a description if the result code is an error.
pub fn check(res: CUresult, what: &str) {
    if res == CUDA_SUCCESS {
        return;
    }
    let name = unsafe {
        match cuGetErrorName().and_then(|f| {
            let mut p: *const i8 = std::ptr::null();
            if f(res, &mut p) == CUDA_SUCCESS && !p.is_null() {
                Some(std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned())
            } else {
                None
            }
        }) {
            Some(s) => s,
            None => format!("kod {res}"),
        }
    };
    panic!("CUDA error ({what}): {name}");
}

/// Print device information (diagnostics).
pub fn device_info() -> Option<(String, i32, i32, usize)> {
    unsafe {
        check(cuInit()?(0), "cuInit");
        let mut dev: CUdevice = 0;
        check(cuDeviceGet()?(&mut dev, 0), "cuDeviceGet");
        let mut name = [0i8; 256];
        check(
            cuDeviceGetName()?(&mut name as *mut i8, 256, dev),
            "cuDeviceGetName",
        );
        let n = std::ffi::CStr::from_ptr(name.as_ptr())
            .to_string_lossy()
            .into_owned();
        let mut maj = 0;
        let mut min = 0;
        check(
            cuDeviceGetAttribute()?(&mut maj, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev),
            "cc major",
        );
        check(
            cuDeviceGetAttribute()?(&mut min, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev),
            "cc minor",
        );
        let mut total = 0usize;
        let _ = cuDeviceTotalMem_v2().map(|f| f(&mut total, dev));
        Some((n, maj, min, total))
    }
}
