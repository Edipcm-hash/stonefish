// src/sys/mod.rs
// Zero dependencies: direct FFI to the CUDA driver.
//
// `cudarc` was removed. This module uses the operating system's dynamic
// library loader (Windows: kernel32!LoadLibraryW/GetProcAddress,
// Linux: libdl!dlopen/dlsym) to load the CUDA driver, cuBLAS and
// NVRTC at RUNTIME.
//
// Benefits:
//  * `[dependencies]` is empty - no external crates
//  * No CUDA SDK needed at build time (no import lib)
//  * Returns `None` gracefully if CUDA is absent, falls back to the CPU path

#![allow(non_snake_case)]

use std::ffi::c_void;
use std::sync::OnceLock;

// ============================================================
// Dynamic library loader
// ============================================================

#[cfg(windows)]
mod loader {
    use std::ffi::c_void;

    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryW(name: *const u16) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
        fn FreeLibrary(module: *mut c_void) -> i32;
    }

    pub fn open(name: &str) -> Option<*mut c_void> {
        let wide: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
        let h = unsafe { LoadLibraryW(wide.as_ptr()) };
        if h.is_null() {
            None
        } else {
            Some(h)
        }
    }

    pub fn sym(module: *mut c_void, name: &str) -> Option<*mut c_void> {
        let c = std::ffi::CString::new(name).ok()?;
        let p = unsafe { GetProcAddress(module, c.as_ptr() as *const u8) };
        if p.is_null() {
            None
        } else {
            Some(p)
        }
    }

    #[allow(dead_code)]
    pub fn close(module: *mut c_void) {
        unsafe {
            FreeLibrary(module);
        }
    }
}

#[cfg(unix)]
mod loader {
    use std::ffi::c_void;

    #[link(name = "dl")]
    extern "C" {
        fn dlopen(name: *const i8, flag: i32) -> *mut c_void;
        fn dlsym(module: *mut c_void, name: *const i8) -> *mut c_void;
        fn dlclose(module: *mut c_void) -> i32;
    }
    const RTLD_NOW: i32 = 2;

    pub fn open(name: &str) -> Option<*mut c_void> {
        let c = std::ffi::CString::new(name).ok()?;
        let h = unsafe { dlopen(c.as_ptr(), RTLD_NOW) };
        if h.is_null() {
            None
        } else {
            Some(h)
        }
    }

    pub fn sym(module: *mut c_void, name: &str) -> Option<*mut c_void> {
        let c = std::ffi::CString::new(name).ok()?;
        let p = unsafe { dlsym(module, c.as_ptr()) };
        if p.is_null() {
            None
        } else {
            Some(p)
        }
    }

    #[allow(dead_code)]
    pub fn close(module: *mut c_void) {
        unsafe {
            dlclose(module);
        }
    }
}

/// Load a library by name (the first name that resolves wins).
pub fn library(names: &[&str]) -> Option<*mut c_void> {
    for n in names {
        if let Some(h) = loader::open(n) {
            return Some(h);
        }
    }
    None
}

/// Get a function pointer from the library.
///
/// # Safety
/// The returned pointer must match the signature the caller declares;
/// calling it with the wrong signature is undefined behaviour.
pub unsafe fn symbol(module: *mut c_void, name: &str) -> Option<*mut c_void> {
    loader::sym(module, name)
}

/// Cast a function pointer to a typed `fn`.
///
/// # Safety
/// `T` must be exactly the signature that `p` really points to.
/// Otherwise the call is undefined behaviour.
pub unsafe fn cast<T: Copy>(p: *mut c_void) -> T {
    debug_assert_eq!(
        std::mem::size_of::<T>(),
        std::mem::size_of::<*mut c_void>(),
        "function pointer size"
    );
    std::mem::transmute_copy(&p)
}

// ============================================================
// Lazily loaded function pointer
// ============================================================

pub struct LazyFn<T> {
    module_names: &'static [&'static str],
    symbol_name: &'static str,
    cell: OnceLock<Option<T>>,
}

impl<T: Copy> LazyFn<T> {
    pub const fn new(module_names: &'static [&'static str], symbol_name: &'static str) -> Self {
        Self {
            module_names,
            symbol_name,
            cell: OnceLock::new(),
        }
    }

    pub fn get(&self) -> Option<T> {
        *self.cell.get_or_init(|| unsafe {
            let h = load_module(self.module_names)?;
            let p = symbol(h, self.symbol_name)?;
            Some(cast::<T>(p))
        })
    }
}

fn load_module(names: &'static [&'static str]) -> Option<*mut c_void> {
    static MODULES: OnceLock<std::sync::Mutex<Vec<(&'static str, usize)>>> = OnceLock::new();
    let cache = MODULES.get_or_init(|| std::sync::Mutex::new(Vec::new()));
    {
        let g = cache.lock().unwrap();
        for (n, h) in g.iter() {
            if *n == names[0] {
                return if *h == 0 {
                    None
                } else {
                    Some(*h as *mut c_void)
                };
            }
        }
    }
    let h = library(names);
    let mut g = cache.lock().unwrap();
    g.push((names[0], h.map(|p| p as usize).unwrap_or(0)));
    h
}

pub const CUDA_MODULES: &[&str] = &["nvcuda.dll", "libcuda.so.1", "libcuda.so"];
pub const CUBLAS_MODULES: &[&str] = &[
    "cublas64_13.dll",
    "cublas64_12.dll",
    "libcublas.so.12",
    "libcublas.so",
];
pub const NVRTC_MODULES: &[&str] = &[
    "nvrtc64_130_0.dll",
    "nvrtc64_120_0.dll",
    "libnvrtc.so.12",
    "libnvrtc.so",
];

pub mod cublas;
pub mod cuda;
pub mod dev;
pub mod nvrtc;
pub mod selftest;
