// src/sys/nvrtc.rs
// The subset of NVRTC (the runtime CUDA C compiler) that we need.
//
// The only way to compile CUDA kernels on machines without a host C++ compiler.

#![allow(non_camel_case_types, dead_code)]

use super::{LazyFn, NVRTC_MODULES};
use std::ffi::c_void;

pub type nvrtcProgram = *mut c_void;
pub type nvrtcResult = i32;

pub const NVRTC_SUCCESS: nvrtcResult = 0;

macro_rules! nvrtc_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name() -> Option<$ty> {
            static F: LazyFn<$ty> = LazyFn::new(NVRTC_MODULES, stringify!($name));
            F.get()
        }
    };
}

nvrtc_fn!(
    nvrtcCreateProgram,
    unsafe extern "system" fn(
        *mut nvrtcProgram,
        *const i8,
        *const i8,
        i32,
        *const *const i8,
        *const *const i8,
    ) -> nvrtcResult
);
nvrtc_fn!(
    nvrtcCompileProgram,
    unsafe extern "system" fn(nvrtcProgram, i32, *const *const i8) -> nvrtcResult
);
nvrtc_fn!(
    nvrtcGetPTXSize,
    unsafe extern "system" fn(nvrtcProgram, *mut usize) -> nvrtcResult
);
nvrtc_fn!(
    nvrtcGetPTX,
    unsafe extern "system" fn(nvrtcProgram, *mut i8) -> nvrtcResult
);
nvrtc_fn!(
    nvrtcGetProgramLogSize,
    unsafe extern "system" fn(nvrtcProgram, *mut usize) -> nvrtcResult
);
nvrtc_fn!(
    nvrtcGetProgramLog,
    unsafe extern "system" fn(nvrtcProgram, *mut i8) -> nvrtcResult
);
nvrtc_fn!(
    nvrtcDestroyProgram,
    unsafe extern "system" fn(*mut nvrtcProgram) -> nvrtcResult
);

/// Compile CUDA C source to PTX.
pub fn compile(src: &str, opts: &[&str]) -> Result<String, String> {
    unsafe {
        let create = nvrtcCreateProgram().ok_or("nvrtc not found")?;
        let compile_f = nvrtcCompileProgram().ok_or("nvrtcCompileProgram unavailable")?;
        let ptx_size = nvrtcGetPTXSize().ok_or("nvrtcGetPTXSize unavailable")?;
        let get_ptx = nvrtcGetPTX().ok_or("nvrtcGetPTX unavailable")?;
        let log_size = nvrtcGetProgramLogSize().ok_or("nvrtcGetProgramLogSize unavailable")?;
        let get_log = nvrtcGetProgramLog().ok_or("nvrtcGetProgramLog unavailable")?;
        let destroy = nvrtcDestroyProgram().ok_or("nvrtcDestroyProgram unavailable")?;

        let cs = std::ffi::CString::new(src).map_err(|e| e.to_string())?;
        let cname = std::ffi::CString::new("stonefish").unwrap();
        let mut prog: nvrtcProgram = std::ptr::null_mut();
        if create(
            &mut prog,
            cs.as_ptr(),
            cname.as_ptr(),
            0,
            std::ptr::null(),
            std::ptr::null(),
        ) != NVRTC_SUCCESS
        {
            return Err("nvrtcCreateProgram failed".into());
        }

        let c_opts: Vec<std::ffi::CString> = opts
            .iter()
            .filter_map(|o| std::ffi::CString::new(*o).ok())
            .collect();
        let opt_ptrs: Vec<*const i8> = c_opts.iter().map(|c| c.as_ptr()).collect();

        let rc = compile_f(prog, opt_ptrs.len() as i32, opt_ptrs.as_ptr());
        if rc != NVRTC_SUCCESS {
            let mut n = 0usize;
            let mut log = String::new();
            if log_size(prog, &mut n) == NVRTC_SUCCESS && n > 1 {
                let mut buf = vec![0i8; n];
                if get_log(prog, buf.as_mut_ptr()) == NVRTC_SUCCESS {
                    log = std::ffi::CStr::from_ptr(buf.as_ptr())
                        .to_string_lossy()
                        .into_owned();
                }
            }
            destroy(&mut prog);
            return Err(format!("NVRTC compilation error (code {rc}):\n{log}"));
        }

        let mut n = 0usize;
        if ptx_size(prog, &mut n) != NVRTC_SUCCESS {
            destroy(&mut prog);
            return Err("nvrtcGetPTXSize failed".into());
        }
        let mut buf = vec![0i8; n];
        if get_ptx(prog, buf.as_mut_ptr()) != NVRTC_SUCCESS {
            destroy(&mut prog);
            return Err("nvrtcGetPTX failed".into());
        }
        destroy(&mut prog);
        Ok(std::ffi::CStr::from_ptr(buf.as_ptr())
            .to_string_lossy()
            .into_owned())
    }
}
