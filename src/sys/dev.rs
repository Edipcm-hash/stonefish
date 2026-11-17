// src/sys/dev.rs
// Device memory, module/kernel and launch helpers.
//
// Replaces `cudarc`: `CudaSlice` -> `DevBuf`, `CudaView` -> `DevView`,
// `CudaModule` → `Module`, `launch_builder` → `launch`.

#![allow(dead_code)]

use super::cuda;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::sync::OnceLock;

// ============================================================
// Context
// ============================================================

struct Ctx {
    ctx: cuda::CUcontext,
}

// The CUDA context handle is used on a single thread and lives for the
// whole process, so marking it Send/Sync is safe.
unsafe impl Send for Ctx {}
unsafe impl Sync for Ctx {}

static CTX: OnceLock<Ctx> = OnceLock::new();

/// Create the context (on first call) and bind it to this thread.
fn ctx() -> cuda::CUcontext {
    let c = CTX.get_or_init(|| unsafe {
        cuda::check(cuda::cuInit().expect("cuInit")(0), "cuInit");
        let mut dev: cuda::CUdevice = 0;
        cuda::check(
            cuda::cuDeviceGet().expect("cuDeviceGet")(&mut dev, 0),
            "cuDeviceGet",
        );
        let mut ctx: cuda::CUcontext = std::ptr::null_mut();
        cuda::check(
            cuda::cuCtxCreate_v2().expect("cuCtxCreate")(&mut ctx, 0, dev),
            "cuCtxCreate",
        );
        Ctx { ctx }
    });
    unsafe {
        if let Some(f) = cuda::cuCtxSetCurrent() {
            let _ = f(c.ctx);
        }
    }
    c.ctx
}

/// The device's compute capability (major, minor).
pub fn compute_capability() -> (i32, i32) {
    ctx();
    unsafe {
        let mut dev: cuda::CUdevice = 0;
        let _ = cuda::cuDeviceGet().map(|f| f(&mut dev, 0));
        let mut maj = 9;
        let mut min = 0;
        if let Some(f) = cuda::cuDeviceGetAttribute() {
            let _ = f(
                &mut maj,
                cuda::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                dev,
            );
            let _ = f(
                &mut min,
                cuda::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                dev,
            );
        }
        (maj, min)
    }
}

/// Device name and total memory.
pub fn device_info() -> (String, usize) {
    ctx();
    unsafe {
        let mut dev: cuda::CUdevice = 0;
        let _ = cuda::cuDeviceGet().map(|f| f(&mut dev, 0));
        let mut name = [0i8; 256];
        let mut nm = String::from("unknown");
        if let Some(f) = cuda::cuDeviceGetName() {
            if f(&mut name as *mut i8, 256, dev) == cuda::CUDA_SUCCESS {
                nm = std::ffi::CStr::from_ptr(name.as_ptr())
                    .to_string_lossy()
                    .into_owned();
            }
        }
        let mut total = 0usize;
        let _ = cuda::cuDeviceTotalMem_v2().map(|f| f(&mut total, dev));
        (nm, total)
    }
}

/// Free / total device memory (bytes).
pub fn mem_info() -> (usize, usize) {
    ctx();
    unsafe {
        let (mut free, mut total) = (0usize, 0usize);
        if let Some(f) = cuda::cuMemGetInfo_v2() {
            let _ = f(&mut free, &mut total);
        }
        (free, total)
    }
}

/// Complete all pending work.
pub fn sync() {
    ctx();
    unsafe {
        if let Some(f) = cuda::cuCtxSynchronize() {
            let _ = f();
        }
    }
}

// ============================================================
// Cihaz tamponu
// ============================================================

/// Owned buffer in device memory (f32).
pub struct DevBuf {
    ptr: cuda::CUdeviceptr,
    len: usize,
}

/// Owned buffer in device memory (i32) - for embedding indices.
pub struct DevBufI32 {
    ptr: cuda::CUdeviceptr,
    len: usize,
}

/// Common interface for device pointers (buffer, view, mutable view).
pub trait DevPtr {
    fn dptr(&self) -> cuda::CUdeviceptr;
    fn dlen(&self) -> usize;
}

impl DevBuf {
    /// Zero-initialized buffer.
    pub fn zeros(len: usize) -> Self {
        ctx();
        let len = len.max(1);
        unsafe {
            let mut ptr: cuda::CUdeviceptr = 0;
            cuda::check(
                cuda::cuMemAlloc_v2().expect("cuMemAlloc")(&mut ptr, len * 4),
                "cuMemAlloc",
            );
            cuda::check(
                cuda::cuMemsetD32_v2().expect("cuMemsetD32")(ptr, 0, len),
                "cuMemsetD32",
            );
            Self { ptr, len }
        }
    }

    /// Host verisinden kopyala.
    pub fn from_host(v: &[f32]) -> Self {
        ctx();
        let len = v.len().max(1);
        unsafe {
            let mut ptr: cuda::CUdeviceptr = 0;
            cuda::check(
                cuda::cuMemAlloc_v2().expect("cuMemAlloc")(&mut ptr, len * 4),
                "cuMemAlloc",
            );
            if !v.is_empty() {
                cuda::check(
                    cuda::cuMemcpyHtoD_v2().expect("HtoD")(
                        ptr,
                        v.as_ptr() as *const c_void,
                        v.len() * 4,
                    ),
                    "cuMemcpyHtoD",
                );
            }
            Self { ptr, len }
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Host'a kopyala (senkron).
    pub fn to_host(&self) -> Vec<f32> {
        ctx();
        let mut out = vec![0f32; self.len];
        unsafe {
            cuda::check(
                cuda::cuMemcpyDtoH_v2().expect("DtoH")(
                    out.as_mut_ptr() as *mut c_void,
                    self.ptr,
                    self.len * 4,
                ),
                "cuMemcpyDtoH",
            );
        }
        out
    }

    /// Copy from any device pointer (sizes must match).
    pub fn copy_from_ptr(&mut self, src: &impl DevPtr) {
        assert_eq!(self.len, src.dlen(), "copy_from_ptr size");
        ctx();
        unsafe {
            cuda::check(
                cuda::cuMemcpyDtoD_v2().expect("DtoD")(self.ptr, src.dptr(), self.len * 4),
                "cuMemcpyDtoD",
            );
        }
    }

    /// Copy from another device buffer (sizes must match).
    pub fn copy_from(&mut self, src: &DevBuf) {
        assert_eq!(self.len, src.len, "DevBuf::copy_from size");
        ctx();
        unsafe {
            cuda::check(
                cuda::cuMemcpyDtoD_v2().expect("DtoD")(self.ptr, src.ptr, self.len * 4),
                "cuMemcpyDtoD",
            );
        }
    }

    /// Slice view (no ownership).
    pub fn slice(&self, range: std::ops::RangeFrom<usize>) -> DevView<'_> {
        let start = range.start.min(self.len);
        DevView {
            ptr: self.ptr + (start * 4) as u64,
            len: self.len - start,
            _p: PhantomData,
        }
    }

    pub fn full(&self) -> DevView<'_> {
        DevView {
            ptr: self.ptr,
            len: self.len,
            _p: PhantomData,
        }
    }

    /// Mutable slice view.
    pub fn slice_mut(&mut self, range: std::ops::RangeFrom<usize>) -> DevViewMut<'_> {
        let start = range.start.min(self.len);
        DevViewMut {
            ptr: self.ptr + (start * 4) as u64,
            len: self.len - start,
            _p: PhantomData,
        }
    }
}

impl DevBufI32 {
    pub fn zeros(len: usize) -> Self {
        ctx();
        let len = len.max(1);
        unsafe {
            let mut ptr: cuda::CUdeviceptr = 0;
            cuda::check(
                cuda::cuMemAlloc_v2().expect("cuMemAlloc")(&mut ptr, len * 4),
                "cuMemAlloc",
            );
            cuda::check(
                cuda::cuMemsetD32_v2().expect("cuMemsetD32")(ptr, 0, len),
                "cuMemsetD32",
            );
            Self { ptr, len }
        }
    }

    pub fn from_host(v: &[i32]) -> Self {
        ctx();
        let len = v.len().max(1);
        unsafe {
            let mut ptr: cuda::CUdeviceptr = 0;
            cuda::check(
                cuda::cuMemAlloc_v2().expect("cuMemAlloc")(&mut ptr, len * 4),
                "cuMemAlloc",
            );
            if !v.is_empty() {
                cuda::check(
                    cuda::cuMemcpyHtoD_v2().expect("HtoD")(
                        ptr,
                        v.as_ptr() as *const c_void,
                        v.len() * 4,
                    ),
                    "cuMemcpyHtoD",
                );
            }
            Self { ptr, len }
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Write host data into an existing buffer (no new allocation).
    pub fn upload(&mut self, v: &[i32]) {
        assert!(v.len() <= self.len, "upload: buffer too small");
        if v.is_empty() {
            return;
        }
        ctx();
        unsafe {
            cuda::check(
                cuda::cuMemcpyHtoD_v2().expect("HtoD")(
                    self.ptr,
                    v.as_ptr() as *const c_void,
                    v.len() * 4,
                ),
                "cuMemcpyHtoD",
            );
        }
    }

    pub fn to_host(&self) -> Vec<i32> {
        ctx();
        let mut out = vec![0i32; self.len];
        unsafe {
            cuda::check(
                cuda::cuMemcpyDtoH_v2().expect("DtoH")(
                    out.as_mut_ptr() as *mut c_void,
                    self.ptr,
                    self.len * 4,
                ),
                "cuMemcpyDtoH",
            );
        }
        out
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        unsafe {
            let _ = cuda::cuMemFree_v2().map(|f| f(self.ptr));
        }
    }
}

impl Drop for DevBufI32 {
    fn drop(&mut self) {
        unsafe {
            let _ = cuda::cuMemFree_v2().map(|f| f(self.ptr));
        }
    }
}

impl DevPtr for DevBuf {
    fn dptr(&self) -> cuda::CUdeviceptr {
        self.ptr
    }
    fn dlen(&self) -> usize {
        self.len
    }
}

impl DevPtr for DevBufI32 {
    fn dptr(&self) -> cuda::CUdeviceptr {
        self.ptr
    }
    fn dlen(&self) -> usize {
        self.len
    }
}

pub struct DevView<'a> {
    ptr: cuda::CUdeviceptr,
    len: usize,
    _p: PhantomData<&'a ()>,
}

pub struct DevViewMut<'a> {
    ptr: cuda::CUdeviceptr,
    len: usize,
    _p: PhantomData<&'a mut ()>,
}

impl DevPtr for DevView<'_> {
    fn dptr(&self) -> cuda::CUdeviceptr {
        self.ptr
    }
    fn dlen(&self) -> usize {
        self.len
    }
}

impl DevPtr for DevViewMut<'_> {
    fn dptr(&self) -> cuda::CUdeviceptr {
        self.ptr
    }
    fn dlen(&self) -> usize {
        self.len
    }
}

impl DevPtr for &DevView<'_> {
    fn dptr(&self) -> cuda::CUdeviceptr {
        self.ptr
    }
    fn dlen(&self) -> usize {
        self.len
    }
}

impl DevPtr for &DevViewMut<'_> {
    fn dptr(&self) -> cuda::CUdeviceptr {
        self.ptr
    }
    fn dlen(&self) -> usize {
        self.len
    }
}

// ============================================================
// Module and kernel
// ============================================================

pub struct Module {
    m: cuda::CUmodule,
}

// The module handle is used on a single thread for the whole process.
unsafe impl Send for Module {}
unsafe impl Sync for Module {}

/// A loaded kernel (copyable handle).
#[derive(Clone, Copy)]
pub struct Function {
    f: cuda::CUfunction,
}

impl Module {
    /// Load a module from PTX text.
    pub fn from_ptx(ptx: &str) -> Result<Self, String> {
        ctx();
        let c = std::ffi::CString::new(ptx).map_err(|e| e.to_string())?;
        unsafe {
            let mut m: cuda::CUmodule = std::ptr::null_mut();
            let f = cuda::cuModuleLoadData().ok_or("cuModuleLoadData unavailable")?;
            let rc = f(&mut m, c.as_ptr() as *const c_void);
            if rc != cuda::CUDA_SUCCESS {
                return Err(format!("cuModuleLoadData failed (code {rc})"));
            }
            Ok(Self { m })
        }
    }

    pub fn func(&self, name: &str) -> Result<Function, String> {
        let c = std::ffi::CString::new(name).map_err(|e| e.to_string())?;
        unsafe {
            let mut f: cuda::CUfunction = std::ptr::null_mut();
            let get = cuda::cuModuleGetFunction().ok_or("cuModuleGetFunction unavailable")?;
            let rc = get(&mut f, self.m, c.as_ptr());
            if rc != cuda::CUDA_SUCCESS {
                return Err(format!("kernel not found: {name} (code {rc})"));
            }
            Ok(Function { f })
        }
    }
}

/// Prepare kernel arguments.
///
/// CUDA takes arguments as "a pointer to each value".
/// A single u64 slot fits both a device pointer (8 bytes) and an i32/f32
/// (the low 4 bytes, little-endian).
pub struct Kargs {
    vals: [u64; 24],
    ptrs: [*mut c_void; 24],
    n: usize,
}

impl Default for Kargs {
    fn default() -> Self {
        Self::new()
    }
}

impl Kargs {
    pub fn new() -> Self {
        Self {
            vals: [0; 24],
            ptrs: [std::ptr::null_mut(); 24],
            n: 0,
        }
    }

    fn push(&mut self, v: u64) -> &mut Self {
        assert!(self.n < 24, "too many kernel arguments");
        self.vals[self.n] = v;
        self.n += 1;
        self
    }

    /// Device buffer / view argument.
    pub fn buf<P: DevPtr>(&mut self, p: &P) -> &mut Self {
        self.push(p.dptr())
    }

    /// i32 argument (4 bytes are read).
    pub fn i32(&mut self, v: i32) -> &mut Self {
        self.push(v as i64 as u64)
    }

    /// f32 argument (4 bytes are read).
    pub fn f32(&mut self, v: f32) -> &mut Self {
        self.push(v.to_bits() as u64)
    }

    fn as_ptrs(&mut self) -> *mut *mut c_void {
        for i in 0..self.n {
            self.ptrs[i] = &mut self.vals[i] as *mut u64 as *mut c_void;
        }
        self.ptrs.as_mut_ptr()
    }
}

/// Launch the kernel (default stream).
pub fn launch(
    f: Function,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared_mem: u32,
    args: &mut Kargs,
) {
    ctx();
    unsafe {
        let l = cuda::cuLaunchKernel().expect("cuLaunchKernel");
        let rc = l(
            f.f,
            grid.0,
            grid.1,
            grid.2,
            block.0,
            block.1,
            block.2,
            shared_mem,
            std::ptr::null_mut(),
            args.as_ptrs(),
            std::ptr::null_mut(),
        );
        if rc != cuda::CUDA_SUCCESS {
            panic!("cuLaunchKernel failed (code {rc})");
        }
    }
}
