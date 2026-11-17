// src/sys/dev.rs
// Cihaz belleği, modül/çekirdek ve başlatma yardımcıları.
//
// `cudarc`'ın yerini alır: `CudaSlice` → `DevBuf`, `CudaView` → `DevView`,
// `CudaModule` → `Module`, `launch_builder` → `launch`.

#![allow(dead_code)]

use super::cuda;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::sync::OnceLock;

// ============================================================
// Bağlam
// ============================================================

struct Ctx {
    ctx: cuda::CUcontext,
}

// CUDA bağlamı tutamacı tek iş parçacığında kullanılır; süreç ömrü boyunca
// yaşadığı için Send/Sync işaretlemek güvenli.
unsafe impl Send for Ctx {}
unsafe impl Sync for Ctx {}

static CTX: OnceLock<Ctx> = OnceLock::new();

/// Bağlamı (ilk çağrıda) oluştur ve bu iş parçacığına bağla.
fn ctx() -> cuda::CUcontext {
    let c = CTX.get_or_init(|| unsafe {
        cuda::check(cuda::cuInit().expect("cuInit") (0), "cuInit");
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

/// Cihazın hesaplama yeteneği (major, minor).
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

/// Cihaz adı ve toplam belleği.
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

/// Boşta olan / toplam cihaz belleği (bayt).
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

/// Tüm bekleyen işleri tamamla.
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

/// Cihaz belleğinde sahipli tampon (f32).
pub struct DevBuf {
    ptr: cuda::CUdeviceptr,
    len: usize,
}

/// Cihaz belleğinde sahipli tampon (i32) — embedding indeksleri için.
pub struct DevBufI32 {
    ptr: cuda::CUdeviceptr,
    len: usize,
}

/// Cihaz belleğine işaret eden ortak arayüz (tampon, görünüm, değiştirilebilir görünüm).
pub trait DevPtr {
    fn dptr(&self) -> cuda::CUdeviceptr;
    fn dlen(&self) -> usize;
}

impl DevBuf {
    /// Sıfırlanmış tampon.
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

    /// Herhangi bir cihaz işaretçisinden kopyala (boyutlar eşit olmalı).
    pub fn copy_from_ptr(&mut self, src: &impl DevPtr) {
        assert_eq!(self.len, src.dlen(), "copy_from_ptr boyut");
        ctx();
        unsafe {
            cuda::check(
                cuda::cuMemcpyDtoD_v2().expect("DtoD")(self.ptr, src.dptr(), self.len * 4),
                "cuMemcpyDtoD",
            );
        }
    }

    /// Başka bir cihaz tamponundan kopyala (boyutlar eşit olmalı).
    pub fn copy_from(&mut self, src: &DevBuf) {
        assert_eq!(self.len, src.len, "DevBuf::copy_from boyut");
        ctx();
        unsafe {
            cuda::check(
                cuda::cuMemcpyDtoD_v2().expect("DtoD")(self.ptr, src.ptr, self.len * 4),
                "cuMemcpyDtoD",
            );
        }
    }

    pub fn fill_zero(&mut self) {
        ctx();
        unsafe {
            cuda::check(
                cuda::cuMemsetD32_v2().expect("memset")(self.ptr, 0, self.len),
                "cuMemsetD32",
            );
        }
    }

    /// Dilim görünümü (sahiplik yok).
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

    /// Değiştirilebilir dilim görünümü.
    pub fn slice_mut(&mut self, range: std::ops::RangeFrom<usize>) -> DevViewMut<'_> {
        let start = range.start.min(self.len);
        DevViewMut {
            ptr: self.ptr + (start * 4) as u64,
            len: self.len - start,
            _p: PhantomData,
        }
    }

    pub fn full_mut(&mut self) -> DevViewMut<'_> {
        DevViewMut {
            ptr: self.ptr,
            len: self.len,
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

    /// Mevcut tampona host verisini yaz (yeni tahsis yok).
    pub fn upload(&mut self, v: &[i32]) {
        assert!(v.len() <= self.len, "upload: tampon çok küçük");
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
// Modül ve çekirdek
// ============================================================

pub struct Module {
    m: cuda::CUmodule,
}

// Modül tutamacı süreç ömrü boyunca tek iş parçacığında kullanılır.
unsafe impl Send for Module {}
unsafe impl Sync for Module {}

/// Yüklenmiş çekirdek (kopyalanabilir tutamaç).
#[derive(Clone, Copy)]
pub struct Function {
    f: cuda::CUfunction,
}

impl Module {
    /// PTX metninden modül yükle.
    pub fn from_ptx(ptx: &str) -> Result<Self, String> {
        ctx();
        let c = std::ffi::CString::new(ptx).map_err(|e| e.to_string())?;
        unsafe {
            let mut m: cuda::CUmodule = std::ptr::null_mut();
            let f = cuda::cuModuleLoadData().ok_or("cuModuleLoadData yok")?;
            let rc = f(&mut m, c.as_ptr() as *const c_void);
            if rc != cuda::CUDA_SUCCESS {
                return Err(format!("cuModuleLoadData başarısız (kod {rc})"));
            }
            Ok(Self { m })
        }
    }

    pub fn func(&self, name: &str) -> Result<Function, String> {
        let c = std::ffi::CString::new(name).map_err(|e| e.to_string())?;
        unsafe {
            let mut f: cuda::CUfunction = std::ptr::null_mut();
            let get = cuda::cuModuleGetFunction().ok_or("cuModuleGetFunction yok")?;
            let rc = get(&mut f, self.m, c.as_ptr());
            if rc != cuda::CUDA_SUCCESS {
                return Err(format!("çekirdek bulunamadı: {name} (kod {rc})"));
            }
            Ok(Function { f })
        }
    }
}

/// Çekirdek argümanlarını hazırla.
///
/// CUDA, argümanları "her biri bir işaretçinin gösterdiği değer" olarak alır.
/// Tek bir u64 slotu hem cihaz işaretçisi (8 bayt) hem i32/f32 (ilk 4 bayt,
/// little-endian) için yeterli.
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
        assert!(self.n < 24, "çok fazla çekirdek argümanı");
        self.vals[self.n] = v;
        self.n += 1;
        self
    }

    /// Cihaz tamponu / görünümü argümanı.
    pub fn buf<P: DevPtr>(&mut self, p: &P) -> &mut Self {
        self.push(p.dptr())
    }

    /// i32 argüman (4 bayt okunur).
    pub fn i32(&mut self, v: i32) -> &mut Self {
        self.push(v as i64 as u64)
    }

    /// f32 argüman (4 bayt okunur).
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

/// Çekirdeği başlat (varsayılan akış).
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
            panic!("cuLaunchKernel başarısız (kod {rc})");
        }
    }
}
