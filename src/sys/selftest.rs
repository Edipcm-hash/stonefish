// src/sys/selftest.rs
// Kendi CUDA FFI'mizin doğrulaması: bağlam + bellek + NVRTC derleme +
// çekirdek başlatma + cuBLAS SGEMM. Hepsi `cudarc` olmadan.
//
// CPU referansına karşı sayısal karşılaştırma yapar.

use super::{cublas, cuda, nvrtc};
use std::ffi::c_void;

const ADD_KERNEL: &str = r#"
extern "C" __global__ void add_kernel(float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
"#;

/// Tüm FFI yolunu test eder. Döner: başarılıysa true.
pub fn run() -> bool {
    println!("🐟 CUDA FFI öz-testi (cudarc YOK — ham sürücü API'si)\n");

    // ---- 1) Sürücü + cihaz ----
    let info = match cuda::device_info() {
        Some(v) => v,
        None => {
            println!("❌ CUDA sürücüsü yüklenemedi.");
            return false;
        }
    };
    let (name, maj, min, total_mem) = info;
    println!("Cihaz      : {name} (sm_{maj}{min})");
    println!("Bellek     : {:.1} GB", total_mem as f64 / (1 << 30) as f64);

    unsafe {
        // ---- 2) Bağlam ----
        let mut ctx: cuda::CUcontext = std::ptr::null_mut();
        let cu_ctx_create = match cuda::cuCtxCreate_v2() {
            Some(f) => f,
            None => {
                println!("❌ cuCtxCreate bulunamadı");
                return false;
            }
        };
        cuda::check(cu_ctx_create(&mut ctx, 0, 0), "cuCtxCreate");
        println!("Bağlam     : oluşturuldu");

        // ---- 3) Bellek + kopyalama ----
        let n = 1024usize;
        let a_host: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let b_host: Vec<f32> = (0..n).map(|i| 1.0 - i as f32 * 0.25).collect();
        let bytes = n * 4;

        let mut da: cuda::CUdeviceptr = 0;
        let mut db: cuda::CUdeviceptr = 0;
        let mut dc: cuda::CUdeviceptr = 0;
        cuda::check(cuda::cuMemAlloc_v2().unwrap()(&mut da, bytes), "alloc a");
        cuda::check(cuda::cuMemAlloc_v2().unwrap()(&mut db, bytes), "alloc b");
        cuda::check(cuda::cuMemAlloc_v2().unwrap()(&mut dc, bytes), "alloc c");
        cuda::check(
            cuda::cuMemcpyHtoD_v2().unwrap()(da, a_host.as_ptr() as *const c_void, bytes),
            "HtoD a",
        );
        cuda::check(
            cuda::cuMemcpyHtoD_v2().unwrap()(db, b_host.as_ptr() as *const c_void, bytes),
            "HtoD b",
        );
        println!("Bellek     : 3 × {} KB ayrıldı, kopyalandı", bytes / 1024);

        // ---- 4) NVRTC ile çekirdek derle → modül yükle ----
        let t0 = std::time::Instant::now();
        let ptx = match nvrtc::compile(ADD_KERNEL, &["--gpu-architecture=compute_120", "--fmad=true"]) {
            Ok(p) => p,
            Err(e) => {
                println!("❌ NVRTC: {e}");
                return false;
            }
        };
        let compile_ms = t0.elapsed().as_millis();
        println!("NVRTC      : PTX üretildi ({} KB, {compile_ms} ms)", ptx.len() / 1024);

        let c_ptx = std::ffi::CString::new(ptx).unwrap();
        let mut module: cuda::CUmodule = std::ptr::null_mut();
        cuda::check(
            cuda::cuModuleLoadData().unwrap()(&mut module, c_ptx.as_ptr() as *const c_void),
            "cuModuleLoadData",
        );
        let kname = std::ffi::CString::new("add_kernel").unwrap();
        let mut func: cuda::CUfunction = std::ptr::null_mut();
        cuda::check(
            cuda::cuModuleGetFunction().unwrap()(&mut func, module, kname.as_ptr()),
            "cuModuleGetFunction",
        );
        println!("Modül      : yüklendi, çekirdek bulundu");

        // ---- 5) Çekirdek başlat ----
        let threads = 256u32;
        let blocks = ((n as u64).div_ceil(threads as u64)) as u32;
        let ni = n as i32;
        let mut args: [*mut c_void; 4] = [
            &dc as *const _ as *mut c_void,
            &da as *const _ as *mut c_void,
            &db as *const _ as *mut c_void,
            &ni as *const _ as *mut c_void,
        ];
        cuda::check(
            cuda::cuLaunchKernel().unwrap()(
                func,
                blocks,
                1,
                1,
                threads,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            ),
            "cuLaunchKernel",
        );
        cuda::check(cuda::cuCtxSynchronize().unwrap()(), "sync");

        let mut out_host = vec![0f32; n];
        cuda::check(
            cuda::cuMemcpyDtoH_v2().unwrap()(out_host.as_mut_ptr() as *mut c_void, dc, bytes),
            "DtoH",
        );
        let mut max_diff = 0.0f32;
        for i in 0..n {
            max_diff = max_diff.max((out_host[i] - (a_host[i] + b_host[i])).abs());
        }
        if max_diff > 1e-6 {
            println!("❌ Çekirdek yanlış: max fark {max_diff}");
            return false;
        }
        println!("Çekirdek   : ✅ doğru (max fark {max_diff:.1e})");

        // ---- 6) cuBLAS SGEMM ----
        // Row-major C(2,3) = A(2,4)·B(4,3); cuBLAS kolon-major olduğu için
        // operandlar yer değiştirir: gemm(m=3, n=2, k=4, lda=3, ldb=4, ldc=3)
        let (m, k, nn) = (2usize, 4usize, 3usize);
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2x4
        let b = [1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0]; // 4x3
        let mut c_cpu = vec![0f32; m * nn];
        for i in 0..m {
            for j in 0..nn {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += a[i * k + kk] * b[kk * nn + j];
                }
                c_cpu[i * nn + j] = acc;
            }
        }

        let mut ga: cuda::CUdeviceptr = 0;
        let mut gb: cuda::CUdeviceptr = 0;
        let mut gc: cuda::CUdeviceptr = 0;
        cuda::check(cuda::cuMemAlloc_v2().unwrap()(&mut ga, m * k * 4), "alloc ga");
        cuda::check(cuda::cuMemAlloc_v2().unwrap()(&mut gb, k * nn * 4), "alloc gb");
        cuda::check(cuda::cuMemAlloc_v2().unwrap()(&mut gc, m * nn * 4), "alloc gc");
        cuda::check(
            cuda::cuMemcpyHtoD_v2().unwrap()(ga, a.as_ptr() as *const c_void, m * k * 4),
            "HtoD ga",
        );
        cuda::check(
            cuda::cuMemcpyHtoD_v2().unwrap()(gb, b.as_ptr() as *const c_void, k * nn * 4),
            "HtoD gb",
        );

        let mut handle: cublas::cublasHandle_t = std::ptr::null_mut();
        cublas::check(cublas::cublasCreate_v2().unwrap()(&mut handle), "cublasCreate");
        let alpha = 1.0f32;
        let beta = 0.0f32;
        cublas::check(
            cublas::cublasSgemm_v2().unwrap()(
                handle,
                cublas::CUBLAS_OP_N,
                cublas::CUBLAS_OP_N,
                nn as i32,   // m
                m as i32,    // n
                k as i32,    // k
                &alpha,
                gb as *const f32,
                nn as i32,   // lda = B'nin satır adımı
                ga as *const f32,
                k as i32,    // ldb = A'nın satır adımı
                &beta,
                gc as *mut f32,
                nn as i32,   // ldc
            ),
            "cublasSgemm",
        );
        cuda::check(cuda::cuCtxSynchronize().unwrap()(), "sync gemm");
        let mut c_gpu = vec![0f32; m * nn];
        cuda::check(
            cuda::cuMemcpyDtoH_v2().unwrap()(c_gpu.as_mut_ptr() as *mut c_void, gc, m * nn * 4),
            "DtoH c",
        );
        let mut gdiff = 0.0f32;
        for i in 0..m * nn {
            gdiff = gdiff.max((c_gpu[i] - c_cpu[i]).abs());
        }
        if gdiff > 1e-5 {
            println!("❌ cuBLAS yanlış: max fark {gdiff}\n  gpu={c_gpu:?}\n  cpu={c_cpu:?}");
            return false;
        }
        println!("cuBLAS     : ✅ SGEMM doğru (max fark {gdiff:.1e})");

        // ---- temizlik ----
        cublas::cublasDestroy_v2().unwrap()(handle);
        for p in [da, db, dc, ga, gb, gc] {
            cuda::cuMemFree_v2().unwrap()(p);
        }
        cuda::cuCtxDestroy_v2().unwrap()(ctx);
    }

    println!("\n🎉 FFI öz-testi geçti — cudarc olmadan CUDA tam çalışıyor.");
    true
}
