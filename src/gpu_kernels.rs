// src/gpu_kernels.rs
// CUDA çekirdeklerinin Rust tarafı: NVRTC ile PTX derleme + modül/çekirdek
// önbelleği + tipli sarmalayıcılar.
//
// Neden NVRTC: bu makinede host C++ derleyicisi yok (nvcc `cl.exe` istiyor),
// NVRTC ise yalnız cihaz kodu derler ve `nvrtc64_*.dll` ile çalışır.
// PTX bir kez derlenir ve `target/gpu-cache/` altında saklanır; sonraki
// çalıştırmalar derleme maliyeti ödemez.

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Çekirdek kaynağı — derleme zamanında gömülür.
pub const SRC: &str = include_str!("../kernels/stonefish.cu");

static MODULE: OnceLock<Arc<CudaModule>> = OnceLock::new();
static FUNCS: OnceLock<Mutex<HashMap<String, CudaFunction>>> = OnceLock::new();

/// Tek bir çekirdek çağrısı: stream'i bağlar, argümanları sırayla geçirir.
macro_rules! launch {
    ($name:expr, $cfg:expr, [$($arg:expr),* $(,)?]) => {{
        let stream = crate::gpu::get_ctx().default_stream();
        let f = func($name);
        let mut b = stream.launch_builder(&f);
        // Tek ifade: geçici argümanlar launch() bitene kadar yaşasın.
        unsafe { b$(.arg($arg))*.launch($cfg) }
            .unwrap_or_else(|e| panic!("çekirdek hatası {}: {e}", $name))
    }};
}

fn funcs() -> &'static Mutex<HashMap<String, CudaFunction>> {
    FUNCS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn module() -> &'static Arc<CudaModule> {
    MODULE.get_or_init(|| {
        let ctx = crate::gpu::get_ctx();
        let ptx = compile_cached();
        ctx.load_module(ptx)
            .expect("GPU modülü yüklenemedi (PTX uyumsuz olabilir)")
    })
}

fn func(name: &str) -> CudaFunction {
    // Zehirli mutex'ten de devam et: tek bir çekirdek hatası tüm testleri kirletmesin.
    let mut map = funcs().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(f) = map.get(name) {
        return f.clone();
    }
    let f = module()
        .load_function(name)
        .unwrap_or_else(|e| panic!("çekirdek yüklenemedi: {name}: {e}"));
    map.insert(name.to_string(), f.clone());
    f
}

/// NVRTC ile derle; PTX'i diske önbellekle (anahtar: kaynak + arch + sürüm).
fn compile_cached() -> cudarc::nvrtc::Ptx {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let arch = pick_arch();
    let mut h = DefaultHasher::new();
    SRC.hash(&mut h);
    arch.hash(&mut h);
    crate::VERSION.hash(&mut h);
    let key = format!("{:016x}", h.finish());

    let dir = std::path::Path::new("target").join("gpu-cache");
    let path = dir.join(format!("kernels-{key}.ptx"));
    if let Ok(s) = std::fs::read_to_string(&path) {
        return cudarc::nvrtc::Ptx::from_src(s);
    }

    let opts = cudarc::nvrtc::CompileOptions {
        arch: Some(arch),
        use_fast_math: Some(true),
        ..Default::default()
    };
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(SRC, opts)
        .unwrap_or_else(|e| panic!("CUDA çekirdekleri derlenemedi (NVRTC, arch={arch}): {e}"));

    // Diske yaz (başarısız olursa sorun değil, sadece yavaşlar).
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::write(&path, ptx.to_src());
    ptx
}

/// Cihazın compute capability'sine göre NVRTC arch'i.
fn pick_arch() -> &'static str {
    use cudarc::driver::sys::CUdevice_attribute as Attr;
    let ctx = crate::gpu::get_ctx();
    let major = ctx
        .attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .unwrap_or(9);
    let minor = ctx
        .attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .unwrap_or(0);
    match (major, minor) {
        (12, _) => "compute_120",
        (10, _) => "compute_100",
        (9, _) => "compute_90",
        (8, 9) => "compute_89",
        (8, 6) => "compute_86",
        (8, 0) => "compute_80",
        _ => "compute_90",
    }
}

/// 1D grid: 256 thread/blok.
fn cfg_1d(n: usize) -> LaunchConfig {
    let threads = 256u32;
    let blocks = ((n as u64).div_ceil(threads as u64)).max(1) as u32;
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Satır başına bir blok.
fn cfg_rows(rows: usize, threads: u32, shared: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: shared,
    }
}

// ============================================================
// Elementwise
// ============================================================

/// out = a + bias (satır bazlı broadcast)
pub fn add_bias(
    out: &mut CudaSlice<f32>,
    a: &CudaSlice<f32>,
    bias: &CudaSlice<f32>,
    rows: usize,
    cols: usize,
) {
    let n = rows * cols;
    launch!(
        "add_scalar_broadcast",
        cfg_1d(n),
        [out, a, bias, &(rows as i32), &(cols as i32)]
    );
}

/// x += bias (satır bazlı broadcast, yerinde)
pub fn add_bias_inplace(x: &mut CudaSlice<f32>, bias: &CudaSlice<f32>, rows: usize, cols: usize) {
    let n = rows * cols;
    launch!(
        "add_bias_inplace",
        cfg_1d(n),
        [x, bias, &(rows as i32), &(cols as i32)]
    );
}

pub fn add_inplace(dst: &mut CudaSlice<f32>, src: &CudaSlice<f32>, n: usize) {
    launch!("add_inplace", cfg_1d(n), [dst, src, &(n as i32)]);
}

pub fn zero_fill(dst: &mut CudaSlice<f32>, n: usize) {
    launch!("zero_fill", cfg_1d(n), [dst, &(n as i32)]);
}

pub fn gelu_forward(out: &mut CudaSlice<f32>, x: &CudaSlice<f32>, n: usize) {
    launch!("gelu_forward", cfg_1d(n), [out, x, &(n as i32)]);
}

pub fn gelu_backward(dx: &mut CudaSlice<f32>, dy: &CudaSlice<f32>, x: &CudaSlice<f32>, n: usize) {
    launch!("gelu_backward", cfg_1d(n), [dx, dy, x, &(n as i32)]);
}

// ============================================================
// LayerNorm
// ============================================================

#[allow(clippy::too_many_arguments)]
pub fn layernorm_forward(
    out: &mut CudaSlice<f32>,
    xhat: &mut CudaSlice<f32>,
    inv: &mut CudaSlice<f32>,
    x: &CudaSlice<f32>,
    gamma: &CudaSlice<f32>,
    beta: &CudaSlice<f32>,
    rows: usize,
    cols: usize,
    eps: f32,
) {
    let threads = 128u32;
    let shared = 2 * threads * 4;
    launch!(
        "layernorm_forward",
        cfg_rows(rows, threads, shared),
        [
            out,
            xhat,
            inv,
            x,
            gamma,
            beta,
            &(rows as i32),
            &(cols as i32),
            &eps
        ]
    );
}

pub fn layernorm_backward(
    dx: &mut CudaSlice<f32>,
    dy: &CudaSlice<f32>,
    xhat: &CudaSlice<f32>,
    inv: &CudaSlice<f32>,
    gamma: &CudaSlice<f32>,
    rows: usize,
    cols: usize,
) {
    let threads = 128u32;
    let shared = 2 * threads * 4;
    launch!(
        "layernorm_backward",
        cfg_rows(rows, threads, shared),
        [
            dx,
            dy,
            xhat,
            inv,
            gamma,
            &(rows as i32),
            &(cols as i32)
        ]
    );
}

pub fn layernorm_param_grad(
    dgamma: &mut CudaSlice<f32>,
    dbeta: &mut CudaSlice<f32>,
    dy: &CudaSlice<f32>,
    xhat: &CudaSlice<f32>,
    rows: usize,
    cols: usize,
) {
    let cfg = LaunchConfig {
        grid_dim: (cols as u32, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    launch!(
        "layernorm_param_grad",
        cfg,
        [dgamma, dbeta, dy, xhat, &(rows as i32), &(cols as i32)]
    );
}

// ============================================================
// Attention softmax'ı
// ============================================================

pub fn masked_softmax_forward(
    w: &mut CudaSlice<f32>,
    scores: &CudaSlice<f32>,
    batch: usize,
    n_heads: usize,
    seq: usize,
    scale: f32,
) {
    let rows = batch * n_heads * seq;
    launch!(
        "masked_softmax_forward",
        cfg_rows(rows, 128, 0),
        [
            w,
            scores,
            &(seq as i32),
            &(n_heads as i32),
            &(batch as i32),
            &scale
        ]
    );
}

pub fn masked_softmax_backward(
    ds: &mut CudaSlice<f32>,
    w: &CudaSlice<f32>,
    dw: &CudaSlice<f32>,
    batch: usize,
    n_heads: usize,
    seq: usize,
    scale: f32,
) {
    let rows = batch * n_heads * seq;
    launch!(
        "masked_softmax_backward",
        cfg_rows(rows, 128, 0),
        [
            ds,
            w,
            dw,
            &(seq as i32),
            &(n_heads as i32),
            &(batch as i32),
            &scale
        ]
    );
}

// ============================================================
// Embedding
// ============================================================

pub fn embedding_forward(
    out: &mut CudaSlice<f32>,
    ids: &CudaSlice<i32>,
    tok_emb: &CudaSlice<f32>,
    pos_emb: &CudaSlice<f32>,
    rows: usize,
    cols: usize,
    seq_len: usize,
) {
    launch!(
        "embedding_forward",
        cfg_1d(rows * cols),
        [
            out,
            ids,
            tok_emb,
            pos_emb,
            &(rows as i32),
            &(cols as i32),
            &(seq_len as i32)
        ]
    );
}

pub fn embedding_backward(
    dtok: &mut CudaSlice<f32>,
    dpos: &mut CudaSlice<f32>,
    dout: &CudaSlice<f32>,
    ids: &CudaSlice<i32>,
    rows: usize,
    cols: usize,
    seq_len: usize,
) {
    launch!(
        "embedding_backward",
        cfg_1d(rows * cols),
        [
            dtok,
            dpos,
            dout,
            ids,
            &(rows as i32),
            &(cols as i32),
            &(seq_len as i32)
        ]
    );
}

// ============================================================
// Cross-entropy
// ============================================================

pub fn ce_forward(
    loss: &mut CudaSlice<f32>,
    probs: &mut CudaSlice<f32>,
    logits: &CudaSlice<f32>,
    targets: &CudaSlice<i32>,
    rows: usize,
    classes: usize,
) {
    launch!(
        "ce_forward",
        cfg_rows(rows, 128, 0),
        [loss, probs, logits, targets, &(rows as i32), &(classes as i32)]
    );
}

pub fn ce_backward(
    dlogits: &mut CudaSlice<f32>,
    probs: &CudaSlice<f32>,
    targets: &CudaSlice<i32>,
    rows: usize,
    classes: usize,
    upstream: f32,
) {
    launch!(
        "ce_backward",
        cfg_1d(rows * classes),
        [
            dlogits,
            probs,
            targets,
            &(rows as i32),
            &(classes as i32),
            &upstream
        ]
    );
}

// ============================================================
// Adam / gradyan kırpma
// ============================================================

#[allow(clippy::too_many_arguments)]
pub fn adam_step(
    param: &mut CudaSlice<f32>,
    grad: &CudaSlice<f32>,
    m: &mut CudaSlice<f32>,
    v: &mut CudaSlice<f32>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    bc1: f32,
    bc2: f32,
    n: usize,
) {
    launch!(
        "adam_step",
        cfg_1d(n),
        [
            param, grad, m, v, &lr, &beta1, &beta2, &eps, &bc1, &bc2, &(n as i32)
        ]
    );
}

pub fn sum_squares(acc: &mut CudaSlice<f32>, x: &CudaSlice<f32>, n: usize) {
    let threads = 256u32;
    let blocks = ((n as u64).div_ceil(threads as u64)).max(1) as u32;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    launch!("sum_squares_atomic", cfg, [acc, x, &(n as i32)]);
}

pub fn scale_inplace(x: &mut CudaSlice<f32>, n: usize, s: f32) {
    launch!("scale_inplace", cfg_1d(n), [x, &(n as i32), &s]);
}
