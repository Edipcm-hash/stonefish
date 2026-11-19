// src/profile.rs
// Lightweight profiling counters - to settle the "full GPU" debate with
// measurement instead of guesses. When disabled the cost is a single atomic
// load; `enable()` turns it on (used by the CLI's `test-gpt-profile` mode).
//
// Measured buckets:
//   matmul  : total time inside gpu::matmul_gpu (including transfer)
//   transfer: the CPU->GPU + GPU->CPU copies inside it
//   attn_fwd / attn_bwd : Op::CausalAttention forward/backward
// The rest = elementwise (LN, GELU, softmax, CE) + optimizer + packing.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

pub static ENABLED: AtomicBool = AtomicBool::new(false);
pub static MATMUL_CALLS: AtomicU64 = AtomicU64::new(0);
pub static MATMUL_NANOS: AtomicU64 = AtomicU64::new(0);
pub static TRANSFER_NANOS: AtomicU64 = AtomicU64::new(0);
pub static ATTN_FWD_NANOS: AtomicU64 = AtomicU64::new(0);
pub static ATTN_BWD_NANOS: AtomicU64 = AtomicU64::new(0);
pub static GELU_NANOS: AtomicU64 = AtomicU64::new(0);
pub static LN_NANOS: AtomicU64 = AtomicU64::new(0);
pub static CE_NANOS: AtomicU64 = AtomicU64::new(0);
pub static ADAM_NANOS: AtomicU64 = AtomicU64::new(0);
pub static SOFTMAX_NANOS: AtomicU64 = AtomicU64::new(0);
pub static ALLOC_NANOS: AtomicU64 = AtomicU64::new(0);
pub static ZEROGRAD_NANOS: AtomicU64 = AtomicU64::new(0);

#[inline]
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

pub fn enable() {
    ENABLED.store(true, Ordering::Relaxed);
}

pub fn disable() {
    ENABLED.store(false, Ordering::Relaxed);
}

pub fn reset() {
    MATMUL_CALLS.store(0, Ordering::Relaxed);
    MATMUL_NANOS.store(0, Ordering::Relaxed);
    TRANSFER_NANOS.store(0, Ordering::Relaxed);
    ATTN_FWD_NANOS.store(0, Ordering::Relaxed);
    ATTN_BWD_NANOS.store(0, Ordering::Relaxed);
    GELU_NANOS.store(0, Ordering::Relaxed);
    LN_NANOS.store(0, Ordering::Relaxed);
    CE_NANOS.store(0, Ordering::Relaxed);
    ADAM_NANOS.store(0, Ordering::Relaxed);
    SOFTMAX_NANOS.store(0, Ordering::Relaxed);
    ALLOC_NANOS.store(0, Ordering::Relaxed);
    ZEROGRAD_NANOS.store(0, Ordering::Relaxed);
}

#[inline]
pub fn add_matmul(nanos: u64, transfer_nanos: u64) {
    MATMUL_CALLS.fetch_add(1, Ordering::Relaxed);
    MATMUL_NANOS.fetch_add(nanos, Ordering::Relaxed);
    TRANSFER_NANOS.fetch_add(transfer_nanos, Ordering::Relaxed);
}

#[inline]
pub fn add_attn_fwd(nanos: u64) {
    ATTN_FWD_NANOS.fetch_add(nanos, Ordering::Relaxed);
}

#[inline]
pub fn add_attn_bwd(nanos: u64) {
    ATTN_BWD_NANOS.fetch_add(nanos, Ordering::Relaxed);
}

/// Instant values: (matmul calls, matmul ns, transfer ns, attn_fwd ns, attn_bwd ns)
pub fn snapshot() -> (u64, u64, u64, u64, u64) {
    (
        MATMUL_CALLS.load(Ordering::Relaxed),
        MATMUL_NANOS.load(Ordering::Relaxed),
        TRANSFER_NANOS.load(Ordering::Relaxed),
        ATTN_FWD_NANOS.load(Ordering::Relaxed),
        ATTN_BWD_NANOS.load(Ordering::Relaxed),
    )
}

/// Detailed snapshot (used by the profile mode).
#[derive(Debug, Default, Clone, Copy)]
pub struct Snapshot {
    pub matmul_calls: u64,
    pub matmul: u64,
    pub transfer: u64,
    pub attn_fwd: u64,
    pub attn_bwd: u64,
    pub gelu: u64,
    pub ln: u64,
    pub ce: u64,
    pub adam: u64,
    pub softmax: u64,
    pub alloc: u64,
    pub zero_grad: u64,
}

pub fn snapshot_full() -> Snapshot {
    Snapshot {
        matmul_calls: MATMUL_CALLS.load(Ordering::Relaxed),
        matmul: MATMUL_NANOS.load(Ordering::Relaxed),
        transfer: TRANSFER_NANOS.load(Ordering::Relaxed),
        attn_fwd: ATTN_FWD_NANOS.load(Ordering::Relaxed),
        attn_bwd: ATTN_BWD_NANOS.load(Ordering::Relaxed),
        gelu: GELU_NANOS.load(Ordering::Relaxed),
        ln: LN_NANOS.load(Ordering::Relaxed),
        ce: CE_NANOS.load(Ordering::Relaxed),
        adam: ADAM_NANOS.load(Ordering::Relaxed),
        softmax: SOFTMAX_NANOS.load(Ordering::Relaxed),
        alloc: ALLOC_NANOS.load(Ordering::Relaxed),
        zero_grad: ZEROGRAD_NANOS.load(Ordering::Relaxed),
    }
}

/// Returns `None` when profiling is off (the cost of a single atomic load).
#[inline]
pub fn start() -> Option<std::time::Instant> {
    if enabled() {
        Some(std::time::Instant::now())
    } else {
        None
    }
}

#[inline]
pub fn stop(t0: Option<std::time::Instant>, counter: &AtomicU64) {
    if let Some(t0) = t0 {
        counter.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}
