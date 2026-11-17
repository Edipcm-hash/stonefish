// src/lib.rs
// StonefishAI — kütüphane kökü.
//
// Tüm motor modülleri burada yaşar; `src/main.rs` sadece CLI kabuğudur.
// Bu ayrım sayesinde op'lar `tests/` altından (gradcheck, birim testleri) doğrudan
// kullanılabilir.

// `needless_range_loop`: bu motorda iç döngüler aynı anda birden fazla diziyi
// (q/k/v, skorlar, x_hat, dx_hat…) aynı indeksle tarıyor ve sınırlar döngü
// değişkeninden türüyor. Iterator zincirine çevirmek hot path'lerde hem
// okunabilirliği hem de derleyicinin vektörleştirme şansını düşürüyor.
#![allow(clippy::needless_range_loop)]
// `manual_is_multiple_of`: `is_multiple_of` yeni bir stabilizasyon; MSRV'yi
// yükseltmemek için `% n == 0` tercih ediliyor.
#![allow(clippy::manual_is_multiple_of)]

pub mod autograd;
pub mod bpe;
pub mod checkpoint;
pub mod corpus;
pub mod data;
pub mod gpu;
#[cfg(feature = "cuda")]
pub mod gpu_kernels;
#[cfg(feature = "cuda")]
pub mod gpu_train;
#[cfg(feature = "cuda")]
pub mod en_train;
pub mod loss;
pub mod nn;
pub mod optim;
pub mod profile;
pub mod tensor;
pub mod tokenizer;
pub mod transformer;

/// Kütüphane sürümü (Cargo.toml ile tek kaynaktan).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
