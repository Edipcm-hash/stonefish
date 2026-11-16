// src/lib.rs
// StonefishAI — kütüphane kökü.
//
// Tüm motor modülleri burada yaşar; `src/main.rs` sadece CLI kabuğudur.
// Bu ayrım sayesinde op'lar `tests/` altından (gradcheck, birim testleri) doğrudan
// kullanılabilir.

pub mod autograd;
pub mod checkpoint;
pub mod data;
pub mod gpu;
pub mod loss;
pub mod nn;
pub mod optim;
pub mod tensor;
pub mod tokenizer;
pub mod transformer;

/// Kütüphane sürümü (Cargo.toml ile tek kaynaktan).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
