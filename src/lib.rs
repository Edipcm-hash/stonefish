// src/lib.rs
// StonefishAI - library root.
//
// All engine modules live here; `src/main.rs` is only the CLI shell.
// Thanks to this split the ops are directly usable from `tests/` (gradcheck,
// unit tests).

// `needless_range_loop`: in this engine the inner loops index several arrays
// at once (q/k/v, scores, x_hat, dx_hat...) with the same index and the bounds
// derive from the loop variable. Rewriting them as iterator chains would hurt
// both readability and the compiler's chance to vectorize on hot paths.
#![allow(clippy::needless_range_loop)]
// `manual_is_multiple_of`: `is_multiple_of` is a recent stabilization; we keep
// `% n == 0` to avoid raising the MSRV.
#![allow(clippy::manual_is_multiple_of)]

pub mod autograd;
pub mod bpe;
pub mod checkpoint;
pub mod corpus;
pub mod data;
#[cfg(feature = "cuda")]
pub mod demo;
#[cfg(feature = "cuda")]
pub mod en_train;
pub mod gpu;
#[cfg(feature = "cuda")]
pub mod gpu_kernels;
#[cfg(feature = "cuda")]
pub mod gpu_train;
#[cfg(feature = "cuda")]
pub mod kv_cache;
pub mod loss;
pub mod nn;
pub mod optim;
pub mod profile;
#[cfg(feature = "cuda")]
pub mod sys;
pub mod tensor;
pub mod tokenizer;
pub mod transformer;

/// Library version (single source of truth with Cargo.toml).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
