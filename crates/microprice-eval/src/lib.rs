//! Chronological out-of-sample evaluation for MicroPrice-Rust (Phase 11):
//! [`split::chronological_split`] (train/test split that respects time
//! ordering), [`metrics`] (MAE, signed bias, directional accuracy — see
//! that module's docs for why a Brier score is a disclosed, not-yet-solved
//! gap rather than a fabricated number), and [`evaluate::evaluate`] (runs a
//! calibrated model against a held-out event stream and compares it to the
//! `mid`/`weighted_mid` baselines).

#![forbid(unsafe_code)]

pub mod error;
pub mod evaluate;
pub mod metrics;
pub mod split;

pub use error::EvalError;
pub use evaluate::{evaluate as evaluate_model, EvalReport};
pub use split::chronological_split;
