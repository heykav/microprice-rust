//! Typed errors for `microprice-calibration`.

use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum CalibrationError {
    /// A `BookEvent`'s `sequence` did not strictly increase relative to
    /// the previous event fed to the same [`crate::transitions::TransitionCounter`].
    #[error("out-of-order event: sequence {sequence} did not increase past the previous {previous_sequence}")]
    OutOfOrderEvent {
        previous_sequence: u64,
        sequence: u64,
    },

    /// Two [`crate::transitions::TransitionCounter`]s being merged were
    /// built against state spaces of different sizes.
    #[error("cannot merge transition counters built for different state counts ({a} vs {b})")]
    StateCountMismatch { a: u32, b: u32 },

    /// A [`crate::smoothing::SmoothingConfig`] had a negative `alpha`.
    #[error("smoothing alpha must be >= 0.0, got {alpha}")]
    InvalidSmoothingConfig { alpha: f64 },

    /// A state had zero observations and `alpha == 0.0`, so no probability
    /// or `G1` value can be estimated for it without inventing one — see
    /// `docs/model-spec.md`'s Smoothing section for why this is a returned
    /// error rather than a silent `0.0`.
    #[error("state {state} has zero observations and smoothing alpha is 0.0 - cannot estimate")]
    InsufficientObservations { state: u32 },

    /// The solver's fixed-point iteration did not converge to
    /// `tolerance` within `max_iterations` — returned rather than
    /// silently handing back a partially-converged result.
    #[error("solver did not converge within {max_iterations} iterations (final delta {final_delta}, tolerance {tolerance})")]
    DidNotConverge {
        max_iterations: usize,
        final_delta: f64,
        tolerance: f64,
    },

    /// A `Q`/`G1` dimension didn't match the declared state count.
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// A model artifact failed validation on load (see
    /// `crate::model::MicroPriceModel::load`).
    #[error("invalid model artifact: {reason}")]
    InvalidModelArtifact { reason: String },

    /// (De)serialization failed.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// I/O failure while loading/saving a model artifact.
    #[error("I/O error: {0}")]
    Io(String),

    /// A book failed state encoding at predict time (an out-of-domain
    /// input — see `microprice_core::StateSpaceConfig::encode`), wrapped
    /// so callers of `MicroPriceModel::predict` only need to match one
    /// error type.
    #[error("state encoding failed: {0}")]
    Core(#[from] microprice_core::MicroPriceError),
}
