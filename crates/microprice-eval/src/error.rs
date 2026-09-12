//! Typed errors for `microprice-eval`.

use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum EvalError {
    /// [`crate::split::chronological_split`] was asked for an empty dataset.
    #[error("cannot split an empty event slice")]
    EmptyDataset,

    /// `train_fraction` was outside the open interval `(0.0, 1.0)`.
    #[error("train_fraction must be in (0.0, 1.0), got {train_fraction}")]
    InvalidTrainFraction { train_fraction: f64 },

    /// A split (or a horizon) would leave one side with zero events, which
    /// can never produce a usable train set or a single evaluable
    /// prediction — returned rather than silently evaluating on nothing.
    #[error("{reason}")]
    InsufficientEvents { reason: String },

    /// `horizon` was `0` — "compare against the event itself" isn't a
    /// meaningful out-of-sample evaluation.
    #[error("horizon must be >= 1, got {horizon}")]
    InvalidHorizon { horizon: usize },

    /// A book failed state encoding during evaluation (see
    /// `microprice_core::StateSpaceConfig::encode`'s error cases).
    #[error("state encoding failed during evaluation: {0}")]
    Core(#[from] microprice_core::MicroPriceError),

    /// Calibrating the training-split model failed.
    #[error("calibration failed: {0}")]
    Calibration(#[from] microprice_calibration::CalibrationError),
}
