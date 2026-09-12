//! Typed errors for `microprice-data`.

use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum DataError {
    /// A [`crate::synthetic::SyntheticConfig`] had an invalid rate (outside
    /// `[0.0, 1.0]`), rates summing above `1.0`, or a sub-1-tick initial
    /// spread. Raised at construction, never mid-generation.
    #[error("invalid synthetic generator configuration: {reason}")]
    InvalidSyntheticConfig { reason: String },
}
