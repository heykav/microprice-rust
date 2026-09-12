//! Configurable additive (Laplace-style) smoothing — see
//! `docs/model-spec.md`'s Smoothing section for why this is the only V1
//! smoothing method and what `alpha == 0.0` means.

use crate::error::CalibrationError;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SmoothingConfig {
    pub alpha: f64,
}

impl SmoothingConfig {
    /// No smoothing: a zero-observation state is reported as
    /// [`CalibrationError::InsufficientObservations`] rather than
    /// estimated.
    pub const NONE: SmoothingConfig = SmoothingConfig { alpha: 0.0 };

    pub fn new(alpha: f64) -> Result<Self, CalibrationError> {
        if !(alpha.is_finite() && alpha >= 0.0) {
            return Err(CalibrationError::InvalidSmoothingConfig { alpha });
        }
        Ok(SmoothingConfig { alpha })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_negative_alpha() {
        assert!(matches!(
            SmoothingConfig::new(-0.1),
            Err(CalibrationError::InvalidSmoothingConfig { .. })
        ));
    }

    #[test]
    fn rejects_non_finite_alpha() {
        assert!(SmoothingConfig::new(f64::NAN).is_err());
        assert!(SmoothingConfig::new(f64::INFINITY).is_err());
    }

    #[test]
    fn accepts_zero_and_positive_alpha() {
        assert!(SmoothingConfig::new(0.0).is_ok());
        assert!(SmoothingConfig::new(1.5).is_ok());
    }
}
