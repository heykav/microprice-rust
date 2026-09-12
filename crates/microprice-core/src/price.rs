//! Integer-tick price representation.
//!
//! See `docs/model-spec.md` for the rationale: prices are never stored as
//! `f32`/`f64` internally. A price of `$187.32` at `tick_size = $0.01` is
//! `PriceTicks(18732)` — exact, deterministic, and comparable/orderable
//! without a floating-point tolerance.

use std::fmt;

/// A price expressed as a signed count of ticks from zero.
///
/// Signed (not `u64`) because intermediate arithmetic — a crossed book's
/// spread, for instance — can be legitimately negative, and the type should
/// represent that rather than panicking or wrapping on subtraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PriceTicks(pub i64);

impl PriceTicks {
    pub const ZERO: PriceTicks = PriceTicks(0);

    /// Converts to a floating-point price given a tick size, for display
    /// and interop only. Never use the result of this for comparisons or
    /// further arithmetic inside the library — that defeats the entire
    /// point of the integer representation.
    pub fn to_f64(self, tick_size: f64) -> f64 {
        self.0 as f64 * tick_size
    }

    /// Checked addition; `None` on `i64` overflow rather than a panic or a
    /// silent wraparound.
    pub fn checked_add(self, rhs: PriceTicks) -> Option<PriceTicks> {
        self.0.checked_add(rhs.0).map(PriceTicks)
    }

    /// Checked subtraction; `None` on `i64` overflow.
    pub fn checked_sub(self, rhs: PriceTicks) -> Option<PriceTicks> {
        self.0.checked_sub(rhs.0).map(PriceTicks)
    }
}

impl fmt::Display for PriceTicks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}ticks", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_f64_applies_tick_size() {
        assert_eq!(PriceTicks(18732).to_f64(0.01), 187.32);
    }

    #[test]
    fn checked_add_detects_overflow() {
        assert_eq!(PriceTicks(i64::MAX).checked_add(PriceTicks(1)), None);
        assert_eq!(
            PriceTicks(1).checked_add(PriceTicks(2)),
            Some(PriceTicks(3))
        );
    }

    #[test]
    fn checked_sub_detects_overflow() {
        assert_eq!(PriceTicks(i64::MIN).checked_sub(PriceTicks(1)), None);
        assert_eq!(
            PriceTicks(5).checked_sub(PriceTicks(2)),
            Some(PriceTicks(3))
        );
    }

    #[test]
    fn ordering_is_exact_integer_ordering() {
        assert!(PriceTicks(100) < PriceTicks(101));
        assert_eq!(PriceTicks(100), PriceTicks(100));
    }
}
