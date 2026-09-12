//! Resting order-book quantity.

use std::fmt;

/// A non-negative resting quantity (shares, contracts, lots — unit is
/// whatever the data source uses). `u64` rules out negative quantities at
/// the type level rather than validating it at runtime: there is no such
/// thing as a negative resting size, so the invalid state simply isn't
/// representable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Quantity(pub u64);

impl Quantity {
    pub const ZERO: Quantity = Quantity(0);

    pub fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// Checked addition; `None` on `u64` overflow. Only reachable with
    /// quantities near `u64::MAX`, but checked rather than silently
    /// wrapping — see `MicroPriceError::QuantityOverflow`.
    pub fn checked_add(self, rhs: Quantity) -> Option<Quantity> {
        self.0.checked_add(rhs.0).map(Quantity)
    }
}

impl fmt::Display for Quantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_zero_detects_zero_and_nonzero() {
        assert!(Quantity::ZERO.is_zero());
        assert!(Quantity(0).is_zero());
        assert!(!Quantity(1).is_zero());
    }

    #[test]
    fn checked_add_detects_overflow() {
        assert_eq!(Quantity(u64::MAX).checked_add(Quantity(1)), None);
        assert_eq!(Quantity(2).checked_add(Quantity(3)), Some(Quantity(5)));
    }
}
