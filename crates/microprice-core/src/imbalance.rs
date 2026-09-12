//! Queue imbalance: `I = Qb / (Qb + Qa)`, `0 <= I <= 1`.
//!
//! See `docs/model-spec.md`'s "Queue imbalance" section for the full
//! derivation and the degenerate-case table this implementation follows
//! exactly.

use crate::error::MicroPriceError;
use crate::quantity::Quantity;

/// Queue imbalance at the top of book, guaranteed to satisfy
/// `0.0 <= value <= 1.0` for every successfully-constructed instance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Imbalance(f64);

impl Imbalance {
    /// Computes `I = Qb / (Qb + Qa)`.
    ///
    /// - `bid_qty = 0, ask_qty > 0` → `Ok(Imbalance(0.0))`, exact.
    /// - `ask_qty = 0, bid_qty > 0` → `Ok(Imbalance(1.0))`, exact.
    /// - both zero → `Err(MicroPriceError::EmptyBook)`, per
    ///   `docs/model-spec.md` — this is a forced consequence of `0/0` being
    ///   undefined, not an arbitrary policy choice.
    pub fn compute(bid_qty: Quantity, ask_qty: Quantity) -> Result<Imbalance, MicroPriceError> {
        let total = bid_qty
            .checked_add(ask_qty)
            .ok_or(MicroPriceError::QuantityOverflow {
                bid_qty: bid_qty.0,
                ask_qty: ask_qty.0,
            })?;
        if total.is_zero() {
            return Err(MicroPriceError::EmptyBook);
        }
        Ok(Imbalance(bid_qty.0 as f64 / total.0 as f64))
    }

    /// The imbalance value, guaranteed to be in `[0.0, 1.0]`.
    pub fn value(self) -> f64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balanced_book_is_one_half() {
        let i = Imbalance::compute(Quantity(100), Quantity(100)).unwrap();
        assert_eq!(i.value(), 0.5);
    }

    #[test]
    fn zero_ask_qty_gives_exact_one() {
        let i = Imbalance::compute(Quantity(500), Quantity(0)).unwrap();
        assert_eq!(i.value(), 1.0);
    }

    #[test]
    fn zero_bid_qty_gives_exact_zero() {
        let i = Imbalance::compute(Quantity(0), Quantity(500)).unwrap();
        assert_eq!(i.value(), 0.0);
    }

    #[test]
    fn both_zero_is_an_error_not_a_guessed_value() {
        let result = Imbalance::compute(Quantity(0), Quantity(0));
        assert_eq!(result, Err(MicroPriceError::EmptyBook));
    }

    #[test]
    fn value_is_always_in_unit_interval() {
        for (b, a) in [(1, 999), (999, 1), (1, 1), (1_000_000, 1)] {
            let i = Imbalance::compute(Quantity(b), Quantity(a)).unwrap();
            assert!((0.0..=1.0).contains(&i.value()));
        }
    }

    #[test]
    fn extreme_quantities_do_not_overflow_or_panic() {
        let i = Imbalance::compute(Quantity(u64::MAX / 2), Quantity(u64::MAX / 2)).unwrap();
        assert!((i.value() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn overflowing_sum_is_a_typed_error_not_a_panic() {
        let result = Imbalance::compute(Quantity(u64::MAX), Quantity(1));
        assert_eq!(
            result,
            Err(MicroPriceError::QuantityOverflow {
                bid_qty: u64::MAX,
                ask_qty: 1
            })
        );
    }
}
