//! Top-of-book (L1) order book state and its validation.

use crate::error::MicroPriceError;
use crate::price::PriceTicks;
use crate::quantity::Quantity;
use std::fmt;

/// Spread in ticks: `S = ask_price - bid_price`.
///
/// Can be zero (locked market) or negative (crossed market) — whether that
/// is *allowed* to exist at all is controlled by [`BookValidationPolicy`] at
/// [`TopOfBook`] construction time, not by this type, which just represents
/// whatever difference the (already-validated-or-not) book implies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SpreadTicks(pub i64);

impl fmt::Display for SpreadTicks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}ticks", self.0)
    }
}

/// Controls what price relationships between best bid and best ask
/// `TopOfBook::new` will accept. This is a modeling decision (see
/// `docs/model-spec.md`), not a hidden default: every call site names the
/// policy it wants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BookValidationPolicy {
    /// Require `ask_price > bid_price` strictly. The right default for
    /// almost all consumers — a normal, healthy quote.
    RejectCrossedAndLocked,
    /// Require `ask_price >= bid_price`. Accepts a locked market
    /// (`ask == bid`), rejects a crossed one (`ask < bid`).
    RejectCrossedAllowLocked,
    /// No price-ordering check at all. For research on raw/malformed feeds
    /// where crossed/locked events are themselves the object of study —
    /// not the right choice for anything computing a tradeable price.
    AllowAll,
}

/// A single top-of-book (L1) snapshot: best bid/ask price and quantity.
///
/// Fields are public, matching the shape callers need for cheap
/// construction in hot paths; the only way to enforce the validation
/// invariant is at construction via [`TopOfBook::new`] — nothing prevents
/// later direct field mutation from re-introducing an invalid state. That's
/// a real, known limitation of this exact shape (see the crate's top-level
/// docs), carried forward from the project's own specified struct layout
/// rather than silently changed to a fully encapsulated type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TopOfBook {
    pub bid_price: PriceTicks,
    pub bid_qty: Quantity,
    pub ask_price: PriceTicks,
    pub ask_qty: Quantity,
}

impl TopOfBook {
    /// Constructs a `TopOfBook`, validating the bid/ask price relationship
    /// against `policy`. Does **not** validate quantities being zero —
    /// a one-sided empty book is a valid `TopOfBook` (imbalance handles
    /// that case explicitly; see [`crate::imbalance::Imbalance::compute`]).
    pub fn new(
        bid_price: PriceTicks,
        bid_qty: Quantity,
        ask_price: PriceTicks,
        ask_qty: Quantity,
        policy: BookValidationPolicy,
    ) -> Result<TopOfBook, MicroPriceError> {
        let ordering_ok = match policy {
            BookValidationPolicy::RejectCrossedAndLocked => ask_price.0 > bid_price.0,
            BookValidationPolicy::RejectCrossedAllowLocked => ask_price.0 >= bid_price.0,
            BookValidationPolicy::AllowAll => true,
        };
        if !ordering_ok {
            return Err(MicroPriceError::InvalidBookOrdering {
                bid_ticks: bid_price.0,
                ask_ticks: ask_price.0,
                policy,
            });
        }
        Ok(TopOfBook {
            bid_price,
            bid_qty,
            ask_price,
            ask_qty,
        })
    }

    /// `S = ask_price - bid_price`, in ticks.
    ///
    /// Uses plain `i64` subtraction: at one cent per tick, `i64` covers a
    /// price difference of roughly ±$92 quintillion, so overflow here is
    /// not a realistic concern for any real instrument — a documented
    /// simplification, not a silent gap.
    pub fn spread(&self) -> SpreadTicks {
        SpreadTicks(self.ask_price.0 - self.bid_price.0)
    }

    /// `M = (bid_price + ask_price) / 2`, in ticks, truncating toward zero
    /// when the sum is odd. See `docs/model-spec.md`'s Open Questions for
    /// why a half-tick-precise mid-price type is deferred to a later phase.
    pub fn mid_price_ticks(&self) -> i64 {
        (self.bid_price.0 + self.ask_price.0) / 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn book(bid: i64, bid_qty: u64, ask: i64, ask_qty: u64) -> Result<TopOfBook, MicroPriceError> {
        TopOfBook::new(
            PriceTicks(bid),
            Quantity(bid_qty),
            PriceTicks(ask),
            Quantity(ask_qty),
            BookValidationPolicy::RejectCrossedAndLocked,
        )
    }

    #[test]
    fn valid_book_constructs() {
        let b = book(10000, 500, 10001, 300).unwrap();
        assert_eq!(b.spread(), SpreadTicks(1));
    }

    #[test]
    fn crossed_book_is_rejected_under_default_policy() {
        let result = book(10001, 500, 10000, 300);
        assert!(matches!(
            result,
            Err(MicroPriceError::InvalidBookOrdering { .. })
        ));
    }

    #[test]
    fn locked_book_is_rejected_under_default_policy() {
        let result = book(10000, 500, 10000, 300);
        assert!(matches!(
            result,
            Err(MicroPriceError::InvalidBookOrdering { .. })
        ));
    }

    #[test]
    fn locked_book_is_accepted_under_reject_crossed_allow_locked() {
        let result = TopOfBook::new(
            PriceTicks(10000),
            Quantity(500),
            PriceTicks(10000),
            Quantity(300),
            BookValidationPolicy::RejectCrossedAllowLocked,
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap().spread(), SpreadTicks(0));
    }

    #[test]
    fn crossed_book_is_still_rejected_under_reject_crossed_allow_locked() {
        let result = TopOfBook::new(
            PriceTicks(10001),
            Quantity(500),
            PriceTicks(10000),
            Quantity(300),
            BookValidationPolicy::RejectCrossedAllowLocked,
        );
        assert!(matches!(
            result,
            Err(MicroPriceError::InvalidBookOrdering { .. })
        ));
    }

    #[test]
    fn crossed_and_locked_books_are_both_accepted_under_allow_all() {
        for (bid, ask) in [(10000, 10000), (10001, 10000)] {
            let result = TopOfBook::new(
                PriceTicks(bid),
                Quantity(500),
                PriceTicks(ask),
                Quantity(300),
                BookValidationPolicy::AllowAll,
            );
            assert!(result.is_ok());
        }
    }

    #[test]
    fn one_sided_zero_depth_is_a_valid_book() {
        // Zero quantity on one side is a valid TopOfBook - imbalance is the
        // layer that gives this exact, well-defined semantics (I = 0 or 1).
        let b = book(10000, 0, 10001, 300).unwrap();
        assert_eq!(b.bid_qty, Quantity::ZERO);
        let b2 = book(10000, 500, 10001, 0).unwrap();
        assert_eq!(b2.ask_qty, Quantity::ZERO);
    }

    #[test]
    fn both_sides_zero_depth_is_still_a_valid_top_of_book() {
        // TopOfBook construction only validates price ordering; the
        // "both sides empty" failure belongs to Imbalance::compute, not here.
        let b = book(10000, 0, 10001, 0).unwrap();
        assert_eq!(b.bid_qty, Quantity::ZERO);
        assert_eq!(b.ask_qty, Quantity::ZERO);
    }

    #[test]
    fn extreme_quantities_do_not_overflow_construction() {
        let b = book(10000, u64::MAX, 10001, u64::MAX).unwrap();
        assert_eq!(b.bid_qty, Quantity(u64::MAX));
    }

    #[test]
    fn spread_calculation_is_exact_for_several_widths() {
        for (bid, ask, expected) in [(10000, 10001, 1), (10000, 10005, 5), (10000, 10100, 100)] {
            let b = book(bid, 100, ask, 100).unwrap();
            assert_eq!(b.spread(), SpreadTicks(expected));
        }
    }

    #[test]
    fn mid_price_truncates_toward_zero_on_odd_sum() {
        let b = book(10000, 100, 10001, 100).unwrap();
        assert_eq!(b.mid_price_ticks(), 10000); // (10000+10001)/2 = 10000 (integer division)
    }

    #[test]
    fn mid_price_is_exact_on_even_sum() {
        let b = TopOfBook::new(
            PriceTicks(10000),
            Quantity(100),
            PriceTicks(10002),
            Quantity(100),
            BookValidationPolicy::RejectCrossedAndLocked,
        )
        .unwrap();
        assert_eq!(b.mid_price_ticks(), 10001);
    }
}
