//! Typed errors for `microprice-core`. Library code never panics or returns
//! a bare `String` for an expected failure mode — every failure a caller
//! might reasonably want to match on gets its own variant.

use thiserror::Error;

/// Errors produced by `microprice-core`'s primitive types.
///
/// This will grow in later phases (state encoding, calibration,
/// serialization each get their own error variants as those phases land);
/// Phase 1 only needs the variants primitive-type construction can produce.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum MicroPriceError {
    /// A book had zero quantity resting on *both* the best bid and the best
    /// ask, so queue imbalance (`Qb / (Qb + Qa)`) is a `0/0` with no
    /// principled value — see `docs/model-spec.md`'s "Degenerate cases"
    /// table. This is not raised for a one-sided empty book (`I = 0` or
    /// `I = 1` are both well-defined and exact).
    #[error("book has zero quantity on both the bid and ask side; queue imbalance is undefined for an empty book")]
    EmptyBook,

    /// The book's best bid/ask prices violate the requested
    /// [`BookValidationPolicy`](crate::book::BookValidationPolicy) — e.g. a
    /// crossed or locked market when the policy requires neither.
    #[error("book failed price-ordering validation: bid={bid_ticks} ask={ask_ticks} (policy={policy:?})")]
    InvalidBookOrdering {
        bid_ticks: i64,
        ask_ticks: i64,
        policy: crate::book::BookValidationPolicy,
    },

    /// `Qb + Qa` overflowed `u64`. Only reachable with quantities near
    /// `u64::MAX`, which is not a realistic resting size for any real
    /// instrument, but the failure is checked rather than silently wrapped.
    #[error("bid_qty + ask_qty overflowed u64 (bid_qty={bid_qty}, ask_qty={ask_qty})")]
    QuantityOverflow { bid_qty: u64, ask_qty: u64 },
}
