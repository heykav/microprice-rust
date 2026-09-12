//! Typed errors for `microprice-core`. Library code never panics or returns
//! a bare `String` for an expected failure mode — every failure a caller
//! might reasonably want to match on gets its own variant.

use thiserror::Error;

/// Errors produced by `microprice-core`'s primitive types.
///
/// This will grow in later phases (calibration, serialization each get
/// their own error variants as those phases land).
///
/// Not `Copy` (dropped after Phase 2 added `InvalidBucketConfig`, whose
/// `reason` is a heap-allocated `String` — fine, since bucket
/// *configuration* is a one-time setup cost, not part of the
/// zero-allocation hot path that state *encoding* must be).
#[derive(Debug, Error, Clone, PartialEq, Eq)]
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

    /// A [`crate::state::ImbalanceBucketing`] or
    /// [`crate::state::SpreadBucketing`] configuration was invalid — e.g.
    /// zero imbalance buckets, or spread bucket bounds that aren't sorted,
    /// aren't strictly increasing, or aren't all positive. Raised at
    /// *configuration construction*, never at encode time, per the Phase 2
    /// requirement that a bad config fails immediately rather than
    /// surfacing confusingly during prediction.
    #[error("invalid bucket configuration: {reason}")]
    InvalidBucketConfig { reason: String },

    /// A book's spread (in ticks) didn't fall within any configured
    /// [`crate::state::SpreadBucketing`] bucket — currently only reachable
    /// with a spread less than 1 tick (a locked or crossed market), since a
    /// validly-*constructed* `SpreadBucketing` always has an unbounded top
    /// bucket covering every spread from its lowest configured bound
    /// upward. This is a data-time condition, not a config error: the
    /// state space's V1 scope is spread >= 1 tick, matching a normally-
    /// validated (non-crossed, non-locked) `TopOfBook`.
    #[error("spread {spread_ticks} ticks is out of range for the configured spread buckets (state space requires spread >= 1 tick)")]
    SpreadOutOfRange { spread_ticks: i64 },
}
