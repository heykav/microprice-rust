//! State discretization: mapping a [`crate::book::TopOfBook`] to a compact,
//! deterministic [`StateId`].
//!
//! V1 state is exactly two dimensions — imbalance bucket and spread bucket
//! — per `docs/model-spec.md`. Encoding is:
//!
//! ```text
//! state_id = spread_bucket * num_imbalance_buckets + imbalance_bucket
//! ```
//!
//! which packs the two bucket indices into one contiguous `u32` with no
//! hashing and no heap allocation.

use crate::book::{SpreadTicks, TopOfBook};
use crate::error::MicroPriceError;
use crate::imbalance::Imbalance;

/// An opaque, deterministic, contiguous index into the discretized state
/// space defined by a [`StateSpaceConfig`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StateId(pub u32);

/// Uniform bucketing of queue imbalance `I ∈ [0.0, 1.0]` into
/// `num_buckets` equal-width, half-open intervals `[k/N, (k+1)/N)`, with
/// the final bucket closed on the right to include `I == 1.0` exactly.
///
/// Bucket lookup is `O(1)`: one multiply, one floor, one clamp — no search,
/// because the buckets are uniform width by construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImbalanceBucketing {
    num_buckets: u32,
}

impl ImbalanceBucketing {
    /// Fails at construction (not at encode time) if `num_buckets == 0`,
    /// since a zero-bucket configuration can never produce a valid state.
    pub fn new(num_buckets: u32) -> Result<Self, MicroPriceError> {
        if num_buckets == 0 {
            return Err(MicroPriceError::InvalidBucketConfig {
                reason: "imbalance bucket count must be at least 1".to_string(),
            });
        }
        Ok(ImbalanceBucketing { num_buckets })
    }

    pub fn num_buckets(&self) -> u32 {
        self.num_buckets
    }

    /// `floor(I * N)`, clamped to `N - 1` so that `I == 1.0` (which would
    /// otherwise compute to the out-of-range index `N`) lands in the last
    /// bucket instead. `I == 0.0` maps to bucket `0` exactly, with no
    /// clamping needed.
    pub fn bucket_for(&self, imbalance: Imbalance) -> u32 {
        let raw = (imbalance.value() * self.num_buckets as f64).floor();
        // `raw` is finite and >= 0.0 because Imbalance::value() is always
        // in [0.0, 1.0] by construction, so this cast cannot produce a
        // nonsense value the way casting an arbitrary f64 could.
        let raw = raw as u32;
        raw.min(self.num_buckets - 1)
    }

    /// The half-open `[lower, upper)` imbalance range a bucket index
    /// covers (the last bucket is `[lower, upper]`, closed on the right),
    /// for debugging/inspection — see [`StateDescription`].
    pub fn bucket_range(&self, bucket: u32) -> (f64, f64) {
        let width = 1.0 / self.num_buckets as f64;
        (bucket as f64 * width, (bucket + 1) as f64 * width)
    }
}

/// Explicit (non-uniform) bucketing of spread, in ticks.
///
/// Configured as a strictly increasing list of inclusive upper bounds for
/// every bucket *except* the last, which is always unbounded above. For
/// example, `SpreadBucketing::new(vec![1, 2, 4])` produces 4 buckets:
/// `{1}`, `{2}`, `{3, 4}`, `{5, 6, 7, ...}` — matching the project brief's
/// `[[1], [2], [3,4], [5,inf]]` example. This shape guarantees that once
/// construction succeeds, every spread `>= 1` tick matches exactly one
/// bucket — there is no way to configure a "gap" a valid spread could fall
/// through, which is what lets [`StateSpaceConfig::encode`] be infallible
/// for its spread-bucket lookup on any normally-validated book.
///
/// Lookup is a linear scan, not a binary search: spread bucket counts are
/// small (single digits in practice), and for `k` this small a linear scan
/// over a tiny contiguous array is at least as fast as a binary search and
/// has no branch-misprediction surprises — this is a real-but-modest
/// simplification, not the `O(1)` guarantee `ImbalanceBucketing` has.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpreadBucketing {
    /// Inclusive upper bound (in ticks) of every bucket except the last.
    finite_upper_bounds_ticks: Vec<i64>,
}

impl SpreadBucketing {
    /// `upper_bounds_ticks` must be strictly increasing and every value
    /// must be `>= 1`. An empty vec is valid and means "one bucket, `spread
    /// >= 1`" (no finite subdivision at all).
    pub fn new(upper_bounds_ticks: Vec<i64>) -> Result<Self, MicroPriceError> {
        let mut prev: i64 = 0;
        for &bound in &upper_bounds_ticks {
            if bound < 1 {
                return Err(MicroPriceError::InvalidBucketConfig {
                    reason: format!("spread bucket upper bound {bound} must be >= 1 tick"),
                });
            }
            if bound <= prev {
                return Err(MicroPriceError::InvalidBucketConfig {
                    reason: format!(
                        "spread bucket upper bounds must be strictly increasing \
                         (saw {bound} after {prev})"
                    ),
                });
            }
            prev = bound;
        }
        Ok(SpreadBucketing {
            finite_upper_bounds_ticks: upper_bounds_ticks,
        })
    }

    /// Total bucket count: the finite buckets plus the one unbounded tail
    /// bucket that always exists.
    pub fn num_buckets(&self) -> u32 {
        self.finite_upper_bounds_ticks.len() as u32 + 1
    }

    /// `Err(SpreadOutOfRange)` only for `spread < 1` tick (a locked or
    /// crossed market) — every `spread >= 1` matches some bucket by
    /// construction.
    pub fn bucket_for(&self, spread: SpreadTicks) -> Result<u32, MicroPriceError> {
        if spread.0 < 1 {
            return Err(MicroPriceError::SpreadOutOfRange {
                spread_ticks: spread.0,
            });
        }
        for (i, &bound) in self.finite_upper_bounds_ticks.iter().enumerate() {
            if spread.0 <= bound {
                return Ok(i as u32);
            }
        }
        Ok(self.finite_upper_bounds_ticks.len() as u32)
    }

    /// `(lower_inclusive, upper_inclusive)` in ticks for a bucket index;
    /// `upper_inclusive = None` means the unbounded tail bucket.
    pub fn bucket_range(&self, bucket: u32) -> (i64, Option<i64>) {
        let lower = if bucket == 0 {
            1
        } else {
            self.finite_upper_bounds_ticks[(bucket - 1) as usize] + 1
        };
        let upper = self.finite_upper_bounds_ticks.get(bucket as usize).copied();
        (lower, upper)
    }
}

/// A human-readable decoding of a [`StateId`], for debugging/inspection —
/// never used on the encoding hot path.
#[derive(Debug, Clone, PartialEq)]
pub struct StateDescription {
    pub imbalance_bucket: u32,
    pub imbalance_range: (f64, f64),
    pub spread_bucket: u32,
    /// `(lower_inclusive, upper_inclusive)` in ticks; `upper_inclusive =
    /// None` means the unbounded tail bucket.
    pub spread_range_ticks: (i64, Option<i64>),
}

/// The full V1 state-space configuration: an imbalance bucketing and a
/// spread bucketing, combined into one contiguous `StateId` space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateSpaceConfig {
    imbalance: ImbalanceBucketing,
    spread: SpreadBucketing,
}

impl StateSpaceConfig {
    pub fn new(imbalance: ImbalanceBucketing, spread: SpreadBucketing) -> Self {
        StateSpaceConfig { imbalance, spread }
    }

    /// Total number of distinct states: `num_imbalance_buckets *
    /// num_spread_buckets`. Every `StateId` this config ever produces
    /// satisfies `state_id.0 < state_count()`.
    pub fn state_count(&self) -> u32 {
        self.imbalance.num_buckets() * self.spread.num_buckets()
    }

    /// Encodes a book's state as `spread_bucket * num_imbalance_buckets +
    /// imbalance_bucket`. `O(1)` for the imbalance side; a small linear
    /// scan (see [`SpreadBucketing`]) for the spread side. No heap
    /// allocation.
    ///
    /// Fails only for genuinely out-of-domain input: a both-sides-empty
    /// book (`MicroPriceError::EmptyBook`, from
    /// [`Imbalance::compute`]) or a spread under 1 tick
    /// (`MicroPriceError::SpreadOutOfRange`, from a locked/crossed book
    /// that a non-default [`crate::book::BookValidationPolicy`] let
    /// through). A normally-validated book can never hit either path.
    pub fn encode(&self, book: &TopOfBook) -> Result<StateId, MicroPriceError> {
        let spread_bucket = self.spread.bucket_for(book.spread())?;
        let imbalance = Imbalance::compute(book.bid_qty, book.ask_qty)?;
        let imbalance_bucket = self.imbalance.bucket_for(imbalance);
        Ok(StateId(
            spread_bucket * self.imbalance.num_buckets() + imbalance_bucket,
        ))
    }

    /// Reverses [`StateSpaceConfig::encode`]'s packing for debugging and
    /// inspection. Not used on the hot path.
    pub fn decode(&self, state: StateId) -> StateDescription {
        let num_imbalance = self.imbalance.num_buckets();
        let imbalance_bucket = state.0 % num_imbalance;
        let spread_bucket = state.0 / num_imbalance;
        StateDescription {
            imbalance_bucket,
            imbalance_range: self.imbalance.bucket_range(imbalance_bucket),
            spread_bucket,
            spread_range_ticks: self.spread.bucket_range(spread_bucket),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::book::BookValidationPolicy;
    use crate::price::PriceTicks;
    use crate::quantity::Quantity;

    #[test]
    fn state_id_equality_and_ordering_are_by_value() {
        assert_eq!(StateId(3), StateId(3));
        assert!(StateId(1) < StateId(2));
    }

    // --- ImbalanceBucketing ---

    #[test]
    fn imbalance_bucketing_rejects_zero_buckets() {
        assert!(matches!(
            ImbalanceBucketing::new(0),
            Err(MicroPriceError::InvalidBucketConfig { .. })
        ));
    }

    #[test]
    fn imbalance_zero_maps_to_bucket_zero() {
        let b = ImbalanceBucketing::new(20).unwrap();
        assert_eq!(
            b.bucket_for(Imbalance::compute(Quantity(0), Quantity(1)).unwrap()),
            0
        );
    }

    #[test]
    fn imbalance_one_maps_to_last_bucket_not_out_of_range() {
        let b = ImbalanceBucketing::new(20).unwrap();
        assert_eq!(
            b.bucket_for(Imbalance::compute(Quantity(1), Quantity(0)).unwrap()),
            19
        );
    }

    #[test]
    fn imbalance_bucket_boundaries_match_the_documented_half_open_intervals() {
        let b = ImbalanceBucketing::new(10).unwrap();
        // [0.0,0.1) -> 0, [0.1,0.2) -> 1, ... exactly matching section 5's example.
        let cases = [(0.05, 0), (0.15, 1), (0.25, 2), (0.95, 9)];
        for (i, expected) in cases {
            let imbalance = Imbalance::compute(
                Quantity((i * 1000.0) as u64),
                Quantity(((1.0 - i) * 1000.0) as u64),
            )
            .unwrap();
            assert_eq!(b.bucket_for(imbalance), expected, "I={i}");
        }
    }

    #[test]
    fn imbalance_bucket_range_covers_the_unit_interval_with_no_gaps() {
        let b = ImbalanceBucketing::new(4).unwrap();
        let ranges: Vec<_> = (0..4).map(|k| b.bucket_range(k)).collect();
        assert_eq!(ranges[0].0, 0.0);
        assert_eq!(ranges[3].1, 1.0);
        for k in 0..3 {
            assert_eq!(ranges[k].1, ranges[k + 1].0);
        }
    }

    // --- SpreadBucketing ---

    #[test]
    fn spread_bucketing_matches_the_documented_example() {
        // [[1],[2],[3,4],[5,inf]] from docs/model-spec.md / section 14.
        let s = SpreadBucketing::new(vec![1, 2, 4]).unwrap();
        assert_eq!(s.num_buckets(), 4);
        assert_eq!(s.bucket_for(SpreadTicks(1)).unwrap(), 0);
        assert_eq!(s.bucket_for(SpreadTicks(2)).unwrap(), 1);
        assert_eq!(s.bucket_for(SpreadTicks(3)).unwrap(), 2);
        assert_eq!(s.bucket_for(SpreadTicks(4)).unwrap(), 2);
        assert_eq!(s.bucket_for(SpreadTicks(5)).unwrap(), 3);
        assert_eq!(s.bucket_for(SpreadTicks(1_000_000)).unwrap(), 3); // unbounded tail
    }

    #[test]
    fn spread_bucketing_with_no_finite_bounds_is_one_catch_all_bucket() {
        let s = SpreadBucketing::new(vec![]).unwrap();
        assert_eq!(s.num_buckets(), 1);
        assert_eq!(s.bucket_for(SpreadTicks(1)).unwrap(), 0);
        assert_eq!(s.bucket_for(SpreadTicks(9_999)).unwrap(), 0);
    }

    #[test]
    fn spread_bucketing_rejects_non_increasing_bounds() {
        assert!(matches!(
            SpreadBucketing::new(vec![2, 2]),
            Err(MicroPriceError::InvalidBucketConfig { .. })
        ));
        assert!(matches!(
            SpreadBucketing::new(vec![3, 2]),
            Err(MicroPriceError::InvalidBucketConfig { .. })
        ));
    }

    #[test]
    fn spread_bucketing_rejects_a_bound_below_one_tick() {
        assert!(matches!(
            SpreadBucketing::new(vec![0]),
            Err(MicroPriceError::InvalidBucketConfig { .. })
        ));
        assert!(matches!(
            SpreadBucketing::new(vec![-1]),
            Err(MicroPriceError::InvalidBucketConfig { .. })
        ));
    }

    #[test]
    fn spread_zero_or_negative_is_out_of_range_not_a_config_error() {
        let s = SpreadBucketing::new(vec![1, 2, 4]).unwrap();
        assert!(matches!(
            s.bucket_for(SpreadTicks(0)),
            Err(MicroPriceError::SpreadOutOfRange { spread_ticks: 0 })
        ));
        assert!(matches!(
            s.bucket_for(SpreadTicks(-3)),
            Err(MicroPriceError::SpreadOutOfRange { spread_ticks: -3 })
        ));
    }

    #[test]
    fn spread_bucket_range_round_trips_the_documented_example() {
        let s = SpreadBucketing::new(vec![1, 2, 4]).unwrap();
        assert_eq!(s.bucket_range(0), (1, Some(1)));
        assert_eq!(s.bucket_range(1), (2, Some(2)));
        assert_eq!(s.bucket_range(2), (3, Some(4)));
        assert_eq!(s.bucket_range(3), (5, None));
    }

    // --- StateSpaceConfig ---

    fn book(bid: i64, bid_qty: u64, ask: i64, ask_qty: u64) -> TopOfBook {
        TopOfBook::new(
            PriceTicks(bid),
            Quantity(bid_qty),
            PriceTicks(ask),
            Quantity(ask_qty),
            BookValidationPolicy::RejectCrossedAndLocked,
        )
        .unwrap()
    }

    fn example_config() -> StateSpaceConfig {
        // 20 imbalance buckets x 4 spread buckets = 80 states, matching
        // section 12's worked example exactly.
        StateSpaceConfig::new(
            ImbalanceBucketing::new(20).unwrap(),
            SpreadBucketing::new(vec![1, 2, 4]).unwrap(),
        )
    }

    #[test]
    fn state_count_matches_the_documented_worked_example() {
        assert_eq!(example_config().state_count(), 80);
    }

    #[test]
    fn every_encoded_state_id_is_within_state_count() {
        let cfg = example_config();
        for spread in [1, 2, 3, 4, 5, 100] {
            for (bid_qty, ask_qty) in [(1, 999), (500, 500), (999, 1)] {
                let b = book(10000, bid_qty, 10000 + spread, ask_qty);
                let state = cfg.encode(&b).unwrap();
                assert!(state.0 < cfg.state_count());
            }
        }
    }

    #[test]
    fn same_input_always_encodes_to_the_same_state() {
        let cfg = example_config();
        let b = book(10000, 412, 10002, 88);
        assert_eq!(cfg.encode(&b).unwrap(), cfg.encode(&b).unwrap());
    }

    #[test]
    fn decode_inverts_encode() {
        let cfg = example_config();
        let b = book(10000, 700, 10003, 300);
        let state = cfg.encode(&b).unwrap();
        let desc = cfg.decode(state);
        // Independently recompute the expected buckets and check decode
        // agrees, rather than just checking decode(encode(x)) doesn't panic.
        let expected_imbalance_bucket = ImbalanceBucketing::new(20)
            .unwrap()
            .bucket_for(Imbalance::compute(Quantity(700), Quantity(300)).unwrap());
        let expected_spread_bucket = SpreadBucketing::new(vec![1, 2, 4])
            .unwrap()
            .bucket_for(b.spread())
            .unwrap();
        assert_eq!(desc.imbalance_bucket, expected_imbalance_bucket);
        assert_eq!(desc.spread_bucket, expected_spread_bucket);
    }

    #[test]
    fn empty_book_state_encoding_fails_with_empty_book_not_a_panic() {
        let cfg = example_config();
        let b = book(10000, 0, 10001, 0);
        assert_eq!(cfg.encode(&b), Err(MicroPriceError::EmptyBook));
    }

    #[test]
    fn locked_book_state_encoding_fails_with_spread_out_of_range() {
        let cfg = example_config();
        let b = TopOfBook::new(
            PriceTicks(10000),
            Quantity(100),
            PriceTicks(10000),
            Quantity(100),
            BookValidationPolicy::RejectCrossedAllowLocked,
        )
        .unwrap();
        assert_eq!(
            cfg.encode(&b),
            Err(MicroPriceError::SpreadOutOfRange { spread_ticks: 0 })
        );
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use crate::book::BookValidationPolicy;
    use crate::price::PriceTicks;
    use crate::quantity::Quantity;
    use proptest::prelude::*;

    fn config_strategy() -> impl Strategy<Value = StateSpaceConfig> {
        (1u32..=50, 0usize..=6).prop_flat_map(|(num_imbalance, num_finite_spread_buckets)| {
            // Generate `num_finite_spread_buckets` strictly increasing
            // positive bounds by generating sorted-unique small deltas.
            proptest::collection::btree_set(1i64..=50, num_finite_spread_buckets).prop_map(
                move |bounds_set| {
                    let bounds: Vec<i64> = bounds_set.into_iter().collect();
                    StateSpaceConfig::new(
                        ImbalanceBucketing::new(num_imbalance).unwrap(),
                        SpreadBucketing::new(bounds).unwrap(),
                    )
                },
            )
        })
    }

    proptest! {
        #[test]
        fn imbalance_bucket_is_always_in_range(
            num_buckets in 1u32..=1000,
            i in 0.0f64..=1.0,
        ) {
            let b = ImbalanceBucketing::new(num_buckets).unwrap();
            // Reconstruct an Imbalance via a book rather than a private
            // constructor, keeping this test honest about the only way
            // real callers ever get an Imbalance.
            let bid = (i * 1_000_000.0).round() as u64;
            let ask = 1_000_000u64.saturating_sub(bid).max(1);
            let imbalance = Imbalance::compute(Quantity(bid), Quantity(ask)).unwrap();
            let bucket = b.bucket_for(imbalance);
            prop_assert!(bucket < num_buckets);
        }

        #[test]
        fn state_id_is_always_less_than_state_count(
            cfg in config_strategy(),
            bid_ticks in 1i64..1_000_000,
            spread_ticks in 1i64..200,
            bid_qty in 1u64..1_000_000,
            ask_qty in 1u64..1_000_000,
        ) {
            let b = TopOfBook::new(
                PriceTicks(bid_ticks),
                Quantity(bid_qty),
                PriceTicks(bid_ticks + spread_ticks),
                Quantity(ask_qty),
                BookValidationPolicy::RejectCrossedAndLocked,
            )
            .unwrap();
            let state = cfg.encode(&b).unwrap();
            prop_assert!(state.0 < cfg.state_count());
        }

        #[test]
        fn encoding_is_deterministic(
            cfg in config_strategy(),
            bid_ticks in 1i64..1_000_000,
            spread_ticks in 1i64..200,
            bid_qty in 1u64..1_000_000,
            ask_qty in 1u64..1_000_000,
        ) {
            let b = TopOfBook::new(
                PriceTicks(bid_ticks),
                Quantity(bid_qty),
                PriceTicks(bid_ticks + spread_ticks),
                Quantity(ask_qty),
                BookValidationPolicy::RejectCrossedAndLocked,
            )
            .unwrap();
            prop_assert_eq!(cfg.encode(&b).unwrap(), cfg.encode(&b).unwrap());
        }

        #[test]
        fn decoded_buckets_are_always_within_their_respective_bucket_counts(
            cfg in config_strategy(),
            bid_ticks in 1i64..1_000_000,
            spread_ticks in 1i64..200,
            bid_qty in 1u64..1_000_000,
            ask_qty in 1u64..1_000_000,
        ) {
            let b = TopOfBook::new(
                PriceTicks(bid_ticks),
                Quantity(bid_qty),
                PriceTicks(bid_ticks + spread_ticks),
                Quantity(ask_qty),
                BookValidationPolicy::RejectCrossedAndLocked,
            )
            .unwrap();
            let state = cfg.encode(&b).unwrap();
            let desc = cfg.decode(state);
            prop_assert!(desc.imbalance_bucket < cfg.imbalance.num_buckets());
            prop_assert!(desc.spread_bucket < cfg.spread.num_buckets());
        }
    }
}
