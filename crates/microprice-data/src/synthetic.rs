//! A deterministic synthetic order-book event generator.
//!
//! **This is not a claim of realistic exchange dynamics** (see section 56
//! of the project brief and `docs/model-spec.md`). Its job is narrower and
//! more useful for development: a fully reproducible (same seed -> byte-
//! identical event sequence), configurable stream of *valid* `BookEvent`s
//! that CI, examples, and later phases (transition counting, the
//! known-truth solver validation in Phase 6) can run against without a
//! market-data subscription or a non-deterministic test suite.
//!
//! ## The (disclosed, not hidden) generative model
//!
//! At each step, exactly one of the following happens, chosen by a
//! weighted random draw against the configured rates (any leftover
//! probability mass is a "nothing happens this step" no-op):
//!
//! - **Price move** (`price_move_probability`): mid shifts by exactly one
//!   tick. Direction is a coin flip *unless* `imbalance_persistence` says
//!   otherwise: with probability `imbalance_persistence` the move repeats
//!   the *previous* price move's direction instead of being redrawn
//!   (`0.0` = every move's direction is an independent coin flip; `1.0` =
//!   every move repeats the last one's direction, producing a persistent
//!   trend). This is a deliberately simple, self-contained
//!   interpretation of "persistence" — a Markov chain on move direction,
//!   not a function of queue imbalance itself — and is this generator's
//!   own modeling choice, not derived from any cited paper. After a move,
//!   both queues are reset to fresh random quantities near the configured
//!   initial sizes (modeling "the book rebuilds after a price change").
//! - **Bid/ask arrival** (`arrival_rate`, split evenly between sides):
//!   the chosen side's resting quantity increases by a random amount.
//! - **Bid/ask cancel** (`cancel_rate`, split evenly between sides): the
//!   chosen side's resting quantity decreases by a random amount (can
//!   deplete a side to zero — a real, valid `TopOfBook` state; see
//!   `docs/model-spec.md`'s imbalance degenerate-case table).
//! - **Market buy/sell** (`market_order_rate`, split evenly): depletes
//!   the *opposite* side's queue (a market buy consumes ask depth, a
//!   market sell consumes bid depth) by a random amount, which can also
//!   drive a side to zero.
//!
//! If the configured rates leave both sides at zero simultaneously (which
//! `Imbalance::compute` cannot give a value for), the generator detects
//! this and forces a price-move step immediately rather than emitting a
//! `BookEvent` no consumer downstream could use.

use microprice_core::{BookEvent, BookValidationPolicy, PriceTicks, Quantity, SymbolId, TopOfBook};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::error::DataError;
use crate::source::MarketDataSource;

/// Configuration for [`SyntheticEventGenerator`]. Every rate is a
/// per-step probability and must be `>= 0.0`; the four rates
/// (`price_move_probability`, `arrival_rate`, `cancel_rate`,
/// `market_order_rate`) must sum to `<= 1.0` (the remainder is the
/// no-op/"nothing happens" probability each step) — validated at
/// construction, not discovered mid-generation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SyntheticConfig {
    pub symbol: SymbolId,
    pub initial_mid_ticks: i64,
    pub initial_spread_ticks: i64,
    pub initial_bid_qty: u64,
    pub initial_ask_qty: u64,
    pub arrival_rate: f64,
    pub cancel_rate: f64,
    pub market_order_rate: f64,
    /// `0.0` (no persistence, direction is iid each price move) to `1.0`
    /// (a price move always repeats the previous move's direction). See
    /// the module docs for why this specific parameterization is this
    /// generator's own choice, not a literature-derived one.
    pub imbalance_persistence: f64,
    pub price_move_probability: f64,
    pub seed: u64,
}

impl SyntheticConfig {
    fn validate(&self) -> Result<(), DataError> {
        if self.initial_spread_ticks < 1 {
            return Err(DataError::InvalidSyntheticConfig {
                reason: format!(
                    "initial_spread_ticks must be >= 1, got {}",
                    self.initial_spread_ticks
                ),
            });
        }
        for (name, rate) in [
            ("arrival_rate", self.arrival_rate),
            ("cancel_rate", self.cancel_rate),
            ("market_order_rate", self.market_order_rate),
            ("price_move_probability", self.price_move_probability),
            ("imbalance_persistence", self.imbalance_persistence),
        ] {
            if !(0.0..=1.0).contains(&rate) {
                return Err(DataError::InvalidSyntheticConfig {
                    reason: format!("{name} must be in [0.0, 1.0], got {rate}"),
                });
            }
        }
        let total = self.arrival_rate
            + self.cancel_rate
            + self.market_order_rate
            + self.price_move_probability;
        if total > 1.0 {
            return Err(DataError::InvalidSyntheticConfig {
                reason: format!(
                    "arrival_rate + cancel_rate + market_order_rate + \
                     price_move_probability must sum to <= 1.0, got {total}"
                ),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LastMoveDirection {
    Up,
    Down,
}

/// A deterministic, configurable synthetic `BookEvent` stream. See the
/// module-level docs for the generative model.
pub struct SyntheticEventGenerator {
    config: SyntheticConfig,
    rng: ChaCha8Rng,
    mid_ticks: i64,
    bid_qty: u64,
    ask_qty: u64,
    sequence: u64,
    timestamp_ns: u64,
    last_move_direction: LastMoveDirection,
}

impl SyntheticEventGenerator {
    pub fn new(config: SyntheticConfig) -> Result<Self, DataError> {
        config.validate()?;
        Ok(SyntheticEventGenerator {
            rng: ChaCha8Rng::seed_from_u64(config.seed),
            mid_ticks: config.initial_mid_ticks,
            bid_qty: config.initial_bid_qty,
            ask_qty: config.initial_ask_qty,
            sequence: 0,
            timestamp_ns: 0,
            last_move_direction: LastMoveDirection::Up,
            config,
        })
    }

    fn half_spread_floor(&self) -> i64 {
        self.config.initial_spread_ticks / 2
    }

    fn current_book(&self) -> TopOfBook {
        let half = self.half_spread_floor();
        let bid = self.mid_ticks - half;
        let ask = bid + self.config.initial_spread_ticks;
        // Construction cannot fail here: initial_spread_ticks >= 1 is
        // validated at config-construction time and nothing in step()
        // ever changes it, so bid < ask always holds.
        TopOfBook::new(
            PriceTicks(bid),
            Quantity(self.bid_qty),
            PriceTicks(ask),
            Quantity(self.ask_qty),
            BookValidationPolicy::RejectCrossedAndLocked,
        )
        .expect("synthetic generator must always maintain a valid spread")
    }

    fn random_qty_delta(&mut self, current: u64) -> u64 {
        // A modest, bounded random change - large enough to be visible,
        // small enough that a single step can't implausibly jump a queue
        // from near-zero to enormous.
        let scale = (self
            .config
            .initial_bid_qty
            .max(self.config.initial_ask_qty)
            .max(1)
            / 10)
            .max(1);
        self.rng
            .random_range(1..=scale.max(1))
            .min(current.max(scale))
    }

    fn reset_queues_after_price_move(&mut self) {
        let bid_scale = self.config.initial_bid_qty.max(1);
        let ask_scale = self.config.initial_ask_qty.max(1);
        self.bid_qty = self
            .rng
            .random_range((bid_scale / 2).max(1)..=bid_scale.saturating_mul(2).max(1));
        self.ask_qty = self
            .rng
            .random_range((ask_scale / 2).max(1)..=ask_scale.saturating_mul(2).max(1));
    }

    fn do_price_move(&mut self) {
        let repeat = self.rng.random_bool(self.config.imbalance_persistence);
        let direction = if repeat {
            self.last_move_direction
        } else if self.rng.random_bool(0.5) {
            LastMoveDirection::Up
        } else {
            LastMoveDirection::Down
        };
        self.mid_ticks += match direction {
            LastMoveDirection::Up => 1,
            LastMoveDirection::Down => -1,
        };
        self.last_move_direction = direction;
        self.reset_queues_after_price_move();
    }

    /// Advances internal state by exactly one step, choosing an action per
    /// the module-level docs' weighted draw. Guarantees the resulting
    /// book is never both-sides-empty (forcing a price move instead, so
    /// every emitted `BookEvent` is directly usable by
    /// `Imbalance::compute`).
    fn step(&mut self) {
        let roll: f64 = self.rng.random_range(0.0..1.0);
        let c = &self.config;

        let price_move_end = c.price_move_probability;
        let arrival_end = price_move_end + c.arrival_rate;
        let cancel_end = arrival_end + c.cancel_rate;
        let market_end = cancel_end + c.market_order_rate;

        if roll < price_move_end {
            self.do_price_move();
        } else if roll < arrival_end {
            if self.rng.random_bool(0.5) {
                self.bid_qty = self
                    .bid_qty
                    .saturating_add(self.random_qty_delta(self.bid_qty));
            } else {
                self.ask_qty = self
                    .ask_qty
                    .saturating_add(self.random_qty_delta(self.ask_qty));
            }
        } else if roll < cancel_end {
            if self.rng.random_bool(0.5) {
                let delta = self.random_qty_delta(self.bid_qty.max(1));
                self.bid_qty = self.bid_qty.saturating_sub(delta);
            } else {
                let delta = self.random_qty_delta(self.ask_qty.max(1));
                self.ask_qty = self.ask_qty.saturating_sub(delta);
            }
        } else if roll < market_end {
            if self.rng.random_bool(0.5) {
                // market buy: consumes ask depth
                let delta = self.random_qty_delta(self.ask_qty.max(1));
                self.ask_qty = self.ask_qty.saturating_sub(delta);
            } else {
                // market sell: consumes bid depth
                let delta = self.random_qty_delta(self.bid_qty.max(1));
                self.bid_qty = self.bid_qty.saturating_sub(delta);
            }
        }
        // else: no-op step - nothing changes, but an event is still
        // emitted (a repeated, unchanged book is a legitimate real-world
        // observation, not something to suppress).

        if self.bid_qty == 0 && self.ask_qty == 0 {
            // Both sides depleted simultaneously: not a state any
            // downstream consumer (Imbalance::compute) can use. Force a
            // price move immediately, which resets both queues - see the
            // module docs' final paragraph.
            self.do_price_move();
        }

        self.sequence += 1;
        // A fixed, modest inter-event gap. Real feeds have irregular
        // gaps; this generator's timestamps exist to give downstream code
        // *something* monotonic and plausible-shaped to sort/window by,
        // not to model real inter-arrival time distributions.
        self.timestamp_ns += 100_000; // 100 microseconds
    }
}

impl MarketDataSource for SyntheticEventGenerator {
    fn next_event(&mut self) -> Option<BookEvent> {
        self.step();
        Some(BookEvent {
            timestamp_ns: self.timestamp_ns,
            sequence: self.sequence,
            symbol: self.config.symbol,
            book: self.current_book(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_config(seed: u64) -> SyntheticConfig {
        SyntheticConfig {
            symbol: SymbolId(1),
            initial_mid_ticks: 10_000,
            initial_spread_ticks: 2,
            initial_bid_qty: 500,
            initial_ask_qty: 500,
            arrival_rate: 0.3,
            cancel_rate: 0.2,
            market_order_rate: 0.2,
            imbalance_persistence: 0.5,
            price_move_probability: 0.1,
            seed,
        }
    }

    #[test]
    fn rejects_rates_that_sum_above_one() {
        let mut cfg = base_config(1);
        cfg.arrival_rate = 0.9;
        cfg.cancel_rate = 0.9;
        assert!(matches!(
            SyntheticEventGenerator::new(cfg),
            Err(DataError::InvalidSyntheticConfig { .. })
        ));
    }

    #[test]
    fn rejects_a_rate_outside_unit_interval() {
        let mut cfg = base_config(1);
        cfg.cancel_rate = -0.1;
        assert!(matches!(
            SyntheticEventGenerator::new(cfg),
            Err(DataError::InvalidSyntheticConfig { .. })
        ));
    }

    #[test]
    fn rejects_sub_one_tick_initial_spread() {
        let mut cfg = base_config(1);
        cfg.initial_spread_ticks = 0;
        assert!(matches!(
            SyntheticEventGenerator::new(cfg),
            Err(DataError::InvalidSyntheticConfig { .. })
        ));
    }

    #[test]
    fn same_seed_produces_an_identical_event_sequence() {
        let mut gen_a = SyntheticEventGenerator::new(base_config(42)).unwrap();
        let mut gen_b = SyntheticEventGenerator::new(base_config(42)).unwrap();
        for _ in 0..500 {
            assert_eq!(gen_a.next_event(), gen_b.next_event());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut gen_a = SyntheticEventGenerator::new(base_config(1)).unwrap();
        let mut gen_b = SyntheticEventGenerator::new(base_config(2)).unwrap();
        let events_a: Vec<_> = (0..200).map(|_| gen_a.next_event()).collect();
        let events_b: Vec<_> = (0..200).map(|_| gen_b.next_event()).collect();
        assert_ne!(events_a, events_b);
    }

    #[test]
    fn timestamps_and_sequence_numbers_are_strictly_increasing() {
        let mut gen = SyntheticEventGenerator::new(base_config(7)).unwrap();
        let mut last_seq = 0u64;
        let mut last_ts = 0u64;
        for _ in 0..1000 {
            let e = gen.next_event().unwrap();
            assert!(e.sequence > last_seq);
            assert!(e.timestamp_ns > last_ts);
            last_seq = e.sequence;
            last_ts = e.timestamp_ns;
        }
    }

    #[test]
    fn every_generated_book_is_never_both_sides_empty() {
        // The one invariant the generator itself must never violate:
        // Imbalance::compute must always be able to handle every emitted
        // book (one-sided-empty is fine and expected; both-sided is not).
        let mut cfg = base_config(9);
        cfg.cancel_rate = 0.5;
        cfg.market_order_rate = 0.5;
        cfg.arrival_rate = 0.0;
        cfg.price_move_probability = 0.0;
        let mut gen = SyntheticEventGenerator::new(cfg).unwrap();
        for _ in 0..5000 {
            let e = gen.next_event().unwrap();
            assert!(
                !(e.book.bid_qty.is_zero() && e.book.ask_qty.is_zero()),
                "generator produced a both-sides-empty book"
            );
        }
    }

    #[test]
    fn every_generated_book_maintains_the_configured_spread() {
        let mut gen = SyntheticEventGenerator::new(base_config(3)).unwrap();
        for _ in 0..1000 {
            let e = gen.next_event().unwrap();
            assert_eq!(e.book.spread().0, 2);
        }
    }

    #[test]
    fn high_price_move_probability_actually_moves_the_price() {
        // A statistical sanity check, not an exact assertion: with
        // price_move_probability = 1.0 every step is a price move, so the
        // mid must actually change from the initial value.
        let mut cfg = base_config(11);
        cfg.price_move_probability = 1.0;
        cfg.arrival_rate = 0.0;
        cfg.cancel_rate = 0.0;
        cfg.market_order_rate = 0.0;
        let mut gen = SyntheticEventGenerator::new(cfg).unwrap();
        let first = gen.next_event().unwrap();
        let tenth = (0..9).fold(first, |_, _| gen.next_event().unwrap());
        assert_ne!(first.book.mid_price_ticks(), tenth.book.mid_price_ticks());
    }

    #[test]
    fn zero_price_move_probability_never_moves_the_mid_when_depletion_cannot_happen() {
        // Isolate the property this test actually means to check: with
        // price_move_probability = 0, the *direct* price-move draw never
        // fires. Cancel/market rates are also zeroed here because with
        // them active, queues can occasionally both hit zero by chance -
        // and the documented both-sides-empty safety net (see the module
        // docs) forces a price move regardless of price_move_probability,
        // since staying at zero/zero isn't a state any consumer can use.
        // That's a real, intentional escape hatch, not a bug - so testing
        // "price moves never happen" needs a config where the escape
        // hatch can't trigger, not a weaker assertion on this one.
        let mut cfg = base_config(13);
        cfg.price_move_probability = 0.0;
        cfg.cancel_rate = 0.0;
        cfg.market_order_rate = 0.0;
        let mut gen = SyntheticEventGenerator::new(cfg).unwrap();
        let initial_mid = cfg.initial_mid_ticks;
        for _ in 0..2000 {
            let e = gen.next_event().unwrap();
            assert_eq!(e.book.mid_price_ticks(), initial_mid);
        }
    }

    #[test]
    fn both_sides_depleting_simultaneously_forces_a_price_move_even_at_zero_probability() {
        // The escape hatch itself, tested directly and honestly (rather
        // than left as an implicit side effect the other test just had to
        // avoid triggering): drive both queues toward zero with
        // price_move_probability = 0.0, and confirm the mid eventually
        // moves anyway - proving the both-sides-empty invariant really is
        // enforced even when it contradicts the configured probability.
        let mut cfg = base_config(17);
        cfg.price_move_probability = 0.0;
        cfg.cancel_rate = 0.5;
        cfg.market_order_rate = 0.5;
        cfg.arrival_rate = 0.0;
        cfg.initial_bid_qty = 5;
        cfg.initial_ask_qty = 5;
        let mut gen = SyntheticEventGenerator::new(cfg).unwrap();
        let initial_mid = cfg.initial_mid_ticks;
        let moved = (0..500)
            .map(|_| gen.next_event().unwrap())
            .any(|e| e.book.mid_price_ticks() != initial_mid);
        assert!(
            moved,
            "expected the both-sides-empty safety net to force at least one price move"
        );
    }
}
