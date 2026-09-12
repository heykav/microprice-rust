//! Streaming transition counting: turns a chronological `BookEvent` stream
//! into per-state visit counts, per-(state,state) transition counts, and
//! per-state signed price-delta sums — the raw material Phase 7's
//! estimator turns into `Q` and `G1`.
//!
//! Uses **event-to-event sampling** (see `docs/model-spec.md`): consecutive
//! events, in `sequence` order, form one observed transition. Dense
//! storage (flat `Vec<u64>`/`Vec<i64>` of length `state_count^2` /
//! `state_count`) is used throughout — appropriate for the V1 state-space
//! sizes (tens to low hundreds of states); a sparse backend for much
//! larger state spaces is a documented, unimplemented extension, not
//! something this module pretends to need yet.

use microprice_core::{BookEvent, StateId, StateSpaceConfig};

use crate::error::CalibrationError;

/// Accumulated transition statistics for one [`StateSpaceConfig`].
#[derive(Debug, Clone, PartialEq)]
pub struct TransitionCounter {
    state_count: u32,
    /// `visits[i]`: number of transitions that *started* at state `i`.
    visits: Vec<u64>,
    /// Flattened `state_count x state_count`: `counts[i * state_count + j]`
    /// is how many observed transitions went from `i` to `j` **without a
    /// price change** (`delta_ticks == 0` — see [`TransitionCounter::record`]).
    /// The price-changing subset is *not* in here at all; it's implicitly
    /// `visits[i] - sum_j counts[i][j]`, since `Q` (Phase 7/8; see the
    /// module docs on `crate::estimator`) is only ever defined over
    /// non-price-changing destinations.
    counts: Vec<u64>,
    /// `delta_sum[i]`: sum of every observed signed mid-price tick delta
    /// over transitions starting at `i`.
    delta_sum: Vec<i64>,
    last_sequence: Option<u64>,
    /// The `(state, mid_ticks)` of the most recently observed *encodable*
    /// event, carried **across** calls to [`TransitionCounter::observe_events`]
    /// (not reset per-call) - this is what lets several sequential calls on
    /// the same counter (streaming ingestion in chunks) correctly form the
    /// transition spanning each call boundary, rather than silently
    /// dropping it. See [`TransitionCounter::merge`] for what this does
    /// and does *not* buy you when chunks are counted in **separate**
    /// counter instances (true parallel processing) instead.
    last_observed: Option<(StateId, i64)>,
}

impl TransitionCounter {
    pub fn new(state_count: u32) -> Self {
        let n = state_count as usize;
        TransitionCounter {
            state_count,
            visits: vec![0; n],
            counts: vec![0; n * n],
            delta_sum: vec![0; n],
            last_sequence: None,
            last_observed: None,
        }
    }

    pub fn state_count(&self) -> u32 {
        self.state_count
    }

    pub fn visits(&self, state: StateId) -> u64 {
        self.visits[state.0 as usize]
    }

    pub fn count(&self, from: StateId, to: StateId) -> u64 {
        self.counts[from.0 as usize * self.state_count as usize + to.0 as usize]
    }

    pub fn delta_sum(&self, state: StateId) -> i64 {
        self.delta_sum[state.0 as usize]
    }

    /// Total observed transitions across every starting state.
    pub fn total_observations(&self) -> u64 {
        self.visits.iter().sum()
    }

    /// Feeds one already-encoded `(from, to)` transition plus its signed
    /// mid-price tick delta directly, bypassing event-to-`StateId`
    /// encoding. This is what [`TransitionCounter::observe_events`] uses
    /// internally, and is exposed directly for the known-truth validation
    /// tests (Phase 6/Prompt 6), which sample transitions straight from a
    /// hand-specified toy Markov chain rather than from `BookEvent`s.
    ///
    /// `counts[from][to]` is only incremented when `delta_ticks == 0` -
    /// i.e. only for the *non-price-changing* subset of transitions, which
    /// is the only thing `Q` (see `crate::estimator`) is defined over. A
    /// transition can land on `to == from` while the price still moved
    /// (the bucket happens to round back to the same index), and that
    /// case must **not** count toward `Q[from][from]` - `visits` and
    /// `delta_sum` still update unconditionally, since those track *every*
    /// observed transition regardless of whether it changed price.
    pub fn record(&mut self, from: StateId, to: StateId, delta_ticks: i64) {
        let n = self.state_count as usize;
        self.visits[from.0 as usize] += 1;
        if delta_ticks == 0 {
            self.counts[from.0 as usize * n + to.0 as usize] += 1;
        }
        self.delta_sum[from.0 as usize] += delta_ticks;
    }

    /// Encodes and records every consecutive pair in a chronological
    /// (`sequence`-ordered) slice of `BookEvent`s for one symbol, using
    /// `config` to map each book to a `StateId`.
    ///
    /// Events whose book fails encoding (an out-of-domain book — see
    /// `StateSpaceConfig::encode`'s own error cases) are skipped rather
    /// than aborting the whole batch, since a single malformed
    /// observation shouldn't discard everything after it; `sequence`
    /// ordering is still checked against every event actually present in
    /// `events`, including skipped ones.
    ///
    /// Calling this multiple times on the same counter with successive,
    /// contiguous chunks of one logical event stream correctly counts the
    /// transition spanning each call's boundary (via `last_observed`) -
    /// equivalent to one call with the whole stream concatenated.
    pub fn observe_events(
        &mut self,
        config: &StateSpaceConfig,
        events: &[BookEvent],
    ) -> Result<(), CalibrationError> {
        let mut prev = self.last_observed;
        for event in events {
            if let Some(last_seq) = self.last_sequence {
                if event.sequence <= last_seq {
                    return Err(CalibrationError::OutOfOrderEvent {
                        previous_sequence: last_seq,
                        sequence: event.sequence,
                    });
                }
            }
            self.last_sequence = Some(event.sequence);

            let mid = event.book.mid_price_ticks();
            let Ok(state) = config.encode(&event.book) else {
                prev = None; // can't bridge a transition across a skipped event
                continue;
            };

            if let Some((prev_state, prev_mid)) = prev {
                self.record(prev_state, state, mid - prev_mid);
            }
            prev = Some((state, mid));
        }
        self.last_observed = prev;
        Ok(())
    }

    /// Merges another counter's accumulated statistics into this one.
    ///
    /// **This is equivalent to serial processing only up to the
    /// transitions spanning each chunk boundary** - two counters built
    /// independently (e.g. on separate threads, each starting fresh with
    /// `last_observed: None`) have no way to know what event immediately
    /// preceded their own first event, so the one transition connecting
    /// the end of one chunk to the start of the next is *not* counted by
    /// either counter and is therefore missing after a merge. For exact
    /// equivalence with serial processing, split chunks so each one
    /// (after the first) starts by repeating the previous chunk's last
    /// event - `observe_events` correctly records that repeated event's
    /// row as a same-state, zero-delta transition contributing nothing
    /// new, while still using it to form the real transition to what
    /// follows. This is a real, disclosed limitation of naive
    /// non-overlapping chunking, not a subtle bug to discover later - see
    /// `tests::merging_non_overlapping_chunks_loses_exactly_the_boundary_transitions`.
    pub fn merge(&mut self, other: &TransitionCounter) -> Result<(), CalibrationError> {
        if self.state_count != other.state_count {
            return Err(CalibrationError::StateCountMismatch {
                a: self.state_count,
                b: other.state_count,
            });
        }
        for i in 0..self.visits.len() {
            self.visits[i] += other.visits[i];
            self.delta_sum[i] += other.delta_sum[i];
        }
        for i in 0..self.counts.len() {
            self.counts[i] += other.counts[i];
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use microprice_core::{
        BookValidationPolicy, ImbalanceBucketing, PriceTicks, Quantity, SpreadBucketing, SymbolId,
        TopOfBook,
    };

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

    fn event(seq: u64, mid: i64, spread: i64, bid_qty: u64, ask_qty: u64) -> BookEvent {
        BookEvent {
            timestamp_ns: seq * 1000,
            sequence: seq,
            symbol: SymbolId(1),
            book: book(
                mid - spread / 2,
                bid_qty,
                mid - spread / 2 + spread,
                ask_qty,
            ),
        }
    }

    fn small_config() -> StateSpaceConfig {
        StateSpaceConfig::new(
            ImbalanceBucketing::new(2).unwrap(),
            SpreadBucketing::new(vec![]).unwrap(), // one catch-all spread bucket
        )
    }

    #[test]
    fn record_accumulates_visits_counts_and_delta_sum() {
        let mut counter = TransitionCounter::new(4);
        counter.record(StateId(0), StateId(1), 0); // non-price-changing -> counts
        counter.record(StateId(0), StateId(1), 0); // non-price-changing -> counts
        counter.record(StateId(0), StateId(2), 1); // price-changing -> NOT in counts

        assert_eq!(counter.visits(StateId(0)), 3);
        assert_eq!(counter.count(StateId(0), StateId(1)), 2);
        assert_eq!(counter.count(StateId(0), StateId(2)), 0);
        assert_eq!(counter.delta_sum(StateId(0)), 1); // 0 + 0 + 1
        assert_eq!(counter.total_observations(), 3);
    }

    #[test]
    fn observe_events_counts_every_consecutive_pair() {
        let config = small_config();
        // 100 bid / 100 ask -> imbalance 0.5 -> bucket 1 (of 2, since
        // bucket_for is floor(0.5*2)=1). 900/100 -> imbalance 0.9 -> bucket 1 too
        // (floor(0.9*2)=1). Use clearly distinct imbalances instead.
        let events = vec![
            event(1, 100, 2, 900, 100), // high imbalance -> bucket 1
            event(2, 100, 2, 100, 900), // low imbalance -> bucket 0; mid unchanged (100->100)
            event(3, 101, 2, 100, 900), // mid moved +1 (price-changing), still bucket 0
        ];
        let mut counter = TransitionCounter::new(config.state_count());
        counter.observe_events(&config, &events).unwrap();

        assert_eq!(counter.total_observations(), 2); // 3 events -> 2 transitions
        let s_high = config.encode(&events[0].book).unwrap();
        let s_low = config.encode(&events[1].book).unwrap();
        // event1 -> event2: mid stayed at 100 -> a real non-price-changing
        // transition, counted.
        assert_eq!(counter.count(s_high, s_low), 1);
        // event2 -> event3: mid moved 100 -> 101, so even though both land
        // in the same bucket (s_low), this must NOT count toward Q -
        // it's captured only in delta_sum.
        assert_eq!(counter.count(s_low, s_low), 0);
        assert_eq!(counter.delta_sum(s_low), 1); // the 100 -> 101 move
    }

    #[test]
    fn observe_events_rejects_out_of_order_sequence() {
        let config = small_config();
        let events = vec![event(5, 100, 2, 500, 500), event(3, 100, 2, 500, 500)];
        let mut counter = TransitionCounter::new(config.state_count());
        let result = counter.observe_events(&config, &events);
        assert_eq!(
            result,
            Err(CalibrationError::OutOfOrderEvent {
                previous_sequence: 5,
                sequence: 3
            })
        );
    }

    #[test]
    fn observe_events_rejects_repeated_sequence() {
        let config = small_config();
        let events = vec![event(5, 100, 2, 500, 500), event(5, 100, 2, 500, 500)];
        let mut counter = TransitionCounter::new(config.state_count());
        assert!(counter.observe_events(&config, &events).is_err());
    }

    #[test]
    fn sequential_calls_on_the_same_counter_bridge_the_call_boundary() {
        let config = small_config();
        let all_events = vec![
            event(1, 100, 2, 900, 100),
            event(2, 100, 2, 100, 900),
            event(3, 101, 2, 100, 900),
            event(4, 100, 2, 900, 100),
        ];

        let mut serial = TransitionCounter::new(config.state_count());
        serial.observe_events(&config, &all_events).unwrap();
        assert_eq!(serial.total_observations(), 3); // 4 events -> 3 transitions

        // Two sequential calls to observe_events *on the same counter*:
        // last_observed carries across the call boundary automatically.
        let mut sequential = TransitionCounter::new(config.state_count());
        sequential
            .observe_events(&config, &all_events[0..2])
            .unwrap();
        sequential
            .observe_events(&config, &all_events[2..4])
            .unwrap();
        assert_eq!(serial.visits, sequential.visits);
        assert_eq!(serial.counts, sequential.counts);
        assert_eq!(serial.delta_sum, sequential.delta_sum);
    }

    #[test]
    fn merging_overlapping_chunks_from_separate_counters_matches_serial() {
        // Independently-built counters (as true parallel processing would
        // produce) *can* still reproduce the serial result exactly - but
        // only if consecutive chunks overlap by the one boundary event, as
        // documented on `TransitionCounter::merge`. events[2] (event 3)
        // appears in both chunks here.
        let config = small_config();
        let all_events = vec![
            event(1, 100, 2, 900, 100),
            event(2, 100, 2, 100, 900),
            event(3, 101, 2, 100, 900),
            event(4, 100, 2, 900, 100),
        ];
        let mut serial = TransitionCounter::new(config.state_count());
        serial.observe_events(&config, &all_events).unwrap();

        let mut chunk_a = TransitionCounter::new(config.state_count());
        chunk_a
            .observe_events(&config, &all_events[0..3]) // events 1,2,3
            .unwrap();
        let mut chunk_b = TransitionCounter::new(config.state_count());
        chunk_b
            .observe_events(&config, &all_events[2..4]) // events 3,4 (event 3 repeated)
            .unwrap();
        chunk_a.merge(&chunk_b).unwrap();

        assert_eq!(serial.visits, chunk_a.visits);
        assert_eq!(serial.counts, chunk_a.counts);
        assert_eq!(serial.delta_sum, chunk_a.delta_sum);
    }

    #[test]
    fn merging_non_overlapping_chunks_loses_exactly_the_boundary_transitions() {
        // The honest failure mode documented on `TransitionCounter::merge`:
        // independently-built counters over non-overlapping chunks each
        // start with last_observed = None, so the transition spanning
        // the chunk boundary (event 2 -> event 3) is never recorded by
        // either counter and is genuinely missing after merge - this test
        // exists so that limitation is verified and quantified, not
        // silently true or silently wrong.
        let config = small_config();
        let all_events = vec![
            event(1, 100, 2, 900, 100),
            event(2, 100, 2, 100, 900),
            event(3, 101, 2, 100, 900),
            event(4, 100, 2, 900, 100),
        ];
        let mut serial = TransitionCounter::new(config.state_count());
        serial.observe_events(&config, &all_events).unwrap();

        let mut chunk_a = TransitionCounter::new(config.state_count());
        chunk_a
            .observe_events(&config, &all_events[0..2]) // events 1,2
            .unwrap();
        let mut chunk_b = TransitionCounter::new(config.state_count());
        chunk_b
            .observe_events(&config, &all_events[2..4]) // events 3,4 - no overlap
            .unwrap();
        chunk_a.merge(&chunk_b).unwrap();

        assert_eq!(serial.total_observations(), 3);
        assert_eq!(chunk_a.total_observations(), 2); // exactly one transition short
    }

    #[test]
    fn merge_rejects_mismatched_state_counts() {
        let mut a = TransitionCounter::new(4);
        let b = TransitionCounter::new(8);
        assert_eq!(
            a.merge(&b),
            Err(CalibrationError::StateCountMismatch { a: 4, b: 8 })
        );
    }
}
