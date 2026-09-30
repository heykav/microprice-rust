//! Streaming transition counting: turns a chronological `BookEvent` stream
//! into per-state visit counts, per-(state,state) transition counts,
//! per-state signed price-delta sums, and the per-state up/down move counts
//! a genuine `P(price goes up)` is computed from — the raw material
//! Phase 7's estimator turns into `Q` and `G1`.
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
    /// Flattened `state_count x state_count`: `pc_counts[i * state_count + j]`
    /// is how many observed transitions went from `i` to `j` **with** a
    /// price change (`delta_ticks != 0`). Disjoint from `counts`, so
    /// `sum_j counts[i][j] + sum_j pc_counts[i][j] == visits[i]`. It does
    /// not enter `Q`, `G1` or `G*`; it exists so the martingale diagnostic
    /// (`crate::diagnostics`) can evaluate the model's own full transition
    /// kernel, including where price-changing moves land.
    pc_counts: Vec<u64>,
    /// `delta_sum[i]`: sum of every observed signed mid-price tick delta
    /// over transitions starting at `i`.
    delta_sum: Vec<i64>,
    /// `up_moves[i]` / `down_moves[i]`: how many transitions starting at `i`
    /// moved the mid-price **up** / **down** by at least one tick.
    ///
    /// `delta_sum` cannot recover this split — `+2, -1` and `+1, 0` sum
    /// alike — which is precisely why a state's `P(price goes up)` was not
    /// computable from this struct before, and why the Brier score was a
    /// disclosed gap rather than a silently wrong number. See
    /// [`TransitionCounter::p_up`].
    ///
    /// `up_moves + down_moves <= visits`: the remainder are the
    /// price-unchanged transitions that populate `counts`.
    up_moves: Vec<u64>,
    down_moves: Vec<u64>,
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
            pc_counts: vec![0; n * n],
            delta_sum: vec![0; n],
            up_moves: vec![0; n],
            down_moves: vec![0; n],
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

    /// How many observed transitions went `from -> to` **with** a
    /// price change (the complement of [`Self::count`]).
    pub fn price_change_count(&self, from: StateId, to: StateId) -> u64 {
        self.pc_counts[from.0 as usize * self.state_count as usize + to.0 as usize]
    }

    pub fn delta_sum(&self, state: StateId) -> i64 {
        self.delta_sum[state.0 as usize]
    }

    /// How many transitions starting at `state` moved the mid-price up.
    pub fn up_moves(&self, state: StateId) -> u64 {
        self.up_moves[state.0 as usize]
    }

    /// How many transitions starting at `state` moved the mid-price down.
    pub fn down_moves(&self, state: StateId) -> u64 {
        self.down_moves[state.0 as usize]
    }

    /// The empirical probability that the mid-price moves **up** on a
    /// transition out of `state`, conditioned on the transitions that
    /// actually moved the price.
    ///
    /// Price-unchanged transitions are excluded from the denominator rather
    /// than counted as "not an up move", matching
    /// `microprice_eval::direction_accuracy`, which likewise excludes
    /// zero-move pairs instead of scoring them as free correct guesses.
    ///
    /// `None` when no directional move has ever been observed from `state` —
    /// an undefined probability, reported as absent rather than invented, in
    /// keeping with `docs/model-spec.md`'s stance on undefined cases.
    ///
    /// This is the honest source of the `P(up)` the Brier score needs. It is
    /// deliberately **not** derived from `G*`, which is a signed
    /// tick-magnitude expectation, not a probability.
    pub fn p_up(&self, state: StateId) -> Option<f64> {
        let i = state.0 as usize;
        let directional = self.up_moves[i] + self.down_moves[i];
        if directional == 0 {
            None
        } else {
            Some(self.up_moves[i] as f64 / directional as f64)
        }
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
    ///
    /// `delta_ticks`'s sign additionally drives `up_moves`/`down_moves`,
    /// which together with `visits` are the only things `p_up` reads.
    pub fn record(&mut self, from: StateId, to: StateId, delta_ticks: i64) {
        let n = self.state_count as usize;
        let from_idx = from.0 as usize;
        self.visits[from_idx] += 1;
        if delta_ticks == 0 {
            self.counts[from_idx * n + to.0 as usize] += 1;
        } else {
            self.pc_counts[from_idx * n + to.0 as usize] += 1;
            if delta_ticks > 0 {
                self.up_moves[from_idx] += 1;
            } else {
                self.down_moves[from_idx] += 1;
            }
        }
        self.delta_sum[from_idx] += delta_ticks;
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
            self.up_moves[i] += other.up_moves[i];
            self.down_moves[i] += other.down_moves[i];
        }
        for i in 0..self.counts.len() {
            self.counts[i] += other.counts[i];
            self.pc_counts[i] += other.pc_counts[i];
        }
        Ok(())
    }

    /// Returns the **mirror-symmetrized** counter: every observed
    /// transition `i -> j` with signed move `d` is also counted as the
    /// transition `sigma(i) -> sigma(j)` with move `-d`, where `sigma` is
    /// [`mirror_state`] (imbalance bucket `b -> N-1-b`, spread bucket
    /// unchanged). See `docs/model-spec.md`, "Imbalance symmetrization",
    /// for the derivation and the guarantees that follow (exact
    /// antisymmetry of `G1` and `G*`).
    ///
    /// `num_imbalance_buckets` is `N`; `state_count` must be a multiple of
    /// it (state ids are `spread_bucket * N + imbalance_bucket`). Every
    /// count is exactly doubled in total (each transition contributes once
    /// as observed and once mirrored), so with a fixed smoothing `alpha`
    /// the pseudo-observations weigh half as much relative to the data as
    /// without symmetrization; this is inherent to pooling and is
    /// documented, not compensated. The result carries no
    /// `last_observed`/`last_sequence` (it is a derived statistic, not a
    /// stream cursor).
    pub fn symmetrized(
        &self,
        num_imbalance_buckets: u32,
    ) -> Result<TransitionCounter, CalibrationError> {
        let n_imb = num_imbalance_buckets;
        if n_imb == 0 || !self.state_count.is_multiple_of(n_imb) {
            return Err(CalibrationError::DimensionMismatch {
                expected: n_imb as usize,
                actual: self.state_count as usize,
            });
        }
        let n = self.state_count as usize;
        let sigma = |s: usize| mirror_state(StateId(s as u32), n_imb).0 as usize;
        let mut out = TransitionCounter::new(self.state_count);
        for i in 0..n {
            let mi = sigma(i);
            out.visits[i] = self.visits[i] + self.visits[mi];
            out.delta_sum[i] = self.delta_sum[i] - self.delta_sum[mi];
            out.up_moves[i] = self.up_moves[i] + self.down_moves[mi];
            out.down_moves[i] = self.down_moves[i] + self.up_moves[mi];
            for j in 0..n {
                let mj = sigma(j);
                out.counts[i * n + j] = self.counts[i * n + j] + self.counts[mi * n + mj];
                out.pc_counts[i * n + j] = self.pc_counts[i * n + j] + self.pc_counts[mi * n + mj];
            }
        }
        Ok(out)
    }
}

/// The mirror map `sigma` on state ids: imbalance bucket `b -> N-1-b`,
/// spread bucket unchanged. An involution (`sigma(sigma(s)) == s`).
/// Corresponds to the market mirror `I -> 1 - I` (swap bid and ask sizes)
/// composed with `price change -> -price change`. Requires
/// `num_imbalance_buckets >= 1`.
pub fn mirror_state(state: StateId, num_imbalance_buckets: u32) -> StateId {
    let n = num_imbalance_buckets;
    let spread_bucket = state.0 / n;
    let b = state.0 % n;
    StateId(spread_bucket * n + (n - 1 - b))
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
        .unwrap()
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
        assert_eq!(serial.up_moves, sequential.up_moves);
        assert_eq!(serial.down_moves, sequential.down_moves);
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
        assert_eq!(serial.up_moves, chunk_a.up_moves);
        assert_eq!(serial.down_moves, chunk_a.down_moves);
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

    #[test]
    fn record_splits_directional_moves_from_flat_ones() {
        let mut counter = TransitionCounter::new(4);
        counter.record(StateId(0), StateId(1), 0); // flat
        counter.record(StateId(0), StateId(1), 2); // up
        counter.record(StateId(0), StateId(1), -1); // down
        counter.record(StateId(0), StateId(1), 3); // up

        assert_eq!(counter.visits(StateId(0)), 4);
        assert_eq!(counter.up_moves(StateId(0)), 2);
        assert_eq!(counter.down_moves(StateId(0)), 1);
        // The flat transition is counted in neither directional bucket, so
        // the directional counts sum to strictly less than visits.
        assert_eq!(
            counter.up_moves(StateId(0)) + counter.down_moves(StateId(0)),
            3
        );
        assert_eq!(counter.delta_sum(StateId(0)), 4); // 0 + 2 - 1 + 3
    }

    #[test]
    fn p_up_is_none_when_no_directional_move_was_ever_observed() {
        let mut counter = TransitionCounter::new(4);
        counter.record(StateId(0), StateId(1), 0);
        counter.record(StateId(0), StateId(1), 0);
        assert_eq!(counter.p_up(StateId(0)), None);
        // A state never visited at all is equally undefined.
        assert_eq!(counter.p_up(StateId(3)), None);
    }

    #[test]
    fn p_up_excludes_flat_transitions_from_its_denominator() {
        let mut counter = TransitionCounter::new(4);
        counter.record(StateId(0), StateId(1), 1); // up
        counter.record(StateId(0), StateId(1), 1); // up
        counter.record(StateId(0), StateId(1), -1); // down
        counter.record(StateId(0), StateId(1), 0); // flat -> must not dilute
        assert_eq!(counter.p_up(StateId(0)), Some(2.0 / 3.0));
    }

    #[test]
    fn p_up_is_per_state_and_reaches_its_bounds() {
        let mut counter = TransitionCounter::new(4);
        counter.record(StateId(0), StateId(1), 1); // state 0: always up
        counter.record(StateId(1), StateId(0), -1); // state 1: always down

        assert_eq!(counter.p_up(StateId(0)), Some(1.0));
        assert_eq!(counter.p_up(StateId(1)), Some(0.0));
        assert_eq!(counter.p_up(StateId(2)), None);
    }

    #[test]
    fn merge_adds_up_and_down_counts_so_p_up_survives_chunking() {
        let mut a = TransitionCounter::new(2);
        let mut b = TransitionCounter::new(2);
        a.record(StateId(0), StateId(1), 1);
        b.record(StateId(0), StateId(1), 1);
        b.record(StateId(0), StateId(1), -1);

        a.merge(&b).unwrap();
        assert_eq!(a.up_moves(StateId(0)), 2);
        assert_eq!(a.down_moves(StateId(0)), 1);
        assert_eq!(a.p_up(StateId(0)), Some(2.0 / 3.0));
    }
}
