# MicroPrice-Rust

A research-grade, performance-oriented implementation of state-conditioned
limit-order-book micro-price estimation, in the queue-imbalance / Markov-chain
tradition associated with Stoikov, Cont, Sirignano and related
queue-reactive work.

**Status: Phase 9 of the roadmap below (model artifacts) — a real, working,
end-to-end calibration → prediction pipeline exists**, though without a
CLI, real-data ingestion, Python bindings, or visualization yet (Phases
10-15). See [`docs/model-spec.md`](docs/model-spec.md) for the precise
mathematical definitions this crate implements, and the
[Roadmap](#roadmap) below for what's next.

## What exists today

**`microprice-core`** — the primitive types an L1 order book needs to be
described unambiguously, plus the V1 state discretization engine.

- `PriceTicks` — integer-tick price representation (no `f32`/`f64` prices
  internally — see the model spec for why).
- `Quantity` — non-negative resting size.
- `TopOfBook` — a validated top-of-book snapshot. Validation (crossed/locked
  market handling) is an explicit, configurable `BookValidationPolicy`
  rather than a hidden default.
- `Imbalance` — queue imbalance `I = Qb / (Qb + Qa)`, with the zero/zero
  degenerate case returning a typed error (`MicroPriceError::EmptyBook`)
  rather than an invented value.
- `SymbolId`, `BookEvent` — an exchange-agnostic instrument identifier and
  a timestamped/sequenced top-of-book observation.
- `StateId`, `ImbalanceBucketing`, `SpreadBucketing`, `StateSpaceConfig` —
  the state discretization engine: uniform imbalance buckets (`O(1)`
  lookup), explicit spread-tick bucket boundaries, packed into one
  contiguous `StateId`. Measured at ~2.5–2.7 ns per encode on the hardware
  in [`docs/benchmarking.md`](docs/benchmarking.md) — real numbers, not a
  target.

**`microprice-data`** — a `MarketDataSource` trait, and a deterministic
synthetic `BookEvent` generator (`SyntheticEventGenerator`) that becomes
the development dataset for every phase after this one. Explicitly **not**
a claim of realistic exchange dynamics (see the module docs and
`docs/model-spec.md`) — its actual job is a fully reproducible (same seed
→ byte-identical output), configurable stream of always-valid events for
CI, examples, and the Phase 6 known-truth solver validation. Configurable
arrival/cancel/market-order/price-move rates, plus an
`imbalance_persistence` parameter (a simple Markov chain on price-move
direction) that is explicitly disclosed as this generator's own modeling
choice, not something derived from a cited paper.

**`microprice-calibration`** — the real calibration pipeline: streaming
transition counting (`TransitionCounter`, event-to-event sampling — see the
model spec), price-changing-event classification (mid-price crossing, with
the *signed tick delta* retained, not just up/down/unchanged), probability
estimation with configurable Laplace smoothing (`estimate`), the
`G* = G1 + Q @ G*` fixed-point solver (`solve` — never a matrix inverse),
and `MicroPriceModel`: a serializable (bincode + a human-readable JSON
metadata sidecar), self-validating-on-load, allocation-free-inference
trained artifact (`predict`/`predict_batch`).

Correctness story, concretely:
- The solver is checked against a **hand-derived toy example**
  (`solver::tests::matches_the_hand_derived_toy_example`): a 2-state chain
  worked out by hand to `G* = [0.2, 0.225]`.
- A **known-truth test** (`tests/known_truth.rs`) samples directly from
  that same toy chain's real probabilities, feeds the samples through the
  actual `TransitionCounter → estimate → solve` pipeline, and confirms the
  estimate converges toward the hand-derived truth as sample size grows
  (measured: |error| ≈ 0.028 at n=1,000 → 0.015 at n=10,000 → 0.004 at
  n=100,000 — a real, logged result, not an assumption).
- A real bug was caught and fixed while building this: `TransitionCounter`
  initially counted *every* observed `state_i → state_j` transition toward
  `Q`, when `Q` is only defined over the *non-price-changing* subset — a
  transition that changes price but happens to land back in the same
  bucket index must not count toward `Q[i][i]`. Caught by a unit test
  whose hand-computed expected values didn't match, not discovered later.
- A second real gap: naive chunked parallel counting (build a separate
  `TransitionCounter` per chunk, merge them) silently drops the one
  transition spanning each chunk boundary, since a fresh counter has no
  way to know what preceded its own first event. This is now an explicit,
  tested, documented property of `merge` (see its doc comment and
  `transitions::tests::merging_non_overlapping_chunks_loses_exactly_the_boundary_transitions`),
  with the correct fix (overlap consecutive chunks by one event) also
  tested directly — not an unverified claim either way.

**`microprice-data`** — a `MarketDataSource` trait, and a deterministic
synthetic `BookEvent` generator (`SyntheticEventGenerator`) that becomes
the development dataset for every phase after this one. Explicitly **not**
a claim of realistic exchange dynamics (see the module docs and
`docs/model-spec.md`) — its actual job is a fully reproducible (same seed
→ byte-identical output), configurable stream of always-valid events for
CI and examples. Configurable arrival/cancel/market-order/price-move
rates, plus an `imbalance_persistence` parameter (a simple Markov chain on
price-move direction) that is explicitly disclosed as this generator's own
modeling choice, not something derived from a cited paper.

Everything else in the workspace (`microprice-eval`, `microprice-cli`)
exists as an empty workspace member so the crate graph is in place, and is
explicitly unimplemented — each crate's `lib.rs`/`main.rs` says so.

## Build and test

```bash
git clone https://github.com/heykav/microprice-rust.git
cd microprice-rust
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all --check
```

89 tests currently: 49 in `microprice-core`, 11 in `microprice-data`, and
29 in `microprice-calibration` (27 unit tests across transition counting,
estimation, smoothing, the solver, and model serialization, plus 2
known-truth integration tests) — including the hand-derived toy-matrix
check, the known-truth convergence test, and the chunk-boundary
merge-correctness tests described above.

```bash
cargo bench -p microprice-core   # see docs/benchmarking.md for the last measured result
```

## Design commitments carried from day one

- `#![forbid(unsafe_code)]` in every crate.
- No heap allocation in any primitive-type calculation.
- Typed errors (`MicroPriceError`, via `thiserror`) — no bare strings, no
  `unwrap()` in library code.
- CI runs `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test`, and
  `cargo doc` (warnings-as-errors) on Linux and macOS for every push.

## Roadmap

This project is being built in the phased order documented in the project
brief, not all at once:

1. ~~Mathematical specification + core primitive types~~ (Phase 1, done)
2. ~~State discretization engine (imbalance/spread bucketing → `StateId`)~~ (Phase 3, done)
3. ~~Synthetic order-book event generator (development dataset)~~ (Phase 4, done)
4. ~~Streaming transition counting~~ (Phase 5, done)
5. ~~Price-movement classification~~ (Phase 6, done — mid-price crossing, signed delta retained)
6. ~~Transition-probability estimation (with configurable smoothing)~~ (Phase 7, done)
7. ~~The micro-price adjustment solver~~ (Phase 8, done — fixed-point iteration)
8. ~~Model artifact serialization~~ (Phase 9, done — bincode + JSON metadata, validated on load)
9. Allocation-free hot-path inference — `MicroPriceModel::predict` already exists; the
   dedicated benchmark + any further optimization is still open
10. Calibration/prediction CLI
11. Chronological out-of-sample evaluation
12. Parquet ingestion
13. Python bindings (PyO3)
14. Visualization (imbalance curves, heatmaps, transition matrices)
15. Profiling-driven optimization (SIMD, parallel calibration, sparse matrices)

No benchmark numbers, accuracy claims, or example predictions will be added
to this README until they come from a real, reproducible run against real
(or explicitly synthetic-and-labeled) data — see `docs/model-spec.md` for
why this project treats that as a hard rule rather than a suggestion.

## License

MIT.
