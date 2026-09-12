# MicroPrice-Rust

A research-grade, performance-oriented implementation of state-conditioned
limit-order-book micro-price estimation, in the queue-imbalance / Markov-chain
tradition associated with Stoikov, Cont, Sirignano and related
queue-reactive work.

**Status: Phase 4 of the roadmap below (synthetic data) — not yet a
working micro-price model.** This README describes what actually exists
right now, not the project's eventual shape. See
[`docs/model-spec.md`](docs/model-spec.md) for the precise mathematical
definitions this crate implements, and the [Roadmap](#roadmap) below for
what's next.

## What exists today

**`microprice-core`** — the primitive types an L1 order book needs to be
described unambiguously, plus the V1 state discretization engine. Still no
calibration/estimation logic (no transition counting, no solver, no
trained model) — that starts in `microprice-calibration`, Phase 5+.

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

Everything else in the workspace (`microprice-calibration`,
`microprice-eval`, `microprice-cli`) exists as an empty workspace member so
the crate graph is in place, and is explicitly unimplemented — each
crate's `lib.rs`/`main.rs` says so.

## Build and test

```bash
git clone https://github.com/heykav/microprice-rust.git
cd microprice-rust
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all --check
```

60 tests currently: 49 in `microprice-core` (valid/crossed/locked books
under every validation policy, one- and both-sided zero depth, extreme
(near-`u64::MAX`) quantities, overflow detection, spread and mid-price
arithmetic, the imbalance degenerate cases, and state-encoding unit tests
plus `proptest` property tests proving `state_id < state_count`,
determinism, and in-range bucket indices across randomly generated
configurations and books) and 11 in `microprice-data` (config validation,
same-seed determinism, different-seed divergence, strictly-increasing
timestamps/sequence numbers, the configured spread always being
maintained, and the both-sides-empty safety net actually firing when
depletion is forced — tested directly, not just avoided).

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
4. Streaming transition counting
5. Price-movement classification
6. Transition-probability estimation (with configurable smoothing)
7. The micro-price adjustment solver
8. Model artifact serialization
9. Allocation-free hot-path inference
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
