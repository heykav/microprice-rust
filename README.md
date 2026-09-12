# MicroPrice-Rust

A research-grade, performance-oriented implementation of state-conditioned
limit-order-book micro-price estimation, in the queue-imbalance / Markov-chain
tradition associated with Stoikov, Cont, Sirignano and related
queue-reactive work.

**Status: Phase 1 (repository bootstrap) — not yet a working micro-price
model.** This README describes what actually exists right now, not the
project's eventual shape. See [`docs/model-spec.md`](docs/model-spec.md) for
the precise mathematical definitions this crate implements, and the
[Roadmap](#roadmap) below for what's next.

## What exists today

`microprice-core` — the primitive types an L1 order book needs to be
described unambiguously, with no calibration or estimation logic yet:

- `PriceTicks` — integer-tick price representation (no `f32`/`f64` prices
  internally — see the model spec for why).
- `Quantity` — non-negative resting size.
- `TopOfBook` — a validated top-of-book snapshot. Validation (crossed/locked
  market handling) is an explicit, configurable `BookValidationPolicy`
  rather than a hidden default.
- `Imbalance` — queue imbalance `I = Qb / (Qb + Qa)`, with the zero/zero
  degenerate case returning a typed error (`MicroPriceError::EmptyBook`)
  rather than an invented value.
- `StateId` — an opaque placeholder for the state-discretization work that
  starts in Phase 3. It is *only* a newtype right now; there is no encoder.

Everything else in the workspace (`microprice-calibration`,
`microprice-data`, `microprice-eval`, `microprice-cli`) exists as an empty
workspace member so the crate graph is in place, and is explicitly
unimplemented — each crate's `lib.rs`/`main.rs` says so.

## Build and test

```bash
git clone https://github.com/heykav/microprice-rust.git
cd microprice-rust
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all --check
```

26 tests currently, all in `microprice-core`: valid/crossed/locked books
under every validation policy, one- and both-sided zero depth, extreme
(near-`u64::MAX`) quantities, overflow detection, spread and mid-price
arithmetic, and the imbalance degenerate cases.

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

1. ~~Mathematical specification + core primitive types~~ (this repository, now)
2. State discretization engine (imbalance/spread bucketing → `StateId`)
3. Synthetic order-book event generator (development dataset)
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
