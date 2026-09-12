# MicroPrice-Rust

A research-grade, performance-oriented implementation of state-conditioned
limit-order-book micro-price estimation, in the queue-imbalance / Markov-chain
tradition associated with Stoikov, Cont, Sirignano and related
queue-reactive work.

**Status: Phase 13 of the roadmap below (Python bindings) — a real,
working, end-to-end train → predict → inspect → benchmark → evaluate
pipeline exists, reads/writes real Parquet files, and is now also usable
from Python** via `microprice-python` (PyO3), though still without
visualization (Phases 14-15). See
[`docs/model-spec.md`](docs/model-spec.md) for the precise mathematical
definitions this crate implements, and the [Roadmap](#roadmap) below for
what's next.

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

**`microprice-data`** — a `MarketDataSource` trait, a deterministic
synthetic `BookEvent` generator (`SyntheticEventGenerator`) that becomes
the development dataset for every phase after this one, and (behind the
`parquet-ingestion` Cargo feature, **off by default**) real Parquet
ingestion. Explicitly **not** a claim of realistic exchange dynamics (see
the module docs and `docs/model-spec.md`) — its actual job is a fully
reproducible (same seed → byte-identical output), configurable stream of
always-valid events for CI, examples, and the Phase 6 known-truth solver
validation. Configurable arrival/cancel/market-order/price-move rates, plus
an `imbalance_persistence` parameter (a simple Markov chain on price-move
direction) that is explicitly disclosed as this generator's own modeling
choice, not something derived from a cited paper.

`parquet-ingestion` (`write_events_to_parquet`/`read_events_from_parquet`)
is off by default because `arrow`+`parquet` pull in a genuinely large
dependency tree (~50 crates) that only matters to callers actually reading
real Parquet files — the synthetic generator, and CI's default test job,
shouldn't pay that build-time cost. Reading validates every row through
the same `TopOfBook::new` path everything else uses (a row that fails,
e.g. a crossed book, aborts the read with a typed error naming the row
index, rather than being silently dropped) — this is a data *ingestion*
boundary, where a malformed row is worth stopping for, unlike the
Phase 5 transition counter's deliberate skip-on-encode-failure behavior at
the state-encoding boundary. Verified with a real 250-row write → read
round-trip test (exact equality) and a hand-built-bad-bytes test proving
the row-index reporting is accurate, not just plausible-looking — see
`microprice-data/src/parquet.rs`'s tests.

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

**`microprice-eval`** — chronological out-of-sample evaluation:
`chronological_split` (a train/test split that respects time order —
**never** a random shuffle, which would leak future information into
training), `evaluate` (runs a calibrated model against a held-out
chronological stream at a configurable horizon and reports MAE, signed
bias, and directional accuracy against the `mid`/`weighted_mid` baselines
the model spec already defines), and `metrics` (the plain error functions
underneath). A Brier score is **not** implemented here — it needs a
probabilistic `P(up)` prediction, and `TransitionCounter` only ever
accumulates a signed delta *sum* per state, never separate up/down
transition counts, so there's nothing honest to compute one from without
inventing data the calibration pipeline doesn't collect. That's recorded
as a disclosed gap (see the module's own doc comment), not silently
skipped.

**`microprice-cli`** — the `microprice` binary: `train` (generate synthetic
data, run the full counting → estimation → solving pipeline, save a model
artifact), `predict` (load a model, predict one book's micro-price),
`inspect` (load a model, print its metadata plus a summary of what was
*actually* calibrated — g_star range, per-state visit counts, how many
states have zero real observations), `benchmark` (measure real
`predict`/`predict_batch` throughput on the machine it's run on, printed
with an explicit note that hardware/toolchain aren't auto-captured the way
`docs/benchmarking.md`'s Criterion numbers are), and `evaluate` (generate
synthetic data, split it chronologically, calibrate only on the train
side, and report real out-of-sample MAE/bias/direction-accuracy against
the held-out test side — see below for what this actually measured on
synthetic data, reported honestly rather than cherry-picked). `train`'s
only data source today is `microprice-data`'s synthetic generator — the
CLI says so in its own output, not just in this README. `microprice-data`
can also read/write real Parquet files as of Phase 12 (see below), but
that isn't wired into a CLI flag yet — only used directly as a library
today.

A real run (`microprice evaluate --num-events 300000
--num-imbalance-buckets 10 --spread-bucket-bounds "1,2,4"`, seed 42,
horizon 1, 90,000 held-out events) measured **microprice MAE 0.1094 ticks
vs. naive-mid MAE 0.0988 ticks — microprice did *not* beat the naive mid
baseline on this synthetic dataset at this horizon**, and direction
accuracy was 0.5147 (barely above the 0.5 coin-flip floor). This is
reported as-is, not hidden: this generator's `imbalance_persistence` drives
price-move direction as its own Markov chain independent of queue
imbalance (see `microprice-data`'s module docs), so there is no strong
imbalance → future-price-direction signal in this *particular* synthetic
dataset for the model to find — a properly negative result about this
synthetic data's structure, not (yet) evidence about real order-book data.

**`microprice-python`** — PyO3 bindings exposing `MicroPriceModel`
(`load`/`save`/`predict`/`metadata`) and `train_synthetic` (the same
counting → estimation → solving pipeline as `microprice train`) to Python.
**Deliberately its own Cargo workspace**, not a member of the root one: a
PyO3 extension-module `cdylib` can't be built/tested by plain `cargo
build`/`cargo test` the way every other crate here is (it expects to be
loaded into a running Python process, not linked against `libpython`
directly) — it's built and verified with `maturin` instead, with its own
dedicated CI job (`python-bindings`) running a real
`maturin build` → `pip install` → import-and-exercise smoke test rather
than being silently uncovered. See
[`docs/python-bindings.md`](docs/python-bindings.md) for exact setup
steps and a real measured session (on Python 3.12): training and
predicting from Python reproduced **the same numbers** the Rust CLI
produces on identical inputs (it's the same Rust core underneath, not a
reimplementation), and both error paths (a crossed book, a missing model
file) were confirmed to raise clean Python `ValueError`s rather than
panicking or segfaulting. Not yet wired up: Parquet ingestion from
Python, and batch prediction (`predict_batch`) — see that doc's "What
isn't wired up yet" section.

## Build and test

```bash
git clone https://github.com/heykav/microprice-rust.git
cd microprice-rust
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all --check
```

105 tests currently (run with `--all-features` — CI's `test` job does, so
the Parquet tests below actually execute, not just type-check): 49 in
`microprice-core`, 14 in `microprice-data` (11 for the synthetic
generator, plus 3 behind `parquet-ingestion`: a 250-row write→read
round-trip, a hand-built-bad-bytes row-validation test, and a
nonexistent-file test), 29 in `microprice-calibration` (27 unit tests
across transition counting, estimation, smoothing, the solver, and model
serialization, plus 2 known-truth integration tests), and 13 in
`microprice-eval` (splitting, metrics, and end-to-end evaluation,
including a hand-computed-by-hand MAE/bias/direction-accuracy check and a
test proving an unencodable book is skipped rather than aborting the whole
evaluation) — including the hand-derived toy-matrix check, the
known-truth convergence test, and the chunk-boundary merge-correctness
tests described above. `microprice-cli` is a binary crate (no unit tests
of its own); it was verified by actually running
`train`/`predict`/`inspect`/`benchmark`/`evaluate` end to end, including
the error paths (a nonexistent model path, a crossed book, malformed
spread bucket bounds), and confirming none of them panic — see the commit
history for the exact commands and output.

```bash
cargo bench -p microprice-core   # see docs/benchmarking.md for the last measured result

# End-to-end, against synthetic data (the CLI's only wired-up data source):
cargo run -p microprice-cli -- train --output /tmp/model.bin --num-events 500000
cargo run -p microprice-cli -- inspect --model /tmp/model.bin
cargo run -p microprice-cli -- predict --model /tmp/model.bin \
    --bid-price-ticks 10000 --bid-qty 500 --ask-price-ticks 10002 --ask-qty 500
cargo run -p microprice-cli -- benchmark --model /tmp/model.bin
cargo run -p microprice-cli -- evaluate --num-events 300000 \
    --num-imbalance-buckets 10 --spread-bucket-bounds "1,2,4"

# Parquet ingestion (a library function, not yet a CLI flag - feature-gated,
# see microprice-data/src/parquet.rs):
cargo test -p microprice-data --features parquet-ingestion

# Python bindings (their own Cargo workspace - see docs/python-bindings.md):
cd crates/microprice-python && maturin develop
```

## Design commitments carried from day one

- `#![forbid(unsafe_code)]` in every crate.
- No heap allocation in any primitive-type calculation.
- Typed errors (`MicroPriceError`, via `thiserror`) — no bare strings, no
  `unwrap()` in library code.
- CI runs `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test
  --all-features`, and `cargo doc` (warnings-as-errors) on Linux and
  macOS for every push, plus a dedicated `python-bindings` job that
  builds `microprice-python` with `maturin` and runs a real Python
  import/exercise smoke test (see `docs/python-bindings.md` for why that
  crate needs its own job rather than being covered by `cargo test
  --workspace`).

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
9. ~~Allocation-free hot-path inference~~ (Phase 9-10, done — `predict`/
   `predict_batch` exist and are exercised by `microprice benchmark`,
   which measures real, on-machine ns/predict — see the README's "Build
   and test" section for the exact command; no fabricated numbers)
10. ~~Calibration/prediction CLI~~ (Phase 10, done — `microprice
    train`/`predict`/`inspect`/`benchmark`)
11. ~~Chronological out-of-sample evaluation~~ (Phase 11, done —
    `microprice-eval`'s `chronological_split`/`evaluate`, wired into
    `microprice evaluate`; see above for a real measured result and its
    honest interpretation, and the module docs for why a Brier score is a
    disclosed gap rather than a fabricated one)
12. ~~Parquet ingestion~~ (Phase 12, done — `microprice-data`'s
    `parquet-ingestion` feature; a 250-row write→read round trip and a
    hand-built-bad-bytes row-validation test both pass, feature off by
    default to keep the base crate's dependency tree small)
13. ~~Python bindings (PyO3)~~ (Phase 13, done — `microprice-python`;
    see above for a real cross-checked session and
    [`docs/python-bindings.md`](docs/python-bindings.md) for why it's its
    own Cargo workspace, verified via `maturin` and a dedicated CI job
    rather than the root `cargo test --workspace`)
14. Visualization (imbalance curves, heatmaps, transition matrices)
15. Profiling-driven optimization (SIMD, parallel calibration, sparse matrices)

No benchmark numbers, accuracy claims, or example predictions will be added
to this README until they come from a real, reproducible run against real
(or explicitly synthetic-and-labeled) data — see `docs/model-spec.md` for
why this project treats that as a hard rule rather than a suggestion.

## License

MIT.
