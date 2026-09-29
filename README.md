# MicroPrice-Rust

Queue-imbalance micro-price estimation for limit order books, in Rust. Given
a stream of best bid / best ask quotes (price and size), it calibrates a
Markov-chain model over discretized (imbalance, spread) states and outputs
`mid + G*(state)`, an estimate of the expected future mid-price, following
the construction in Stoikov (2018) with the departures listed
[below](#departures-from-the-paper).

It is aimed at people who want an auditable, reproducible implementation to
study or extend, not a trading signal.

**Status: research code, no real-data result yet.** On the project's own
synthetic data the calibrated micro-price does **not** beat the naive
mid-price (MAE 0.1094 vs 0.0988 ticks, horizon 1, 90,000 held-out events;
[details](#what-was-measured-on-synthetic-data)). The synthetic generator has
no imbalance-to-direction signal, so that says little about real markets,
and no real-data number exists yet ([why](#real-data-result)). The CSV
ingestion for real Level-1 data and a pre-registered evaluation protocol
([`docs/real-data-evaluation.md`](docs/real-data-evaluation.md)) are in
place, waiting on a run with network access to a dataset host. The Binance
and LOBSTER column mappings are unverified against vendor documentation.

## Quickstart

Requires a recent stable Rust toolchain. On Linux the CLI needs the system
fontconfig development files (used by `plotters`' text rendering):
`sudo apt-get install libfontconfig1-dev`.

```bash
git clone https://github.com/heykav/microprice-rust.git
cd microprice-rust

# Train on the deterministic synthetic generator, inspect, predict one book.
cargo run --release -p microprice-cli -- train --output /tmp/model.bin --num-events 500000
cargo run --release -p microprice-cli -- inspect --model /tmp/model.bin
cargo run --release -p microprice-cli -- predict --model /tmp/model.bin \
    --bid-price-ticks 10000 --bid-qty 500 --ask-price-ticks 10002 --ask-qty 500

# Out-of-sample evaluation on a chronological split (synthetic data).
cargo run --release -p microprice-cli -- evaluate --num-events 300000 \
    --num-imbalance-buckets 10 --spread-bucket-bounds "1,2,4"

# Library-level example: synthetic -> CSV -> ingest -> calibrate -> compare.
cargo run --release -p microprice-eval --example synthetic_then_csv
```

To run on your own Level-1 quotes (any CSV with timestamp, bid price/size,
ask price/size), see [Real-data result](#real-data-result) and
[`docs/real-data-evaluation.md`](docs/real-data-evaluation.md).

**Live demo:** [Micro-Price Terminal](https://heykav.github.io/microprice-rust/)
is an interactive console over an embedded, pre-trained model (static; see
[`site/README.md`](site/README.md) for exactly what that means).

## Real-data result

**Result pending. There is no real-data number in this repository.**
Everything measured so far used the synthetic generator (see the negative
result in the CLI section below) or tiny hand-written format fixtures.

What exists: an ingestion path for real Level-1 quote CSVs (a generic named
column schema plus Binance USD-M futures `bookTicker` and LOBSTER presets), a
`microprice evaluate-csv` command, and an evaluation protocol written before
any result -
[`docs/real-data-evaluation.md`](docs/real-data-evaluation.md) - with a
chronological split, naive-mid and size-weighted-mid baselines, MAE/MSE and
directional accuracy at fixed event horizons, bootstrap intervals, a
pre-registered decision rule, and a commitment to publish negative results.
The Binance and LOBSTER column mappings are **UNVERIFIED** against the
vendors' documentation.

Why it is pending: the environment this was built in could not reach the
dataset hosts (`data.binance.vision`, `lobsterdata.com`). Producing the result
needs network access to one of them. The exact command, on a machine with
normal internet access:

```bash
TICK_SIZE=0.1 scripts/fetch_binance_bookticker.sh BTCUSDT 2024-01-15
```

(`TICK_SIZE` is the symbol's price tick; 0.1 for BTCUSDT is itself unverified
here.) It writes `results/binance-bookticker-BTCUSDT-2024-01-15.md`. Downloaded
data is not committed.

## What was measured on synthetic data

`microprice evaluate --num-events 300000 --num-imbalance-buckets 10
--spread-bucket-bounds "1,2,4"` (seed 42, horizon 1 event, 90,000 held-out
events) gave **micro-price MAE 0.1094 ticks versus naive-mid MAE 0.0988
ticks**: the micro-price did not beat the naive mid. Direction accuracy was
0.5147, barely above a coin flip. The generator's `imbalance_persistence`
drives price-move direction as its own Markov chain, independent of queue
imbalance, so there is no imbalance signal for the model to find. This is a
negative result about that synthetic dataset, not evidence about order books.
The Brier score against a climatological baseline shows no skill either
(about -0.0006). The generator is deterministic (same seed, identical
output) and exists for tests and examples, not realism.

## How it works

1. **State.** Best-bid/ask sizes give imbalance `I = Qb / (Qb + Qa)`, split
   into uniform buckets; the spread in ticks is bucketed by explicit bounds.
   Both pack into one `StateId`. Prices are integer ticks, never floats.
2. **Counting.** Consecutive quote events form transitions `state_i ->
   state_j`. A transition is price-changing if the mid-price differs
   (signed tick delta retained); only non-price-changing ones enter `Q`.
3. **Estimation.** Laplace-smoothed `Q`, one-step expected mid change `G1`,
   and per-state `P(up)` (used only for the Brier score, never in `G*`).
4. **Solve.** `G* = G1 + Q G*` by fixed-point iteration (no matrix inverse);
   non-convergence is a returned error.
5. **Predict.** `micro-price = mid + G*[state]`, allocation-free.

Exact definitions and degenerate cases: [`docs/model-spec.md`](docs/model-spec.md).

## Departures from the paper

Reference: Stoikov, S. (2018), "The Micro-Price: A High-Frequency Estimator
of Future Prices", *Quantitative Finance* 18(12). (Volume/issue as recalled;
page range and DOI not checked because the publisher was unreachable during
this work - UNVERIFIED. Nor was the paper's text available, so the list below
reflects the author's understanding of its structure and should be checked
against it.)

This implementation is *in the tradition of* that construction, not a
replication. Known differences:

- **No imbalance symmetrization.** The paper's setting has a natural
  symmetry (mirror the imbalance and flip the direction of moves). Here every
  state is estimated independently; the symmetry is neither imposed nor
  tested. Estimates are noisier and the surface is not guaranteed
  antisymmetric.
- **No explicit martingale check.** The construction is motivated by the
  micro-price being a martingale. This code solves the fixed point but does
  not test that property on data.
- **Event-to-event sampling.** Every consecutive quote event is a
  transition, including size-only updates and (with some feeds) no-op rows.
  Whether that matches the paper's time scale is unverified. Fixed-time and
  N-event sampling are not implemented.
- **Own discretization and smoothing.** Uniform imbalance buckets, explicit
  spread bounds, and additive Laplace smoothing (with a zero-mean prior on
  `G1`) are this project's choices, not the paper's. With `alpha = 0`,
  calibration fails if any state is unvisited.
- **Truncating integer mid in `microprice-core`.** `TopOfBook::mid_price_ticks`
  truncates `(bid + ask) / 2`, so with an odd bid+ask sum the model sees a
  mid off by half a tick. The CSV path avoids this by using half-tick model
  units (`--resolution 2`); direct library users with odd spreads should do
  the same or be aware of it.
- **Multi-tick moves are kept** (signed delta), rather than assuming one-tick
  moves.

## Workspace

| crate | contents |
|---|---|
| `microprice-core` | integer-tick prices, quantities, validated `TopOfBook` (crossed/locked policy is explicit), imbalance, state discretization |
| `microprice-data` | `MarketDataSource`, deterministic synthetic generator, Level-1 CSV ingestion (generic, Binance `bookTicker`, LOBSTER), optional Parquet (`parquet-ingestion` feature, off by default) |
| `microprice-calibration` | transition counting, estimation, solver, serializable `MicroPriceModel` (bincode + JSON sidecar, validated on load) |
| `microprice-eval` | chronological split, metrics, `compare_predictors` (paired baselines with block-bootstrap intervals) |
| `microprice-cli` | `microprice` binary: `train`, `predict`, `inspect`, `benchmark`, `evaluate`, `evaluate-csv`, `visualize` |
| `microprice-python` | PyO3 bindings (`load`/`save`/`predict`/`metadata`, `train_synthetic`); its own Cargo workspace, built with `maturin`, see [`docs/python-bindings.md`](docs/python-bindings.md). No `predict_batch` or Parquet/CSV ingestion from Python yet. |

`microprice-cli` `train` and `evaluate` use the synthetic generator only;
`evaluate-csv` is the real-data entry point. Parquet reading is a library
function, not a CLI flag.

## Correctness evidence

- The solver matches a hand-derived 2-state example (`G* = [0.2, 0.225]`).
- A known-truth test samples from that chain, runs the real
  `TransitionCounter -> estimate -> solve` pipeline, and converges toward
  the truth as samples grow (|error| about 0.028, 0.015, 0.004 at n = 1e3,
  1e4, 1e5).
- Two bugs found while building it are pinned by tests: same-bucket
  price-changing transitions must not count toward `Q[i][i]`, and merging
  non-overlapping chunk counters drops exactly the boundary transitions
  (documented on `merge`, with the overlap fix tested).
- `#![forbid(unsafe_code)]` in every crate; typed errors, no `unwrap()` in
  library code.
- Performance numbers, measured with Criterion on stated hardware, are in
  [`docs/benchmarking.md`](docs/benchmarking.md). Profiling concluded that
  SIMD, parallelism or sparse matrices are not justified at the intended
  state-space size; nothing else about speed is claimed here.

## Build and test

```bash
cargo build --workspace
cargo test --workspace --all-features      # the test count changes; run it
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all --check
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
cargo bench -p microprice-core             # optional; see docs/benchmarking.md
```

CI runs these on Linux and macOS plus a `maturin` build and smoke test of the
Python bindings. The minimum supported Rust version has not been measured;
CI uses current stable. See [`CONTRIBUTING.md`](CONTRIBUTING.md) and
[`CHANGELOG.md`](CHANGELOG.md).

## Component status

All of the following exist and are tested: core types and state encoding;
synthetic generator; transition counting; estimation and smoothing; solver;
model serialization; CLI (`train`, `predict`, `inspect`, `benchmark`,
`evaluate`, `visualize`); chronological evaluation; Parquet ingestion
(library, feature-gated); Python bindings; static PNG visualization (three
plots via `plotters`); CSV ingestion and `evaluate-csv`; profiling.

Not done: a real-data result; wall-clock horizons; imbalance symmetrization;
a martingale diagnostic; `predict_batch` and ingestion in the Python bindings;
a measured MSRV.

No benchmark numbers, accuracy claims or example predictions are added to this
README unless they come from a reproducible run on real or explicitly
synthetic-and-labelled data.

## License

MIT, see [`LICENSE`](LICENSE).
