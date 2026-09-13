# Benchmarking

Every number in this document was actually measured by running
`cargo bench` on the hardware/software listed with it, on the commit noted
below. Nothing here is an estimate, a target, or "should be roughly X" —
per the project's own engineering philosophy, a benchmark number that
wasn't actually produced by running the benchmark doesn't belong in this
repository.

## How to reproduce

```bash
cargo bench -p microprice-core
cargo bench -p microprice-calibration
```

Criterion writes full HTML reports (including outlier analysis) to
`target/criterion/`.

## State encoding (Phase 3 / Prompt 2)

**Measured:** 2026-09-13, commit range starting at `897331f` (Phase 1).

**Hardware:** Apple M3 Pro, macOS 26.6.2, arm64.
**Toolchain:** `rustc 1.98.1 (48a229cea 2026-09-01)`, release profile
(`cargo bench` builds with optimizations).
**Benchmark:** `crates/microprice-core/benches/state_encoding.rs`, using
[Criterion.rs](https://github.com/bheisler/criterion.rs). Configuration:
20 imbalance buckets × 4 spread buckets (80 states — the project brief's
own worked example), single fixed `TopOfBook`.

| Benchmark | Result |
|---|---|
| `state_encode_single` (one `StateSpaceConfig::encode` call) | 2.53–2.69 ns/iter |
| `state_encode_batch_1000` (1000 calls in a tight loop) | 2.2608 µs total (2.26 ns/call) |

That's on the order of **370–440 million encodings/second** on this single
machine for this configuration — comfortably past the "tens of
nanoseconds" target and the "millions of state encodings per second" goal,
though "comfortably past" is a description of this one measurement, not a
guarantee about other hardware, other bucket-count configurations, or
future code changes. Re-run the benchmark rather than trusting this table
if either changes.

**Caveats, stated rather than glossed over:**
- This measures encoding *alone* — constructing a `TopOfBook` and computing
  `Imbalance`/`SpreadTicks` are included (they're inside `encode`), but
  there is no calibrated model to apply an adjustment yet (that's Phase 5+
  work in `microprice-calibration`), so this is not yet an end-to-end
  micro-price inference benchmark.
- `criterion`'s outlier detection flagged 12% of `state_encode_single`
  samples as outliers (9 low-mild, 2 high-mild, 1 high-severe) — normal for
  a benchmark this fast, where system noise (scheduler preemption, thermal
  throttling) is a larger fraction of the measured time than for a slower
  operation, not a sign of a slow-path/fast-path split in the code itself.
- No comparison against a naive/unoptimized baseline exists yet, since
  there's only one implementation. A meaningful "how much did optimization
  X help" table starts once Phase 15 (profiling-driven optimization)
  begins.

## Calibration pipeline (Phase 15 profiling — the actual basis for "no
optimization needed yet")

**Measured:** 2026-09-13, commit `da6f9378e2085cd193d0762cb7f45d9eb3c52280`.

**Hardware/toolchain:** same machine as above (Apple M3 Pro, macOS 26.6.2,
`rustc 1.98.1`).
**Benchmark:** `crates/microprice-calibration/benches/calibration_pipeline.rs`.
Configuration: 20 imbalance buckets × 4 spread buckets (80 states), a
100,000-event synthetic stream (seed 42, the same generator config used
throughout this project's examples).

| Benchmark | Result |
|---|---|
| `observe_events/100k_events` (streaming transition counting over all 100,000 events, fresh `TransitionCounter` per iteration) | 516.16–517.26 µs total (~5.17 ns/event) |
| `estimate_80_states` (`Q`/`G1` from the fully-populated counter above) | 6.69–6.73 µs |
| `solve_80_states` (Jacobi fixed-point solve, default tolerance `1e-10`) | 511.34–512.17 µs |

**The actual profiling-driven conclusion, per Phase 15's own explicit
requirement to profile *before* optimizing:** a full calibration run over
100,000 events at this state-space size costs roughly **1.04 ms total**
(counting + estimation + solving combined) on this machine — negligible
next to the ~500 ms it takes just to *generate* that many synthetic events
in the first place (see `microprice train`'s own timing output). Nothing
here is a bottleneck at V1's intended state-space scale (tens to low
hundreds of states, as `TransitionCounter`'s own module docs already say).
**No SIMD, parallelism, or sparse-matrix work has been added** — doing so
without a real bottleneck to justify it would be exactly the premature
optimization Phase 15's own instructions warn against.

`solve_80_states`' ~512 µs is consistent with roughly 80 Jacobi iterations
to reach `1e-10` tolerance on this data's actual estimated `Q` matrix (an
`O(state_count²)` `row_dot` sweep per iteration, ~80² = 6,400 multiply-adds
per sweep) — normal linear convergence for a well-conditioned
sub-stochastic matrix, not a sign of a slow solver.

**When this would actually need revisiting:** the counting step is
`O(events)` and the estimate/solve steps are `O(state_count²)` per
iteration — a state space of thousands of states (not this project's
current tens-to-hundreds) is where a sparse `Q` representation would start
to matter, and is exactly the documented, unimplemented extension
`transitions.rs`'s own module docs already call out. That's a future
decision to make *if and when* the state space actually grows that large,
backed by a new benchmark at that scale — not something to build now on
spec.
