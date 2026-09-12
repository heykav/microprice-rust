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
  X help" table starts once Phase 16 (profiling-driven optimization)
  begins.
