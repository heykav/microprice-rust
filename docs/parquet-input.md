# Parquet Level-1 input

`microprice evaluate-parquet` runs the same calibration and out-of-sample
evaluation as `evaluate-csv` (they share one implementation after ingestion),
reading Level-1 quotes from a Parquet file. What existed before: a
library-level reader/writer in `microprice-data` (`read_events_from_parquet`,
`write_events_to_parquet`, feature `parquet-ingestion`); the CLI could not read
Parquet. The subcommand is the only addition.

## Enabling it (default build stays lean)

```bash
cargo build --release -p microprice-cli --features parquet
cargo run --release -p microprice-cli --features parquet -- evaluate-parquet \
    --input quotes.parquet --resolution 2 --report-md report.md
```

The `parquet` feature of `microprice-cli` forwards to
`microprice-data/parquet-ingestion`. Without it, `arrow` and `parquet` are not
compiled or even resolved into the CLI's dependency tree (checked with
`cargo tree -p microprice-cli`), and `microprice evaluate-parquet ...` exits
with a message saying to rebuild with `--features parquet` (tested).
`cargo test --workspace --all-features`, as CI runs it, includes the feature.

## Schema

This is this project's own column convention, not an exchange format. One row
per quote update, **in time order**:

| column            | Parquet/Arrow type | meaning |
|-------------------|--------------------|---------|
| `timestamp_ns`    | UInt64, non-null   | event time, nanoseconds since any epoch; must be non-decreasing |
| `sequence`        | UInt64, non-null   | must be **strictly increasing** row to row |
| `symbol_id`       | UInt32, non-null   | read but ignored by the CLI (`--symbol-id` sets the model's) |
| `bid_price_ticks` | Int64, non-null    | best bid, integer **model units** (see below) |
| `bid_qty`         | UInt64, non-null   | size at the best bid |
| `ask_price_ticks` | Int64, non-null    | best ask, integer model units |
| `ask_qty`         | UInt64, non-null   | size at the best ask |

Column types are strict: a column with another type (say Int32 or Float64) is
an error, not converted. Extra columns are ignored. Column order is irrelevant.

**Units.** Prices are integers, not currency amounts. `--resolution R` (default
2) declares that `R` integer units make one exchange tick, exactly as for CSV
(spread bucket bounds are given in ticks and multiplied by `R`; all reported
errors are divided by `R`). Write the file in half-tick units (`R = 2`) so
that `bid + ask` is always even and mids are exact; the CLI warns when
`--resolution 1` is used with odd sums. `--tick-size` is optional and only
printed in the report header; nothing is checked against it.

## Validation and errors

* Every book goes through the strict policy (`Pa > Pb`, not both sizes zero).
  The first invalid row aborts the read with its row index. There is no
  skip mode (unlike `evaluate-csv --skip-invalid-rows`); clean the file first.
* Non-increasing `sequence` or decreasing `timestamp_ns` aborts with the row
  number (0-based index of the offending row).
* Missing file, non-Parquet file, missing or mistyped column: a
  `Parquet I/O error` naming the problem.
* Fewer than 100 rows: too few to calibrate and evaluate.
* `--max-events N` keeps the first N rows (a chronological prefix; stated in
  the report). `--drop-unchanged` is not offered for Parquet.

## Options shared with `evaluate-csv`

Split, buckets, smoothing, horizons (including the additional wall-clock
horizons), `--symmetrize` (exploratory, labelled as such), bootstrap, seed and
report options are identical; see `microprice evaluate-parquet --help` and
[`real-data-evaluation.md`](real-data-evaluation.md), whose pre-registered
protocol applies unchanged.

## Tests

`crates/microprice-cli/tests/evaluate_parquet.rs` generates its tiny fixtures
in the test itself from the synthetic generator (nothing is checked in):
end-to-end report, parity with `evaluate-csv` on identical events,
`--max-events`, and each error path above. In a default-feature build the
same file tests the "rebuild with `--features parquet`" message. These are
plumbing tests; their numbers mean nothing about real markets.
