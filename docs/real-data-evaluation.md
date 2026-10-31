# Real-data evaluation protocol

**Status: one real-data result exists** (Binance USD-M BTCUSDT, 2024-01-15;
see the README's "Real-data result" and the run log at the end of this
document). This document was written, and committed, before any real-data
number was produced by this project, and the protocol below was not changed
afterwards. The result is negative: the calibrated micro-price is worse than
the naive mid at the primary comparison. Historical note: when the protocol
was written the development environment could not reach the dataset hosts, so
the only data the code had seen was the synthetic generator's and tiny
hand-written format fixtures.

The protocol below is fixed in advance so the eventual result cannot be
shaped by what it turns out to be.

## Commitments

1. **Negative results are reported.** If the calibrated micro-price is not
   distinguishably better than the naive mid, or is worse, that is the
   result, and it goes in the README's "Real-data result" section in the same
   words the report generator used. The synthetic-data result already is
   negative and is kept in the README.
2. The primary comparison, horizon, split, model configuration and decision
   rule below are **pre-specified**. Anything else that is run afterwards is
   labelled exploratory.
3. Downloaded vendor data is never committed (licensing). Only the
   generated report and the exact command are.

## Amendments made before any real-data result exists

Dated 2026-09-29. **No real-data result existed when these were made, and
none was looked at**; the project has still never seen real market data.
The default predictors, the pre-specified configuration table, the split, the
primary comparison and the decision rule above are **unchanged**. Everything
listed here is an additional, separately labelled option that is off by
default and is not used by the pre-registered run.

1. **`--symmetrize` (exploratory).** Optional imbalance-mirror pooling in
   calibration (`docs/model-spec.md`, "Imbalance symmetrization"). A report
   produced with it says so in its "Split and model" section and prefixes its
   decision section with a warning that the verdicts are exploratory and are
   not the pre-registered result. Should it be run, all such runs are
   reported alongside the pre-registered one, per commitment 2.
2. **Model diagnostics in the report.** The report now prints the
   antisymmetry residual and the martingale diagnostic
   (`docs/model-spec.md`, "Martingale diagnostic") of the calibrated model on
   the *training* split. They are descriptive, use no test data and take no
   part in the decision rule.
3. **Wall-clock horizons (additional).** `--wall-clock-horizons-ms` adds
   horizons measured in milliseconds, defined by the rule below. They are
   reported in their own sections, labelled "additional, NOT pre-registered",
   and are never the primary or secondary comparison. The event horizons
   `1, 10, 100` and the primary horizon 10 are unchanged.
4. **Parquet input path.** `evaluate-parquet` (feature `parquet`,
   `docs/parquet-input.md`) reads the same kind of Level-1 data from Parquet
   and runs the identical post-ingestion pipeline and decision rule. It
   changes no protocol setting; the Binance/LOBSTER data of this document is
   still ingested through CSV.

## Data

### Binance USD-M futures `bookTicker` (public, no credentials)

Daily files under
`https://data.binance.vision/data/futures/um/daily/bookTicker/<SYMBOL>/<SYMBOL>-bookTicker-<YYYY-MM-DD>.zip`
(plus a `.CHECKSUM` file next to each). Columns, as expected by the
`binance-bookticker` preset:

| column | use |
|---|---|
| `update_id` | ignored (row order defines sequence) |
| `best_bid_price`, `best_bid_qty` | best bid |
| `best_ask_price`, `best_ask_qty` | best ask |
| `transaction_time` | ignored |
| `event_time` | timestamp, Unix milliseconds |

### LOBSTER free sample (level 1 of the `orderbook` file)

| `orderbook` column (0-based) | use |
|---|---|
| 0, 1 | ask price, ask size (level 1) |
| 2, 3 | bid price, bid size (level 1) |

Prices are dollars x 10,000; empty levels use sentinel prices (about
+/-9,999,999,999) and are treated as invalid rows. Timestamps are the first
column of the paired `message` file (seconds after midnight, fractional);
row `i` of one file corresponds to row `i` of the other, and a row-count
mismatch is an error. Only the first four columns are read; deeper levels are
ignored.

### Data source terms and verification

**Terms (read 2026-10-31).** The archive's repository
(`github.com/binance/binance-public-data`) is MIT-licensed for its code, and
its `TERMS_AND_CONDITIONS` licenses the datasets under Creative Commons
Attribution-NonCommercial-ShareAlike 4.0: academic research and non-monetized
open educational projects are permitted; sub-licensing, hosting or
commercially exploiting the datasets is not, and redistributed derivative
work must attribute Binance Vision. No account, key or payment is needed to
download. This project is open, non-commercial research code, uses the data
only to produce the aggregate report, and does **not** commit or host any of
it; credit: Binance Vision. Anyone reusing this for commercial purposes needs
their own licence from Binance. Do not use the data for live trading.

**Binance `bookTicker` mapping: VERIFIED (2026-10-31)** by three checks:

1. The header row of the real file `BTCUSDT-bookTicker-2024-01-15.csv` is
   exactly `update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,transaction_time,event_time`,
   matching the preset. The first rows are `3831603310899,41734.90000000,0.70200000,41735.00000000,2.03300000,1705276800014,1705276800018`
   (bid below ask, sizes in BTC).
2. `event_time` is Unix **milliseconds**: the first row is 2024-01-15
   00:00:00.018 UTC, and the last of the 5,000,000 used rows is at about
   06:26 UTC, in order. In a live capture of the public
   `btcusdt@bookTicker` stream (fields `u, b, B, a, A, T, E`, i.e. update id,
   bid price/qty, ask price/qty, transaction time, event time) `E` was within
   about 0.1 s of the local clock when read in milliseconds, and the field
   order of the archive columns is the same as that stream's.
3. Prices lie on the 0.1 grid (only 34 of 5,000,035 rows were off-grid and
   skipped) and the BTCUSDT `PRICE_FILTER.tickSize` from the exchange's
   `exchangeInfo` is `0.10` (checked 2026-10-31; the 2024-01-15 value is
   inferred from the grid check, not looked up historically).

Caveat: I could not find an official column dictionary for the `bookTicker`
archive files (the archive repository documents aggTrades, klines and trades
only), so the mapping rests on checks 1 to 3, not on a vendor table.

**LOBSTER mapping: still UNVERIFIED** (no LOBSTER file was available).

### Generic CSV

Any CSV with a header row and columns for a timestamp, best bid price/size and
best ask price/size can be used with `--format generic` and the
`--timestamp-col`, `--bid-price-col`, `--bid-qty-col`, `--ask-price-col`,
`--ask-qty-col`, `--timestamp-unit` flags. Column order is irrelevant; names
match case-insensitively.

### Conversion and cleaning rules

* **Tick size is required and never guessed** (`--tick-size`, in price
  units). Every price must be an integer multiple of it (tolerance 1e-4
  ticks); otherwise the row is invalid ("off the tick grid").
* **Model units.** A price becomes `ticks x resolution` integer units, with
  `resolution = 2` by default, so one unit is half a tick. This is needed
  because the core crate's `TopOfBook::mid_price_ticks` truncates
  `(bid + ask) / 2` to an integer; with a 1-tick spread (the norm on liquid
  crypto perpetuals) that would silently make "mid" equal the bid and hide
  ask-side moves. With `resolution = 2` both prices are even, so the mid is
  exact. Spread bucket bounds are given in ticks and converted internally;
  the report converts every error back to ticks.
* **Sizes** are multiplied by `10^qty_decimals` and rounded to integers.
  Imbalance is scale-invariant, so this only affects rounding.
* **Invalid rows** (unparsable field, crossed or locked book, zero size on
  both sides, sentinel/non-positive price, off-grid price, timestamp earlier
  than the previous accepted row's) abort by default, naming the line. With
  `--skip-invalid-rows` they are dropped and **counted by reason in the
  report**. The scripted run uses `--skip-invalid-rows`.
* **Sequence** is the accepted-row index; row order is trusted as the
  happens-before order. Timestamps are only checked for monotonicity.
* `--drop-unchanged` removes rows whose L1 equals the previous accepted
  row's. Default off (see limitations).
* `--max-events N` keeps the first N accepted events. This is a
  chronological *prefix*, not a random sample; it is only a memory guard and
  is stated in the report when it applies.

## Definitions

* **Event.** One accepted row. Time in this protocol is *event time*.
* **State.** The model's existing discretization: `num_imbalance_buckets`
  uniform buckets of `I = Qb / (Qb + Qa)` crossed with spread buckets.
* **Model.** `microprice-calibration` as documented in `docs/model-spec.md`:
  event-to-event transition counting on the training split, Laplace
  smoothing, fixed-point solve of `G* = G1 + Q G*`, prediction
  `mid + G*[state]`. Calibration does not depend on the evaluation horizon.
* **Target.** For event `i` and horizon `h` (events), the exact mid-price
  `(bid + ask) / 2` of event `i + h`.
* **Predictors** (each a prediction of the target made at event `i`):
  1. **naive mid**: `(bid_i + ask_i) / 2`;
  2. **size-weighted mid**: `ask_i * I_i + bid_i * (1 - I_i)` with
     `I = Qb / (Qb + Qa)`;
  3. **calibrated micro-price**: `mid_i + G*[state_i]`.
* **Price-changing observation.** One where target != current mid.

## Wall-clock horizon rule (additional analysis)

For the prediction at event `i` (timestamp `t_i`) and horizon `T`, the target
is the mid of the **quote prevailing at time `t_i + T`**: the last event whose
timestamp is `<= t_i + T`. Precisely:

* **Boundary is closed on the right.** A quote stamped exactly `t_i + T` is
  the target; one stamped `t_i + T + 1 ns` is not (the earlier quote still
  prevails).
* **Ties.** If several events share the prevailing timestamp, the last of
  them (in row order) is used.
* **Quiet window.** If no event lies in `(t_i, t_i + T]`, the prevailing
  quote is event `i` itself: target mid = current mid, the observation is
  kept.
* **End of data.** A candidate whose deadline `t_i + T` is after the last
  timestamp in the test slice is dropped and **counted** ("Candidates
  dropped because the data ends..."), because the prevailing quote there is
  unknown.
* Timestamps must be non-decreasing (ingestion already enforces this).
* The bootstrap block length defaults to `max(1000, 10 x ceil(mean events
  between prediction and target))`, reported per horizon.
* Calibration is unchanged (event-to-event); only the evaluation target
  differs, so the model is asked about a time scale it was not fitted for.
  That mismatch is a property of the analysis, not something it corrects.

## Split

One chronological split: the first 70% of accepted events calibrate the
model; the remaining 30% are the test set. No shuffling. The command asserts
that no training timestamp is later than any test timestamp. The test set is
never used for calibration or for choosing any setting.

## Configuration (fixed in advance, not tuned on the test set)

| setting | value |
|---|---|
| train fraction | 0.7 |
| imbalance buckets | 10 |
| spread bucket bounds | 1, 2, 4 ticks |
| smoothing alpha | 0.5 |
| resolution | 2 |
| horizons | 1, 10, 100 events |
| bootstrap | 1000 resamples, block length max(1000, 10 x largest horizon), seed 42 |

If other settings are tried, all runs are reported and the pre-specified
configuration stays the headline.

## Metrics

For every horizon, on the test set, for each predictor:

* **MAE** and **MSE** of `prediction - target` (in ticks and ticks^2);
* signed **bias**;
* **directional accuracy** on price-changing observations, over those where
  the predictor implies a direction (`prediction != mid`); zero-direction
  predictions abstain, and the number scored is reported, together with the
  majority-direction rate as a reference (a predictor that always guessed the
  more common direction would score that);
* **sample sizes**: events evaluated, skipped, price-changing.

Paired differences (micro-price minus baseline) of MAE and MSE are reported
with 95% intervals from a non-overlapping **block bootstrap** (whole blocks of
consecutive observations are resampled, because overlapping horizons and
microstructure autocorrelation make an i.i.d. bootstrap over-confident).
Directional accuracy carries a Wilson interval that assumes independence and
is therefore optimistic.

## Decision rule (pre-specified)

* **Primary comparison:** MSE difference, micro-price minus naive mid, at
  horizon **10**. MSE is primary because `G*` estimates a conditional
  expectation, which MSE rewards directly.
* The micro-price is reported as **better** only if the 95% interval of that
  difference lies entirely below zero, **worse** if entirely above zero, and
  **not distinguishable** otherwise. The report applies this mechanically.
* The comparison with the weighted mid at horizon 10 is the pre-specified
  secondary comparison.
* Everything else (other horizons, MAE, directional accuracy) is descriptive.
  With three horizons, two baselines and two losses there are many
  comparisons; only the primary one is confirmatory, and no other cell may be
  singled out as the headline after the fact.

## Assumptions and limitations

* **One instrument-day is one draw from one regime.** Even a clear result on
  it does not establish general performance; a negative one does not rule
  out an effect elsewhere.
* **Event time, one-step calibration.** `G*` is an infinite-horizon,
  event-to-event quantity, evaluated here at fixed event horizons. Event
  rates differ wildly across the day and across venues, so `h` events
  corresponds to different wall-clock times. Wall-clock horizons are not
  implemented.
* **Update-process differences.** Binance `bookTicker` emits when the best
  bid/ask price or size changes, so consecutive rows always differ.
  LOBSTER emits one row per message, including messages that leave level 1
  unchanged, so many consecutive rows can be identical; calibration and
  horizons then count those no-op rows. The `--drop-unchanged` variant should
  be run as a sensitivity check on LOBSTER data.
* **Level 1 only, no trades.** Depth beyond the touch, trade signs and queue
  positions are ignored.
* **Chronological prefix.** With `--max-events`, only the earliest part of
  the day is used, which is a specific time-of-day regime.
* **No trading costs, no execution model.** A lower MSE is a statement about
  price prediction, not profit.
* **Bootstrap intervals are approximate**; the block length is a heuristic.
* **The LOBSTER mapping is UNVERIFIED** (the Binance one is verified, see above), and tick size is supplied
  by the operator; a wrong tick size yields off-grid rejections (loud) or, in
  the worst case, a valid but wrong grid (silent).
* Departures of the model from Stoikov (2018) are listed in the README.

## How to run

With normal internet access, from the repository root:

```bash
TICK_SIZE=0.1 scripts/fetch_binance_bookticker.sh BTCUSDT 2024-01-15
```

(The BTCUSDT USD-M tick size of 0.1 was confirmed against `exchangeInfo`
`PRICE_FILTER.tickSize`; for other symbols look it up and never guess.) The script downloads and checksum-checks the file
into the git-ignored `data/` directory, builds the CLI in release mode, runs
`microprice evaluate-csv` with the settings above, and writes
`results/binance-bookticker-<SYMBOL>-<DATE>.md`. The equivalent manual
command is printed by the script and is:

```bash
cargo run --release -p microprice-cli -- evaluate-csv \
  --input data/BTCUSDT-bookTicker-2024-01-15.csv \
  --format binance-bookticker --tick-size 0.1 \
  --skip-invalid-rows --max-events 5000000 \
  --report-md results/binance-bookticker-BTCUSDT-2024-01-15.md
```

For LOBSTER (after downloading a free sample by hand from the vendor):

```bash
cargo run --release -p microprice-cli -- evaluate-csv \
  --input AAPL_2012-06-21_34200000_57600000_orderbook_1.csv \
  --lobster-messages AAPL_2012-06-21_34200000_57600000_message_1.csv \
  --format lobster --tick-size 0.01 --skip-invalid-rows
```

Publishing the result means: copy the generated report into the README's
"Real-data result" section unedited, add the date, the exact command, the
commit hash and the file checksum, and keep the limitations above next to it.

## Live mode

`scripts/record_binance_live.py SYMBOL MINUTES` (needs `websocket-client`,
`TICK_SIZE` set) records the public `<symbol>@bookTicker` stream into a CSV
with the archive's columns and runs the same `evaluate-csv` command and
settings. It is for trying the tool on fresh quotes; short recordings leave
many states unvisited and say little. The recording is written to the
git-ignored `data/` and is covered by the same terms note as above.

## Run log (every run done; none dropped)

All on commit `6787864`, file `BTCUSDT-bookTicker-2024-01-15.csv` (zip SHA-256
`50de14aafe593160e613bcdb46e0f172cbc9ef222ae64ec84b6e3e5ef711d37f`, matches
the published checksum), `--format binance-bookticker --tick-size 0.1
--skip-invalid-rows --max-events 5000000`. Symbol and date were fixed in
advance as the example of this document and were not chosen after looking at
any result.

| # | label | extra flags | report |
|---|---|---|---|
| 1 | **pre-registered** | none | `results/binance-bookticker-BTCUSDT-2024-01-15.md` |
| 2 | exploratory | `--symmetrize` | `results/binance-bookticker-BTCUSDT-2024-01-15-exploratory-symmetrize.md` |
| 3 | exploratory (additional horizons) | `--wall-clock-horizons-ms 100,1000,10000` | `results/binance-bookticker-BTCUSDT-2024-01-15-exploratory-wallclock.md` |
| 4 | tool test, 2 min live, BTCUSDT 2026-10-31 | `scripts/record_binance_live.py BTCUSDT 2` | `results/live-BTCUSDT-20261031T130651Z.md` |

Run 1 output: primary comparison worse than the naive mid, secondary worse
than the weighted mid. No other symbol, date or setting was run. The result
covers only the first 6.4 hours of that day because of the `--max-events`
default.
