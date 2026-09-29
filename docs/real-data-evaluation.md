# Real-data evaluation protocol

**Status: no real-data result exists yet.** This document was written, and
committed, before any real-data number was produced by this project. The
development environment could not reach the dataset hosts (checked with
`curl -I`: `data.binance.vision`, `lobsterdata.com` and several other public
quote sources were unreachable), so the only data the code has ever seen is
the synthetic generator's and tiny hand-written format fixtures. Both are
"plumbing" data; nothing in the repository is a result about real markets.

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

### Verification status of these mappings: UNVERIFIED

Both layouts are taken from the task description this work was done against,
**not** checked against the vendors' own documentation or a real file (the
hosts were unreachable). The reader fails with a typed error naming the
missing column if a header does not match, and rejects a LOBSTER row-count
mismatch, but a layout that is structurally compatible and semantically
different would go undetected. Whoever runs the first real evaluation should
open the first few lines of the file and the vendor's documentation and
confirm the mapping before trusting any number. Timestamp units (Binance
milliseconds, LOBSTER seconds) are part of this caveat.

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
* **Mapping and units are UNVERIFIED** (see above), and tick size is supplied
  by the operator; a wrong tick size yields off-grid rejections (loud) or, in
  the worst case, a valid but wrong grid (silent).
* Departures of the model from Stoikov (2018) are listed in the README.

## How to run

With normal internet access, from the repository root:

```bash
TICK_SIZE=0.1 scripts/fetch_binance_bookticker.sh BTCUSDT 2024-01-15
```

(The BTCUSDT USD-M tick size of 0.1 is itself UNVERIFIED here; confirm it in
the exchange's `exchangeInfo` `PRICE_FILTER.tickSize` and choose a date for
which the file exists.) The script downloads and checksum-checks the file
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
