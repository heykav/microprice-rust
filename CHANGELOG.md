# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). No release has been
cut from these changes; version numbers are unchanged.

## [Unreleased]

### Added
- Python bindings: `MicroPriceModel.predict_batch` (vectorised over integer
  columns; NumPy optional, not a dependency), tested in
  `docs/python_bindings_smoke_test.py` and documented in
  `docs/python-bindings.md`.
- Wall-clock evaluation horizons: `compare_predictors_wall_clock` and
  `resolve_wall_clock_targets` in `microprice-eval`, CLI
  `evaluate-csv --wall-clock-horizons-ms`. Target = quote prevailing at
  `t + T` (closed boundary, last of tied timestamps, end-of-data candidates
  dropped and counted). Additional and not pre-registered; the
  pre-registered event horizons and decision rule are unchanged.
  `ComparisonReport` gains `horizon_ns`, `n_unresolved`, `mean_events_ahead`.
- Optional imbalance-mirror symmetrization in calibration
  (`TransitionCounter::symmetrized`, `mirror_state`; CLI `--symmetrize` on
  `train`, `evaluate`, `evaluate-csv`). Off by default; makes `G*` exactly
  antisymmetric. Derivation in `docs/model-spec.md`.
- Martingale / fixed-point diagnostic (`martingale_diagnostic`,
  `antisymmetry_residual`), printed by `train`, `evaluate` and `evaluate-csv`
  and included in the `evaluate-csv` report; `inspect` prints the
  antisymmetry residual. Finding recorded in the spec: the current recursion
  `G* = G1 + Q G*` is **not** a martingale by construction; the measured
  drift is reported, not assumed away.
- `solve_full_chain` (library only, experimental): solves
  `G = G1 + (Q + R) G`, which is a martingale by construction. Not wired to
  any CLI command or evaluation predictor.
- `TransitionCounter` now also counts where price-changing transitions land
  (`price_change_count`); `EstimatedTransitions` gains the matching `r`
  matrix. Neither enters `Q`, `G1` or `G*`.
- `docs/real-data-evaluation.md`: dated amendments section (2026-09-29,
  before any real-data result exists). Pre-registered predictors,
  configuration and decision rule unchanged.
- `microprice-data::csv`: Level-1 CSV ingestion with a generic named-column
  schema and Binance USD-M `bookTicker` and LOBSTER presets (mappings
  UNVERIFIED against vendor documentation). Half-tick model units keep
  mid-prices exact; invalid rows abort with a line number or are skipped and
  counted per reason.
- `microprice-eval::compare_predictors`: naive mid vs size-weighted mid vs
  micro-price with MAE, MSE, bias, directional accuracy on price-changing
  observations, sample sizes, block-bootstrap and Wilson intervals.
- `microprice evaluate-csv`: chronological calibration and evaluation on a
  CSV, writing a Markdown report that applies the pre-registered decision
  rule.
- `docs/real-data-evaluation.md`: evaluation protocol fixed before any real
  result.
- `scripts/fetch_binance_bookticker.sh`: fetch one public day of Binance
  data and run the evaluation. Downloaded data is git-ignored.
- Runnable example `synthetic_then_csv` (`microprice-eval`) and doctests in
  `microprice-core`, `microprice-calibration` and `microprice-data`.
- `CONTRIBUTING.md` and this changelog.
- Workspace-inherited Cargo metadata (homepage, readme, keywords, categories,
  authors). `rust-version` is intentionally not set: MSRV was not measured.

### Changed
- README rewritten: technical pitch and quickstart first, candid synthetic
  result kept prominent, "Real-data result" (pending) section, a section on
  departures from Stoikov (2018), Linux fontconfig requirement, stale test
  counts replaced by the command to run, inconsistent phase-numbered roadmap
  replaced by a component status list.
- Crate descriptions no longer refer to internal phase numbers.
