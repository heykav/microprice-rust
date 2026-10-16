# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). No release has been
cut from these changes; version numbers are unchanged.

## [Unreleased]

### Fixed
- Smoothed `G1` denominator: with `alpha > 0`, `G1[i]` was
  `delta_sum[i] / (visits[i] + alpha)` while `Q`/`R` used
  `visits[i] + alpha (n + 1)`, so a row was not one probability distribution
  and `G*` could exceed the largest observed move (a 3-state chain whose
  every move was `+1` tick gave `G*` of about 1.157). All three now share one
  denominator; unchanged for `alpha = 0`. Regression and property tests
  added; README numbers, figures and the demo's embedded model regenerated
  (horizon-1 synthetic MAE 0.1094 -> 0.1093). Recorded as amendment 5 in
  `docs/real-data-evaluation.md` (made before any real data was seen).
- `StateSpaceConfig::new` now returns `Result` and rejects bucket counts
  whose product overflows `u32`; previously a model file declaring such a
  state space loaded and then panicked (debug) or indexed out of bounds
  (release) on `predict`.
- `MicroPriceModel::new` now returns `Result` and runs the same validation as
  `load`; `predict` returns an error instead of panicking on an unvalidated
  model.
- `MicroPriceModel::predict` cloned the spread bounds (a heap allocation) on
  every call although documented as allocation-free; the state space is now
  cached in the model.

### Added
- `compare_predictors_next_mid_change` / `resolve_next_mid_change_targets`
  (target = mid at the first later mid change, the quantity `G*` estimates)
  and `calibration_by_state` (per-state reliability table) in
  `microprice-eval`. `microprice evaluate` now also prints a paired MSE/MAE
  comparison with block-bootstrap intervals at the fixed horizon and at the
  next mid change (`--bootstrap-resamples`), and `--calibration-table`.
  Additional analyses; `evaluate-csv` and the pre-registered protocol are
  unchanged.
- Property tests for the estimator/solver
  (`crates/microprice-calibration/tests/properties.rs`).
- Figures: paired MSE differences with intervals, per-state reliability
  (`scripts/make_figures.py`); `scripts/export_site_model.py` regenerates the
  demo page's embedded model.
- Visuals: README banner and crate-layout diagram (dark/light SVG), figures
  regenerated from CLI output by `scripts/make_figures.py` (G* heatmap, MAE
  comparison, martingale drift vs fixed-point residual; all synthetic data),
  a demo-page screenshot and a 1280x640 social-preview image under `docs/img/`.
- `microprice evaluate-parquet` (CLI feature `parquet`, forwarding to
  `microprice-data/parquet-ingestion`; off by default, so the default build
  does not compile arrow/parquet). Schema and rules in
  `docs/parquet-input.md`; tests generate their fixtures in the test. The
  post-ingestion pipeline is now shared with `evaluate-csv`
  (`EvalCommon`, `evaluate_and_report`); CSV behaviour is unchanged.
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
