# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). No release has been
cut from these changes; version numbers are unchanged.

## [Unreleased]

### Added
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
