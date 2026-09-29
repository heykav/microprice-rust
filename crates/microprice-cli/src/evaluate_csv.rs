//! `microprice evaluate-csv`: calibrate on the first part of a real Level-1
//! quote CSV and evaluate out-of-sample on the rest, against the naive-mid
//! and size-weighted-mid baselines, following the protocol fixed in
//! `docs/real-data-evaluation.md` *before* any real result existed.
//!
//! The command prints a Markdown report (also written to `--report-md`) that
//! states its sample sizes and applies the pre-registered decision rule
//! mechanically, so a negative result reads as negative.

use std::fmt::Write as _;
use std::path::PathBuf;

use clap::{Args, ValueEnum};

use microprice_core::{ImbalanceBucketing, SpreadBucketing, StateSpaceConfig, SymbolId};
use microprice_data::csv::{
    read_csv_file, read_lobster_files, CsvIngest, CsvIngestConfig, InvalidRowPolicy, TimestampUnit,
};
use microprice_eval::{
    chronological_split, compare_predictors, compare_predictors_wall_clock, CompareOptions,
    ComparisonReport, Interval,
};

use crate::train::{calibrate_model, parse_spread_bounds, Calibrated};

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum CsvFormatArg {
    /// Named-column CSV; map columns with the --*-col flags.
    Generic,
    /// Binance USD-M futures daily bookTicker file (mapping UNVERIFIED).
    BinanceBookticker,
    /// LOBSTER orderbook + message files (mapping UNVERIFIED).
    Lobster,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum TimestampUnitArg {
    S,
    Ms,
    Us,
    Ns,
}

impl From<TimestampUnitArg> for TimestampUnit {
    fn from(u: TimestampUnitArg) -> Self {
        match u {
            TimestampUnitArg::S => TimestampUnit::Seconds,
            TimestampUnitArg::Ms => TimestampUnit::Milliseconds,
            TimestampUnitArg::Us => TimestampUnit::Microseconds,
            TimestampUnitArg::Ns => TimestampUnit::Nanoseconds,
        }
    }
}

/// Options shared by `evaluate-csv` and `evaluate-parquet`: everything
/// after ingestion (split, model configuration, horizons, bootstrap, report).
#[derive(Args, Debug)]
pub struct EvalCommon {
    /// Model units per exchange tick. 2 keeps mid-prices exact when the
    /// spread is an odd number of ticks; see microprice-data's csv docs.
    #[arg(long, default_value_t = 2)]
    pub resolution: u32,

    /// Fraction of the chronologically ordered stream used for calibration.
    #[arg(long, default_value_t = 0.7)]
    pub train_fraction: f64,

    /// Comma-separated event horizons.
    #[arg(long, default_value = "1,10,100")]
    pub horizons: String,

    /// ADDITIONAL, not pre-registered: comma-separated wall-clock horizons
    /// in milliseconds (e.g. "100,1000"), evaluated with the rule in
    /// docs/real-data-evaluation.md (target = quote prevailing at
    /// t + T, boundary included). Empty (default) = none. Reported after
    /// the event horizons and never used by the decision rule.
    #[arg(long, default_value = "")]
    pub wall_clock_horizons_ms: String,

    /// The one horizon the pre-registered decision rule is applied to. Must
    /// appear in --horizons.
    #[arg(long, default_value_t = 10)]
    pub primary_horizon: usize,

    #[arg(long, default_value_t = 10)]
    pub num_imbalance_buckets: u32,

    /// Spread bucket upper bounds in EXCHANGE TICKS (converted to model
    /// units internally), e.g. "1,2,4".
    #[arg(long, default_value = "1,2,4")]
    pub spread_bucket_bounds: String,

    #[arg(long, default_value_t = 0.5)]
    pub smoothing_alpha: f64,

    /// EXPLORATORY, off by default (the pre-registered configuration does
    /// not symmetrize): pool every transition with its imbalance mirror
    /// image (I <-> 1-I, price moves negated). Runs with this flag are
    /// labelled non-pre-registered in the report.
    #[arg(long)]
    pub symmetrize: bool,

    #[arg(long, default_value_t = 1000)]
    pub bootstrap_resamples: usize,

    /// Bootstrap block length in observations. 0 = max(1000, 10 x largest
    /// horizon).
    #[arg(long, default_value_t = 0)]
    pub block_len: usize,

    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    #[arg(long, default_value_t = 1)]
    pub symbol_id: u32,

    /// Also write the Markdown report to this path.
    #[arg(long)]
    pub report_md: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub struct EvaluateCsvArgs {
    /// Quote CSV (for --format lobster: the `orderbook` file).
    #[arg(long)]
    input: PathBuf,

    #[arg(long, value_enum, default_value_t = CsvFormatArg::Generic)]
    format: CsvFormatArg,

    /// LOBSTER `message` file paired with --input (required for lobster).
    #[arg(long)]
    lobster_messages: Option<PathBuf>,

    /// Currency price of one exchange tick, e.g. 0.1 (BTCUSDT perp) or 0.01.
    /// Prices off this grid are invalid rows. Required: it is not guessed.
    #[arg(long)]
    tick_size: f64,

    /// Decimal digits of size kept (default: 8 for Binance, 0 for LOBSTER,
    /// 8 for generic).
    #[arg(long)]
    qty_decimals: Option<u32>,

    /// Generic format only: timestamp unit.
    #[arg(long, value_enum, default_value_t = TimestampUnitArg::Ms)]
    timestamp_unit: TimestampUnitArg,
    /// Generic format only: header names of the columns.
    #[arg(long, default_value = "timestamp")]
    timestamp_col: String,
    #[arg(long, default_value = "bid_price")]
    bid_price_col: String,
    #[arg(long, default_value = "bid_qty")]
    bid_qty_col: String,
    #[arg(long, default_value = "ask_price")]
    ask_price_col: String,
    #[arg(long, default_value = "ask_qty")]
    ask_qty_col: String,

    /// Drop invalid rows (counted per reason in the report) instead of
    /// aborting on the first one.
    #[arg(long)]
    skip_invalid_rows: bool,

    /// Drop rows whose L1 equals the previous accepted row's.
    #[arg(long)]
    drop_unchanged: bool,

    /// Use only the first N accepted events (a chronological prefix).
    #[arg(long)]
    max_events: Option<usize>,

    #[command(flatten)]
    common: EvalCommon,
}

type BoxErr = Box<dyn std::error::Error>;

fn parse_horizons(s: &str) -> Result<Vec<usize>, BoxErr> {
    let mut out = Vec::new();
    for part in s.split(',') {
        let h: usize = part
            .trim()
            .parse()
            .map_err(|e| format!("invalid --horizons entry {part:?}: {e}"))?;
        if h == 0 {
            return Err("horizons must be >= 1".into());
        }
        out.push(h);
    }
    if out.is_empty() {
        return Err("--horizons is empty".into());
    }
    Ok(out)
}

fn parse_wall_clock_ms(s: &str) -> Result<Vec<u64>, BoxErr> {
    if s.trim().is_empty() {
        return Ok(Vec::new());
    }
    s.split(',')
        .map(|part| {
            let ms: u64 = part
                .trim()
                .parse()
                .map_err(|e| format!("invalid --wall-clock-horizons-ms entry {part:?}: {e}"))?;
            if ms == 0 {
                return Err("wall-clock horizons must be >= 1 ms".into());
            }
            ms.checked_mul(1_000_000)
                .ok_or_else(|| BoxErr::from("wall-clock horizon too large"))
        })
        .collect()
}

fn build_config(args: &EvaluateCsvArgs) -> CsvIngestConfig {
    let mut cfg = match args.format {
        CsvFormatArg::BinanceBookticker => CsvIngestConfig::binance_book_ticker(args.tick_size),
        CsvFormatArg::Lobster => CsvIngestConfig::lobster_orderbook(args.tick_size),
        CsvFormatArg::Generic => CsvIngestConfig::generic(
            &args.timestamp_col,
            args.timestamp_unit.into(),
            &args.bid_price_col,
            &args.bid_qty_col,
            &args.ask_price_col,
            &args.ask_qty_col,
            args.tick_size,
        ),
    };
    cfg.resolution = args.common.resolution;
    if let Some(d) = args.qty_decimals {
        cfg.qty_decimals = d;
    }
    cfg.symbol = SymbolId(args.common.symbol_id);
    cfg.on_invalid_row = if args.skip_invalid_rows {
        InvalidRowPolicy::Skip
    } else {
        InvalidRowPolicy::Error
    };
    cfg.drop_unchanged = args.drop_unchanged;
    cfg.max_events = args.max_events;
    cfg
}

pub fn run(args: EvaluateCsvArgs) -> Result<(), BoxErr> {
    // Validate the shared options before touching the filesystem.
    Plan::parse(&args.common)?;
    let cfg = build_config(&args);

    eprintln!("Reading {} ...", args.input.display());
    let ingest = match args.format {
        CsvFormatArg::Lobster => {
            let msgs = args
                .lobster_messages
                .as_ref()
                .ok_or("--format lobster requires --lobster-messages <message file>")?;
            read_lobster_files(&args.input, msgs, &cfg)?
        }
        _ => read_csv_file(&args.input, &cfg)?,
    };
    let source = SourceInfo {
        command: "evaluate-csv",
        file_name: file_name_of(&args.input),
        format: format!("{:?}", args.format),
        tick_size: Some(args.tick_size),
    };
    evaluate_and_report(&args.common, &source, &ingest)
}

/// Where the events came from, for the report header.
pub struct SourceInfo {
    pub command: &'static str,
    pub file_name: String,
    pub format: String,
    /// Price of one exchange tick, when the input format carries it.
    pub tick_size: Option<f64>,
}

pub fn file_name_of(path: &std::path::Path) -> String {
    path.file_name()
        .map(|f| f.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Parsed, validated horizon options.
struct Plan {
    horizons: Vec<usize>,
    wall_clock_ns: Vec<u64>,
}

impl Plan {
    fn parse(common: &EvalCommon) -> Result<Plan, BoxErr> {
        let horizons = parse_horizons(&common.horizons)?;
        let wall_clock_ns = parse_wall_clock_ms(&common.wall_clock_horizons_ms)?;
        if !horizons.contains(&common.primary_horizon) {
            return Err(format!(
                "--primary-horizon {} must be one of --horizons {:?}",
                common.primary_horizon, horizons
            )
            .into());
        }
        Ok(Plan {
            horizons,
            wall_clock_ns,
        })
    }
}

/// Everything after ingestion: chronological split, calibration, paired
/// comparison at every horizon, Markdown report. Shared by
/// `evaluate-csv` and `evaluate-parquet` so both apply identical logic.
pub fn evaluate_and_report(
    common: &EvalCommon,
    source: &SourceInfo,
    ingest: &CsvIngest,
) -> Result<(), BoxErr> {
    let Plan {
        horizons,
        wall_clock_ns,
    } = Plan::parse(common)?;
    let events = &ingest.events;
    if events.len() < 100 {
        return Err(format!(
            "only {} usable events after ingestion; too few to calibrate and evaluate",
            events.len()
        )
        .into());
    }

    let (train, test) = chronological_split(events, common.train_fraction)?;
    // Hard guard on the no-leakage requirement: every training timestamp
    // must precede or equal every test timestamp.
    let train_last = train.last().map(|e| e.timestamp_ns).unwrap_or(0);
    let test_first = test.first().map(|e| e.timestamp_ns).unwrap_or(0);
    if train_last > test_first {
        return Err("internal error: train/test split is not chronological".into());
    }

    let res = i64::from(common.resolution);
    let bounds_ticks = parse_spread_bounds(&common.spread_bucket_bounds)?;
    let bounds_units: Vec<i64> = bounds_ticks.iter().map(|b| b * res).collect();
    let state_space = StateSpaceConfig::new(
        ImbalanceBucketing::new(common.num_imbalance_buckets)?,
        SpreadBucketing::new(bounds_units.clone())?,
    );

    eprintln!(
        "Calibrating on {} train events, evaluating on {} test events ...",
        train.len(),
        test.len()
    );
    let calibrated = calibrate_model(
        train,
        &state_space,
        common.symbol_id,
        common.num_imbalance_buckets,
        bounds_units,
        common.smoothing_alpha,
        common.symmetrize,
    )?;
    let model = &calibrated.model;

    let max_h = *horizons.iter().max().unwrap_or(&1);
    let block_len = if common.block_len == 0 {
        1000.max(10 * max_h)
    } else {
        common.block_len
    };
    let mut reports = Vec::new();
    for &h in &horizons {
        let r = compare_predictors(
            model,
            test,
            CompareOptions {
                horizon: h,
                bootstrap_resamples: common.bootstrap_resamples,
                block_len,
                seed: common.seed,
            },
        )?;
        reports.push(r.rescaled(1.0 / common.resolution as f64));
    }
    for &ns in &wall_clock_ns {
        let r = compare_predictors_wall_clock(
            model,
            test,
            ns,
            CompareOptions {
                horizon: 1,
                bootstrap_resamples: common.bootstrap_resamples,
                block_len: common.block_len,
                seed: common.seed,
            },
        )?;
        reports.push(r.rescaled(1.0 / common.resolution as f64));
    }

    let md = render_markdown(
        common,
        source,
        ingest,
        (train, test),
        &calibrated,
        &reports,
        block_len,
    );
    println!("{md}");
    if let Some(path) = &common.report_md {
        if let Some(dir) = path.parent() {
            if !dir.as_os_str().is_empty() {
                std::fs::create_dir_all(dir)?;
            }
        }
        std::fs::write(path, &md)?;
        eprintln!("Wrote {}", path.display());
    }
    Ok(())
}

struct ModelSummary {
    states: usize,
    zero_visit_states: usize,
    training_observations: u64,
}

fn model_summary(m: &microprice_calibration::MicroPriceModel, zero: usize) -> ModelSummary {
    ModelSummary {
        states: m.g_star().len(),
        zero_visit_states: zero,
        training_observations: m.metadata().training_observations,
    }
}

fn fmt_iv(i: Interval) -> String {
    if i.lo.is_nan() {
        format!("{:+.6} (interval not computed)", i.estimate)
    } else {
        format!("{:+.6} [{:+.6}, {:+.6}]", i.estimate, i.lo, i.hi)
    }
}

/// The pre-registered decision rule, applied mechanically to a paired
/// difference `microprice - baseline` of a loss (negative favours the
/// micro-price).
fn verdict(diff: Interval, baseline: &str) -> String {
    if diff.lo.is_nan() {
        return format!("no interval computed; cannot apply the decision rule vs {baseline}");
    }
    if diff.hi < 0.0 {
        format!("micro-price loss is LOWER than {baseline}; the 95% interval excludes zero")
    } else if diff.lo > 0.0 {
        format!(
            "micro-price loss is HIGHER than {baseline} (worse); the 95% interval excludes zero"
        )
    } else {
        format!("NO distinguishable difference from {baseline}; the 95% interval includes zero")
    }
}

fn render_markdown(
    args: &EvalCommon,
    source: &SourceInfo,
    ingest: &CsvIngest,
    (train, test): (&[microprice_core::BookEvent], &[microprice_core::BookEvent]),
    calibrated: &Calibrated,
    reports: &[ComparisonReport],
    block_len: usize,
) -> String {
    let res = args.resolution as f64;
    let zero_visit_states = calibrated
        .model
        .visits()
        .iter()
        .filter(|v| **v == 0)
        .count();
    let model = &model_summary(&calibrated.model, zero_visit_states);
    let mut s = String::new();
    let all = || train.iter().chain(test.iter());

    let _ = writeln!(s, "# Real-data evaluation report\n");
    let _ = writeln!(
        s,
        "Generated by `microprice {}` v{}. Protocol: `docs/real-data-evaluation.md`. \
         All lengths are in exchange ticks{}.\n",
        source.command,
        env!("CARGO_PKG_VERSION"),
        match source.tick_size {
            Some(t) => format!(" (1 tick = {t} in price units)"),
            None => format!(
                " (input prices are integer model units; {} units = 1 tick)",
                args.resolution
            ),
        }
    );

    let _ = writeln!(s, "## Data\n");
    let _ = writeln!(s, "- input file: `{}`", source.file_name);
    let _ = writeln!(s, "- format: {}", source.format);
    let _ = writeln!(s, "- rows read: {}", ingest.rows_read);
    let _ = writeln!(s, "- events accepted: {}", ingest.events.len());
    if ingest.skipped.is_empty() {
        let _ = writeln!(s, "- rows skipped as invalid: 0");
    } else {
        for (reason, n) in &ingest.skipped {
            let _ = writeln!(s, "- rows skipped ({}): {n}", reason.as_str());
        }
    }
    let _ = writeln!(
        s,
        "- unchanged-L1 rows dropped: {}",
        ingest.unchanged_dropped
    );
    if ingest.truncated_at_max_events {
        let _ = writeln!(
            s,
            "- **truncated**: only the first {} accepted events (a chronological prefix, not a random sample) were used",
            ingest.events.len()
        );
    }
    let first = all().next().map(|e| e.timestamp_ns).unwrap_or(0);
    let last = test.last().map(|e| e.timestamp_ns).unwrap_or(0);
    let _ = writeln!(
        s,
        "- time span: {:.1} s (unix ns {first} .. {last})",
        (last - first) as f64 / 1e9
    );
    let mut spreads: Vec<i64> = all().map(|e| e.book.spread().0).collect();
    spreads.sort_unstable();
    let median_spread = spreads[spreads.len() / 2] as f64 / res;
    let one_tick = spreads
        .iter()
        .filter(|s| **s == args.resolution as i64)
        .count();
    let _ = writeln!(
        s,
        "- median spread: {median_spread:.2} ticks; spread == 1 tick in {:.1}% of events",
        100.0 * one_tick as f64 / spreads.len() as f64
    );
    let evs: Vec<_> = all().collect();
    let mid_changes = evs
        .windows(2)
        .filter(|w| {
            w[0].book.bid_price.0 + w[0].book.ask_price.0
                != w[1].book.bid_price.0 + w[1].book.ask_price.0
        })
        .count();
    let _ = writeln!(
        s,
        "- consecutive events with a mid change: {:.1}%\n",
        100.0 * mid_changes as f64 / (evs.len().max(2) - 1) as f64
    );

    let _ = writeln!(s, "## Split and model\n");
    let _ = writeln!(
        s,
        "- chronological split at {:.0}%: {} train events, {} test events (test starts at or after the last training timestamp)",
        args.train_fraction * 100.0,
        train.len(),
        test.len()
    );
    let _ = writeln!(
        s,
        "- state space: {} imbalance buckets x spread bounds [{}] ticks = {} states; {} had zero training visits (smoothing alpha {})",
        args.num_imbalance_buckets,
        args.spread_bucket_bounds,
        model.states,
        model.zero_visit_states,
        args.smoothing_alpha
    );
    let _ = writeln!(
        s,
        "- training transitions observed: {}",
        model.training_observations
    );
    let _ = writeln!(
        s,
        "- calibration: {}",
        if calibrated.symmetrized {
            "**imbalance-symmetrized (EXPLORATORY: not the pre-registered configuration)**"
        } else {
            "pre-registered configuration (no symmetrization)"
        }
    );
    let to_ticks = 1.0 / res;
    let _ = writeln!(
        s,
        "- antisymmetry residual max|G*[s] + G*[mirror(s)]|: {:.3e} ticks",
        calibrated.antisymmetry_residual * to_ticks
    );
    let _ = writeln!(
        s,
        "- martingale diagnostic (model's own transition kernel, training split): {}",
        calibrated.martingale.rescaled(to_ticks).summary("ticks")
    );
    let _ = writeln!(
        s,
        "  (Descriptive only. The fixed-point residual is ~0 by construction; a nonzero one-step drift means \
         the micro-price is not a martingale under this model's kernel. See `docs/model-spec.md`.)\n"
    );

    let _ = writeln!(s, "## Results (test split only)\n");
    let _ = writeln!(
        s,
        "Loss = error of predicting the mid-price `horizon` events ahead. Differences are \
         paired per observation, micro-price minus baseline, so **negative favours the \
         micro-price**. Intervals: 95% percentile, non-overlapping block bootstrap \
         ({} resamples, block length {block_len}, seed {}); an approximation under serial dependence.\n",
        args.bootstrap_resamples, args.seed
    );
    for r in reports {
        if let Some(ns) = r.horizon_ns {
            let _ = writeln!(
                s,
                "### Wall-clock horizon {} ms (additional, NOT pre-registered)\n",
                ns as f64 / 1e6
            );
            let _ = writeln!(
                s,
                "Target = mid of the quote prevailing at t + T (a quote stamped exactly t + T is \
                 included). Candidates dropped because the data ends before t + T: {}. Mean events \
                 between prediction and target: {:.2}. Bootstrap block length {}.\n",
                r.n_unresolved, r.mean_events_ahead, r.block_len
            );
        } else {
            let _ = writeln!(s, "### Horizon {} events\n", r.horizon);
        }
        let _ = writeln!(
            s,
            "n evaluated = {} (skipped, unencodable: {}); price-changing observations = {} ({:.1}% up)\n",
            r.n_evaluated,
            r.n_skipped,
            r.n_price_changing,
            100.0 * r.up_share
        );
        let _ = writeln!(
            s,
            "| predictor | MAE (ticks) | MSE (ticks^2) | bias (ticks) |"
        );
        let _ = writeln!(s, "|---|---|---|---|");
        for (name, l) in [
            ("naive mid", r.mid),
            ("weighted mid", r.weighted_mid),
            ("micro-price", r.microprice),
        ] {
            let _ = writeln!(
                s,
                "| {name} | {:.6} | {:.6} | {:+.6} |",
                l.mae, l.mse, l.bias
            );
        }
        let _ = writeln!(s);
        let _ = writeln!(
            s,
            "- MSE difference, micro-price - naive mid: {}",
            fmt_iv(r.mse_diff_vs_mid)
        );
        let _ = writeln!(
            s,
            "- MAE difference, micro-price - naive mid: {}",
            fmt_iv(r.mae_diff_vs_mid)
        );
        let _ = writeln!(
            s,
            "- MSE difference, micro-price - weighted mid: {}",
            fmt_iv(r.mse_diff_vs_weighted_mid)
        );
        let _ = writeln!(
            s,
            "- MAE difference, micro-price - weighted mid: {}",
            fmt_iv(r.mae_diff_vs_weighted_mid)
        );
        for (name, d) in [
            ("micro-price", r.microprice_direction),
            ("weighted mid", r.weighted_mid_direction),
        ] {
            if d.n_scored == 0 {
                let _ = writeln!(
                    s,
                    "- directional accuracy, {name}: not computable (no scored observations)"
                );
            } else {
                let _ = writeln!(
                    s,
                    "- directional accuracy, {name}: {:.4} (Wilson 95% [{:.4}, {:.4}], n = {}; reference: always guessing the majority direction scores {:.4})",
                    d.accuracy,
                    d.ci_lo,
                    d.ci_hi,
                    d.n_scored,
                    r.up_share.max(1.0 - r.up_share)
                );
            }
        }
        let _ = writeln!(s);
    }

    let _ = writeln!(
        s,
        "## Pre-registered decision (primary horizon {})\n",
        args.primary_horizon
    );
    if calibrated.symmetrized {
        let _ = writeln!(
            s,
            "**This run used `--symmetrize`, which is not the pre-registered configuration. \
             The verdicts below are exploratory and must not be reported as the pre-registered result.**\n"
        );
    }
    if let Some(p) = reports
        .iter()
        .find(|r| r.horizon_ns.is_none() && r.horizon == args.primary_horizon)
    {
        let _ = writeln!(
            s,
            "- primary comparison (MSE vs naive mid): {}",
            verdict(p.mse_diff_vs_mid, "the naive mid")
        );
        let _ = writeln!(
            s,
            "- secondary (MSE vs weighted mid): {}",
            verdict(p.mse_diff_vs_weighted_mid, "the weighted mid")
        );
    }
    let _ = writeln!(
        s,
        "\nOther horizons and metrics are reported for completeness and are not used to \
         pick a winner after the fact. Limitations are listed in `docs/real-data-evaluation.md`; \
         one instrument-day is one sample of one regime and does not establish general performance."
    );
    s
}
