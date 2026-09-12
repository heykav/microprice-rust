//! `microprice train`: generate a synthetic event stream, run it through the
//! full transition-counting -> estimation -> solver pipeline, and save the
//! resulting model artifact.
//!
//! Synthetic data is the *only* data source `train` has today (Parquet/CSV
//! ingestion is Phase 12) — this is disclosed in the command's own help
//! text, not hidden behind a generic "train" name.
//!
//! [`calibrate_model`] is also reused directly by `microprice evaluate`
//! (see `crate::evaluate`), which needs to calibrate against a chronological
//! train split rather than a whole synthetic stream — the two commands
//! share the exact same counting/estimation/solving logic rather than
//! having it copied.

use std::path::PathBuf;
use std::time::Instant;

use clap::Args;

use microprice_calibration::{
    estimate, solve, CalibrationError, MicroPriceModel, ModelMetadata, SmoothingConfig,
    SolverConfig, TransitionCounter, SCHEMA_VERSION,
};
use microprice_core::{BookEvent, StateId, StateSpaceConfig, SymbolId};
use microprice_data::{MarketDataSource, SyntheticConfig, SyntheticEventGenerator};

#[derive(Args, Debug)]
pub struct TrainArgs {
    /// Path to write the trained model artifact to (a `<path>.json`
    /// human-readable metadata sidecar is written alongside it).
    #[arg(long)]
    output: PathBuf,

    /// Number of synthetic events to generate and calibrate against.
    #[arg(long, default_value_t = 500_000)]
    num_events: u64,

    #[command(flatten)]
    common: CommonTrainArgs,
}

/// Arguments shared between `microprice train` and `microprice evaluate`
/// (which also needs to calibrate a model, just against a chronological
/// train split rather than a freestanding synthetic stream).
#[derive(Args, Debug, Clone)]
pub struct CommonTrainArgs {
    /// Number of uniform queue-imbalance buckets.
    #[arg(long, default_value_t = 20)]
    pub num_imbalance_buckets: u32,

    /// Comma-separated, strictly increasing finite spread-tick bucket
    /// upper bounds, e.g. "1,2,4" (see `docs/model-spec.md`'s spread
    /// bucketing section). Empty (the default) means a single catch-all
    /// spread bucket.
    #[arg(long, default_value = "")]
    pub spread_bucket_bounds: String,

    /// Laplace smoothing alpha. 0.0 is plain maximum likelihood, which
    /// fails outright if any reachable state was never observed — raise
    /// this or --num-events if training reports unvisited states.
    #[arg(long, default_value_t = 0.5)]
    pub smoothing_alpha: f64,

    /// RNG seed for the synthetic data generator (same seed -> byte
    /// identical training data, see microprice-data's module docs).
    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    /// Symbol id recorded in the model's metadata.
    #[arg(long, default_value_t = 1)]
    pub symbol_id: u32,
    #[arg(long, default_value_t = 10_000)]
    pub initial_mid_ticks: i64,
    #[arg(long, default_value_t = 2)]
    pub initial_spread_ticks: i64,
    #[arg(long, default_value_t = 500)]
    pub initial_bid_qty: u64,
    #[arg(long, default_value_t = 500)]
    pub initial_ask_qty: u64,
    #[arg(long, default_value_t = 0.3)]
    pub arrival_rate: f64,
    #[arg(long, default_value_t = 0.2)]
    pub cancel_rate: f64,
    #[arg(long, default_value_t = 0.2)]
    pub market_order_rate: f64,
    #[arg(long, default_value_t = 0.5)]
    pub imbalance_persistence: f64,
    #[arg(long, default_value_t = 0.1)]
    pub price_move_probability: f64,
}

impl CommonTrainArgs {
    pub fn synthetic_config(&self) -> SyntheticConfig {
        SyntheticConfig {
            symbol: SymbolId(self.symbol_id),
            initial_mid_ticks: self.initial_mid_ticks,
            initial_spread_ticks: self.initial_spread_ticks,
            initial_bid_qty: self.initial_bid_qty,
            initial_ask_qty: self.initial_ask_qty,
            arrival_rate: self.arrival_rate,
            cancel_rate: self.cancel_rate,
            market_order_rate: self.market_order_rate,
            imbalance_persistence: self.imbalance_persistence,
            price_move_probability: self.price_move_probability,
            seed: self.seed,
        }
    }
}

pub fn parse_spread_bounds(s: &str) -> Result<Vec<i64>, Box<dyn std::error::Error>> {
    let s = s.trim();
    if s.is_empty() {
        return Ok(vec![]);
    }
    s.split(',')
        .map(|part| {
            part.trim()
                .parse::<i64>()
                .map_err(|e| format!("invalid --spread-bucket-bounds entry {part:?}: {e}").into())
        })
        .collect()
}

/// Runs the shared counting -> estimation -> solving pipeline against
/// `events` and returns the resulting model. Prints diagnostics (unvisited
/// states, etc.) to stdout, since both `train` and `evaluate` want them.
pub fn calibrate_model(
    events: &[BookEvent],
    state_space: &StateSpaceConfig,
    symbol_id: u32,
    num_imbalance_buckets: u32,
    spread_bounds: Vec<i64>,
    smoothing_alpha: f64,
) -> Result<MicroPriceModel, CalibrationError> {
    let mut counter = TransitionCounter::new(state_space.state_count());
    counter.observe_events(state_space, events)?;
    println!(
        "Observed {} transitions across {} states ({} imbalance buckets x {} spread buckets).",
        counter.total_observations(),
        state_space.state_count(),
        num_imbalance_buckets,
        spread_bounds.len() + 1
    );

    let unvisited: Vec<u32> = (0..state_space.state_count())
        .filter(|&s| counter.visits(StateId(s)) == 0)
        .collect();
    if !unvisited.is_empty() {
        let shown: Vec<String> = unvisited.iter().take(10).map(u32::to_string).collect();
        println!(
            "  warning: {} of {} states had zero visits in this run (e.g. state(s) {}{})",
            unvisited.len(),
            state_space.state_count(),
            shown.join(","),
            if unvisited.len() > 10 { ", ..." } else { "" }
        );
        if smoothing_alpha == 0.0 {
            println!(
                "  --smoothing-alpha is 0.0: estimation will fail for these states. \
                 Pass --smoothing-alpha > 0.0, or --num-events higher, or fewer buckets."
            );
        }
    }

    let smoothing = SmoothingConfig::new(smoothing_alpha)?;
    let estimated = estimate(&counter, smoothing)?;
    let g_star = solve(&estimated, SolverConfig::DEFAULT)?;

    let metadata = ModelMetadata {
        schema_version: SCHEMA_VERSION,
        symbol_id,
        num_imbalance_buckets,
        spread_bucket_bounds_ticks: spread_bounds,
        smoothing_alpha,
        training_observations: counter.total_observations(),
    };
    Ok(MicroPriceModel::new(
        metadata,
        g_star,
        estimated.visits.clone(),
    ))
}

pub fn run(args: TrainArgs) -> Result<(), Box<dyn std::error::Error>> {
    let spread_bounds = parse_spread_bounds(&args.common.spread_bucket_bounds)?;

    let imbalance = microprice_core::ImbalanceBucketing::new(args.common.num_imbalance_buckets)?;
    let spread = microprice_core::SpreadBucketing::new(spread_bounds.clone())?;
    let state_space = StateSpaceConfig::new(imbalance, spread);

    let mut generator = SyntheticEventGenerator::new(args.common.synthetic_config())?;

    println!(
        "Generating {} synthetic events (seed {}) — this is microprice-data's disclosed \
         synthetic generator, not real market data (Parquet/CSV ingestion is a later phase).",
        args.num_events, args.common.seed
    );
    let gen_start = Instant::now();
    let events: Vec<_> = (0..args.num_events)
        .map(|_| {
            generator
                .next_event()
                .expect("SyntheticEventGenerator::next_event never returns None")
        })
        .collect();
    println!("  generated in {:.3}s", gen_start.elapsed().as_secs_f64());

    let model = calibrate_model(
        &events,
        &state_space,
        args.common.symbol_id,
        args.common.num_imbalance_buckets,
        spread_bounds,
        args.common.smoothing_alpha,
    )?;
    model.save(&args.output)?;

    println!(
        "Saved model to {} (plus a JSON metadata sidecar next to it).",
        args.output.display()
    );
    Ok(())
}
