//! Runnable end-to-end example:
//! `cargo run --release -p microprice-eval --example synthetic_then_csv`
//!
//! 1. generates deterministic SYNTHETIC events (not market data),
//! 2. writes them as a Level-1 CSV and reads them back through the same
//!    ingestion path used for real files,
//! 3. calibrates on the first 70% and compares the micro-price with the naive
//!    and size-weighted mid on the remaining 30%.
//!
//! Because the data is synthetic, the printed numbers say nothing about real
//! markets (the generator has no imbalance -> price-direction signal, so no
//! skill is expected). To run on real data see `docs/real-data-evaluation.md`.

use std::fmt::Write as _;

use microprice_calibration::{
    estimate, solve, MicroPriceModel, ModelMetadata, SmoothingConfig, SolverConfig,
    TransitionCounter, SCHEMA_VERSION,
};
use microprice_core::{ImbalanceBucketing, SpreadBucketing, StateSpaceConfig, SymbolId};
use microprice_data::csv::{read_csv_file, CsvIngestConfig, TimestampUnit};
use microprice_data::{MarketDataSource, SyntheticConfig, SyntheticEventGenerator};
use microprice_eval::{chronological_split, compare_predictors, CompareOptions};

const TICK: f64 = 0.01;
const RESOLUTION: i64 = 2; // model units per tick (half-tick => exact mids)

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Synthetic events -> CSV text (prices in currency, tick = 0.01).
    let mut generator = SyntheticEventGenerator::new(SyntheticConfig {
        symbol: SymbolId(1),
        initial_mid_ticks: 10_000,
        initial_spread_ticks: 2,
        initial_bid_qty: 500,
        initial_ask_qty: 500,
        arrival_rate: 0.3,
        cancel_rate: 0.2,
        market_order_rate: 0.2,
        imbalance_persistence: 0.5,
        price_move_probability: 0.1,
        seed: 42,
    })?;
    let mut csv = String::from("ts_ms,bid_px,bid_sz,ask_px,ask_sz\n");
    for i in 0..100_000u64 {
        let e = generator
            .next_event()
            .ok_or("synthetic generator never ends")?;
        writeln!(
            csv,
            "{},{:.2},{},{:.2},{}",
            1_700_000_000_000 + i * 5,
            e.book.bid_price.0 as f64 * TICK,
            e.book.bid_qty.0,
            e.book.ask_price.0 as f64 * TICK,
            e.book.ask_qty.0
        )?;
    }
    let path = std::env::temp_dir().join("microprice-example-quotes.csv");
    std::fs::write(&path, csv)?;

    // 2. Read it back through the CSV ingestion path.
    let mut config = CsvIngestConfig::generic(
        "ts_ms",
        TimestampUnit::Milliseconds,
        "bid_px",
        "bid_sz",
        "ask_px",
        "ask_sz",
        TICK,
    );
    config.qty_decimals = 0;
    let ingest = read_csv_file(&path, &config)?;
    println!(
        "ingested {} events from {}",
        ingest.events.len(),
        path.display()
    );

    // 3. Calibrate on the training split only, evaluate on the test split.
    let (train, test) = chronological_split(&ingest.events, 0.7)?;
    let bounds_units: Vec<i64> = [1, 2, 4].iter().map(|t| t * RESOLUTION).collect();
    let space = StateSpaceConfig::new(
        ImbalanceBucketing::new(10)?,
        SpreadBucketing::new(bounds_units.clone())?,
    );
    let mut counter = TransitionCounter::new(space.state_count());
    counter.observe_events(&space, train)?;
    let estimated = estimate(&counter, SmoothingConfig::new(0.5)?)?;
    let g_star = solve(&estimated, SolverConfig::DEFAULT)?;
    let model = MicroPriceModel::new(
        ModelMetadata {
            schema_version: SCHEMA_VERSION,
            symbol_id: 1,
            num_imbalance_buckets: 10,
            spread_bucket_bounds_ticks: bounds_units,
            smoothing_alpha: 0.5,
            training_observations: counter.total_observations(),
        },
        g_star,
        estimated.p_up.clone(),
        estimated.visits.clone(),
    );

    let r = compare_predictors(
        &model,
        test,
        CompareOptions {
            horizon: 10,
            bootstrap_resamples: 200,
            block_len: 1000,
            seed: 1,
        },
    )?
    .rescaled(1.0 / RESOLUTION as f64);

    println!(
        "test events evaluated: {} (horizon 10 events)",
        r.n_evaluated
    );
    println!(
        "MSE (ticks^2)  naive mid {:.5}  weighted mid {:.5}  micro-price {:.5}",
        r.mid.mse, r.weighted_mid.mse, r.microprice.mse
    );
    println!(
        "MSE(micro-price) - MSE(naive mid) = {:+.6}  95% interval [{:+.6}, {:+.6}]",
        r.mse_diff_vs_mid.estimate, r.mse_diff_vs_mid.lo, r.mse_diff_vs_mid.hi
    );
    std::fs::remove_file(&path)?;
    Ok(())
}
