//! `microprice benchmark`: measure real `MicroPriceModel::predict` /
//! `predict_batch` throughput, on this machine, right now — not a claim
//! about any other machine. Complements the Criterion micro-benchmark in
//! `microprice-core` (see `docs/benchmarking.md`) by exercising the full
//! inference path (state encoding + adjustment lookup + weighted-mid) end
//! to end, against a model actually loaded from disk.

use std::path::PathBuf;
use std::time::Instant;

use clap::Args;

use microprice_calibration::{MicroPriceEstimate, MicroPriceModel};
use microprice_core::SymbolId;
use microprice_data::{MarketDataSource, SyntheticConfig, SyntheticEventGenerator};

#[derive(Args, Debug)]
pub struct BenchmarkArgs {
    /// Path to a model artifact produced by `microprice train`.
    #[arg(long)]
    model: PathBuf,
    /// Number of synthetic top-of-book snapshots to predict against.
    #[arg(long, default_value_t = 100_000)]
    num_books: u64,
    #[arg(long, default_value_t = 7)]
    seed: u64,
}

pub fn run(args: BenchmarkArgs) -> Result<(), Box<dyn std::error::Error>> {
    let model = MicroPriceModel::load(&args.model)?;

    // A fixed, independent synthetic stream just to get a realistic mix of
    // valid books to predict against - its own rates don't need to match
    // whatever the model was trained on, since predict() only cares that
    // each book encodes to some valid state.
    let synth_config = SyntheticConfig {
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
        seed: args.seed,
    };
    let mut generator = SyntheticEventGenerator::new(synth_config)?;
    let books: Vec<_> = (0..args.num_books)
        .map(|_| {
            generator
                .next_event()
                .expect("SyntheticEventGenerator::next_event never returns None")
                .book
        })
        .collect();

    // Scalar predict, one call per book.
    let mut sink = 0.0f64;
    let scalar_start = Instant::now();
    for book in &books {
        sink += model.predict(book)?.microprice_ticks;
    }
    let scalar_elapsed = scalar_start.elapsed();

    // Batch predict into a pre-allocated output slice.
    let mut output = vec![
        MicroPriceEstimate {
            mid_ticks: 0.0,
            weighted_mid_ticks: 0.0,
            microprice_ticks: 0.0,
            adjustment_ticks: 0.0,
            state_id: 0,
            state_observations: 0,
        };
        books.len()
    ];
    let batch_start = Instant::now();
    model.predict_batch(&books, &mut output)?;
    let batch_elapsed = batch_start.elapsed();

    let n = books.len() as f64;
    println!(
        "MicroPriceModel inference benchmark - {} books, seed {} (measured now, on this \
         machine; hardware/toolchain are NOT captured here, unlike docs/benchmarking.md's \
         Criterion numbers - report them yourself if you cite this).",
        books.len(),
        args.seed
    );
    println!(
        "  predict (scalar, one call/book): {:>8.2} ns/predict  (total {:.3} ms, checksum {sink:.3})",
        scalar_elapsed.as_nanos() as f64 / n,
        scalar_elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "  predict_batch:                    {:>8.2} ns/predict  (total {:.3} ms)",
        batch_elapsed.as_nanos() as f64 / n,
        batch_elapsed.as_secs_f64() * 1000.0
    );
    Ok(())
}
