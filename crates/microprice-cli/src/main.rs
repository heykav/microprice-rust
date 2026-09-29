//! `microprice` CLI (Phase 10): train, predict, inspect, and benchmark
//! queue-imbalance micro-price models end-to-end against the synthetic
//! development dataset (`microprice-data`'s `SyntheticEventGenerator`).
//!
//! `train` and `evaluate` use the synthetic generator (and say so);
//! `evaluate-csv` runs the same calibration and out-of-sample evaluation on
//! a real Level-1 quote CSV.

mod benchmark;
mod evaluate;
mod evaluate_csv;
mod inspect;
mod predict;
mod train;
mod visualize;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "microprice",
    version,
    about = "Calibrate and query queue-imbalance / Markov-chain micro-price models."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Calibrate a model against freshly generated synthetic data and save it.
    Train(train::TrainArgs),
    /// Load a model and predict the micro-price for one top-of-book snapshot.
    Predict(predict::PredictArgs),
    /// Load a model and print its metadata and a summary of its calibrated state.
    Inspect(inspect::InspectArgs),
    /// Measure real predict / predict_batch throughput on this machine.
    Benchmark(benchmark::BenchmarkArgs),
    /// Chronological out-of-sample evaluation against mid/weighted_mid baselines.
    Evaluate(evaluate::EvaluateArgs),
    /// Calibrate and evaluate on a real Level-1 quote CSV (generic, Binance bookTicker, LOBSTER).
    EvaluateCsv(evaluate_csv::EvaluateCsvArgs),
    /// Render static PNG plots of a calibrated model's g_star surface.
    Visualize(visualize::VisualizeArgs),
}

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Train(args) => train::run(args),
        Command::Predict(args) => predict::run(args),
        Command::Inspect(args) => inspect::run(args),
        Command::Benchmark(args) => benchmark::run(args),
        Command::Evaluate(args) => evaluate::run(args),
        Command::EvaluateCsv(args) => evaluate_csv::run(args),
        Command::Visualize(args) => visualize::run(args),
    };
    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}
