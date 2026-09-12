//! `microprice` CLI (Phase 10): train, predict, inspect, and benchmark
//! queue-imbalance micro-price models end-to-end against the synthetic
//! development dataset (`microprice-data`'s `SyntheticEventGenerator`).
//!
//! Real-data (CSV/Parquet) ingestion is Phase 12 — `train` only has one
//! data source available today, and says so.

mod benchmark;
mod evaluate;
mod inspect;
mod predict;
mod train;

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
}

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Train(args) => train::run(args),
        Command::Predict(args) => predict::run(args),
        Command::Inspect(args) => inspect::run(args),
        Command::Benchmark(args) => benchmark::run(args),
        Command::Evaluate(args) => evaluate::run(args),
    };
    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}
