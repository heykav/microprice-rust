//! `microprice evaluate`: generate a synthetic stream, split it
//! *chronologically* (never shuffled — see `microprice_eval::split`'s
//! module docs for why), calibrate a model against only the train side, and
//! report real, measured out-of-sample error against the held-out test
//! side, compared honestly against the `mid`/`weighted_mid` baselines
//! (never claiming a win that didn't happen).

use std::time::Instant;

use clap::Args;

use microprice_core::{ImbalanceBucketing, SpreadBucketing, StateSpaceConfig};
use microprice_data::{MarketDataSource, SyntheticEventGenerator};
use microprice_eval::{chronological_split, evaluate_model};

use crate::train::{calibrate_model, parse_spread_bounds, CommonTrainArgs};

#[derive(Args, Debug)]
pub struct EvaluateArgs {
    #[command(flatten)]
    common: CommonTrainArgs,

    /// Total number of synthetic events to generate before splitting.
    #[arg(long, default_value_t = 500_000)]
    num_events: u64,

    /// Fraction of the (chronologically ordered) stream used for training;
    /// the remainder is held out for evaluation.
    #[arg(long, default_value_t = 0.7)]
    train_fraction: f64,

    /// Evaluate each prediction against the actual mid price this many
    /// events later (see `microprice_eval::evaluate`'s module docs on why
    /// this horizon choice is disclosed, not the only valid one).
    #[arg(long, default_value_t = 1)]
    horizon: usize,
}

pub fn run(args: EvaluateArgs) -> Result<(), Box<dyn std::error::Error>> {
    let spread_bounds = parse_spread_bounds(&args.common.spread_bucket_bounds)?;
    let imbalance = ImbalanceBucketing::new(args.common.num_imbalance_buckets)?;
    let spread = SpreadBucketing::new(spread_bounds.clone())?;
    let state_space = StateSpaceConfig::new(imbalance, spread);

    let mut generator = SyntheticEventGenerator::new(args.common.synthetic_config())?;

    println!(
        "Generating {} synthetic events (seed {}) for a chronological \
         {:.0}%/{:.0}% train/test split...",
        args.num_events,
        args.common.seed,
        args.train_fraction * 100.0,
        (1.0 - args.train_fraction) * 100.0
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

    let (train_events, test_events) = chronological_split(&events, args.train_fraction)?;
    println!(
        "Chronological split: {} train events, {} held-out test events.",
        train_events.len(),
        test_events.len()
    );

    println!("Calibrating on the train split only...");
    let model = calibrate_model(
        train_events,
        &state_space,
        args.common.symbol_id,
        args.common.num_imbalance_buckets,
        spread_bounds,
        args.common.smoothing_alpha,
    )?;

    println!(
        "Evaluating out-of-sample at horizon={} against {} held-out events...",
        args.horizon,
        test_events.len()
    );
    let report = evaluate_model(&model, test_events, args.horizon)?;

    println!();
    println!("horizon:                   {}", report.horizon);
    println!(
        "n_evaluated / n_skipped:   {} / {}",
        report.n_evaluated, report.n_skipped
    );
    println!();
    println!(
        "MAE (ticks)      microprice={:.4}  mid={:.4}  weighted_mid={:.4}",
        report.microprice_mae, report.mid_mae, report.weighted_mid_mae
    );
    println!(
        "bias (ticks)     microprice={:.4}  (positive = overshoots, negative = undershoots)",
        report.microprice_bias
    );
    if report.n_directional > 0 {
        println!(
            "direction acc.   microprice={:.4}  weighted_mid={:.4}  ({} directional observations)",
            report.microprice_direction_accuracy,
            report.weighted_mid_direction_accuracy,
            report.n_directional
        );
    } else {
        println!("direction acc.   no directional (actual != mid) observations in this test split");
    }
    println!();
    if report.microprice_mae < report.mid_mae {
        println!(
            "-> microprice beat the naive mid baseline on MAE by {:.4} ticks on this run.",
            report.mid_mae - report.microprice_mae
        );
    } else {
        println!(
            "-> microprice did NOT beat the naive mid baseline on MAE on this run \
             (worse by {:.4} ticks) - reporting this honestly, not hiding it.",
            report.microprice_mae - report.mid_mae
        );
    }

    Ok(())
}
