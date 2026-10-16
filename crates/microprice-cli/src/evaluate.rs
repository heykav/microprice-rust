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
use microprice_eval::{
    calibration_by_state, chronological_split, compare_predictors,
    compare_predictors_next_mid_change, evaluate_model, resolve_next_mid_change_targets,
    CompareOptions, ComparisonReport, Interval,
};

use crate::train::{calibrate_model, parse_spread_bounds, print_diagnostics, CommonTrainArgs};

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

    /// Block-bootstrap resamples for the 95% intervals of the paired
    /// comparison printed at the end (0 disables the intervals).
    #[arg(long, default_value_t = 1000)]
    bootstrap_resamples: usize,

    /// Also print a per-state reliability table on the held-out split:
    /// predicted G* next to the mean realized move to the next mid change.
    #[arg(long)]
    calibration_table: bool,
}

fn fmt_interval(i: Interval) -> String {
    if i.lo.is_finite() && i.hi.is_finite() {
        format!("{:+.5} [{:+.5}, {:+.5}]", i.estimate, i.lo, i.hi)
    } else {
        format!("{:+.5} (no interval)", i.estimate)
    }
}

fn print_paired(label: &str, r: &ComparisonReport) {
    println!("  {label}");
    println!(
        "    n={}  mean events ahead={:.2}  unresolved={}  block_len={}",
        r.n_evaluated, r.mean_events_ahead, r.n_unresolved, r.block_len
    );
    println!(
        "    MAE  microprice={:.5}  mid={:.5}  weighted_mid={:.5}",
        r.microprice.mae, r.mid.mae, r.weighted_mid.mae
    );
    println!(
        "    MSE  microprice={:.5}  mid={:.5}  weighted_mid={:.5}",
        r.microprice.mse, r.mid.mse, r.weighted_mid.mse
    );
    println!(
        "    microprice - mid:  MAE {}   MSE {}",
        fmt_interval(r.mae_diff_vs_mid),
        fmt_interval(r.mse_diff_vs_mid)
    );
}

pub fn run(args: EvaluateArgs) -> Result<(), Box<dyn std::error::Error>> {
    let spread_bounds = parse_spread_bounds(&args.common.spread_bucket_bounds)?;
    let imbalance = ImbalanceBucketing::new(args.common.num_imbalance_buckets)?;
    let spread = SpreadBucketing::new(spread_bounds.clone())?;
    let state_space = StateSpaceConfig::new(imbalance, spread)?;

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
    let calibrated = calibrate_model(
        train_events,
        &state_space,
        args.common.symbol_id,
        args.common.num_imbalance_buckets,
        spread_bounds,
        args.common.smoothing_alpha,
        args.common.symmetrize,
    )?;
    print_diagnostics(&calibrated, "ticks", 1.0);
    let model = calibrated.model;

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
    if report.n_probabilistic > 0 {
        println!(
            "Brier score      microprice={:.4}  climatology={:.4}  skill={:+.4}  ({} probabilistic observations)",
            report.microprice_brier,
            report.brier_baseline,
            report.brier_skill_score,
            report.n_probabilistic
        );
        println!(
            "                 (lower is better; 0.25 = an uninformative coin flip; skill > 0 means \
             the model's P(up) beat always predicting this split's own up-rate)"
        );
    } else {
        println!(
            "Brier score      none computable: no test observation both landed in a state with \
             directional training evidence and actually moved"
        );
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

    // Paired comparison on the same held-out split: MSE alongside MAE, with
    // block-bootstrap intervals, at the fixed event horizon and at the next
    // mid change. MAE rewards the conditional *median*, which is the
    // unchanged mid whenever most short-horizon targets did not move; `G*`
    // estimates a conditional *mean*, which MSE scores. Neither choice is a
    // pre-registered decision rule; both are printed, whatever they show.
    let options = CompareOptions {
        horizon: args.horizon,
        bootstrap_resamples: args.bootstrap_resamples,
        block_len: 1000.max(10 * args.horizon),
        seed: args.common.seed,
    };
    let fixed = compare_predictors(&model, test_events, options)?;
    let next_change = compare_predictors_next_mid_change(
        &model,
        test_events,
        CompareOptions {
            block_len: 0,
            ..options
        },
    )?;
    println!();
    println!(
        "Paired comparison, ticks, 95% block-bootstrap intervals ({} resamples; negative differences favour the microprice):",
        args.bootstrap_resamples
    );
    print_paired(
        &format!("target = mid {} event(s) ahead", args.horizon),
        &fixed,
    );
    print_paired(
        "target = mid at the next mid change (additional; the quantity G* estimates by construction)",
        &next_change,
    );

    if args.calibration_table {
        let (pairs, _) = resolve_next_mid_change_targets(test_events);
        println!();
        println!(
            "Reliability by state (held-out; realized = mid at next mid change - mid, ticks; \
             se assumes independence and understates uncertainty):"
        );
        println!("  state  train_visits  n_test  predicted_gstar  realized_mean  realized_se");
        for row in calibration_by_state(&model, test_events, &pairs) {
            println!(
                "  {:>5}  {:>12}  {:>6}  {:>+15.5}  {:>+13.5}  {:>11.5}",
                row.state_id,
                row.training_visits,
                row.n,
                row.predicted,
                row.realized_mean,
                row.realized_naive_se
            );
        }
    }

    Ok(())
}
