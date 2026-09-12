//! `microprice inspect`: load a saved model and print its metadata plus a
//! summary of what was actually calibrated (not just what was configured).

use std::path::PathBuf;

use clap::Args;

use microprice_calibration::MicroPriceModel;

#[derive(Args, Debug)]
pub struct InspectArgs {
    /// Path to a model artifact produced by `microprice train`.
    #[arg(long)]
    model: PathBuf,
}

pub fn run(args: InspectArgs) -> Result<(), Box<dyn std::error::Error>> {
    let model = MicroPriceModel::load(&args.model)?;
    let meta = model.metadata();

    println!("schema_version:             {}", meta.schema_version);
    println!("symbol_id:                  {}", meta.symbol_id);
    println!("num_imbalance_buckets:      {}", meta.num_imbalance_buckets);
    println!(
        "spread_bucket_bounds_ticks: {:?}",
        meta.spread_bucket_bounds_ticks
    );
    println!("smoothing_alpha:            {}", meta.smoothing_alpha);
    println!("training_observations:      {}", meta.training_observations);

    let g_star = model.g_star();
    let visits = model.visits();
    let state_count = g_star.len();

    let (min_g, max_g) = g_star
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    let mean_g = g_star.iter().sum::<f64>() / state_count as f64;
    let zero_visit_states = visits.iter().filter(|&&v| v == 0).count();
    let min_visits = visits.iter().min().copied().unwrap_or(0);
    let max_visits = visits.iter().max().copied().unwrap_or(0);

    println!();
    println!("state_count:                {state_count}");
    println!("g_star range:               [{min_g:.6}, {max_g:.6}], mean {mean_g:.6}");
    println!("visits range:               [{min_visits}, {max_visits}]");
    println!("states with zero visits:    {zero_visit_states} / {state_count}");
    if zero_visit_states > 0 {
        println!(
            "  (a zero-visit state's g_star came entirely from smoothing's prior, \
             not real data - see docs/model-spec.md's Smoothing section)"
        );
    }

    Ok(())
}
