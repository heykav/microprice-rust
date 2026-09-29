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
    let p_up = model.p_up();
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
    let states_with_p_up = p_up.iter().filter(|p| p.is_some()).count();

    println!();
    println!("state_count:                {state_count}");
    println!("g_star range:               [{min_g:.6}, {max_g:.6}], mean {mean_g:.6}");
    if let Ok(res) =
        microprice_calibration::antisymmetry_residual(g_star, meta.num_imbalance_buckets)
    {
        println!(
            "antisymmetry residual:      {res:.3e} (max |G*[s] + G*[mirror(s)]|; ~0 iff calibrated with --symmetrize)"
        );
    }
    println!("visits range:               [{min_visits}, {max_visits}]");
    println!("states with zero visits:    {zero_visit_states} / {state_count}");
    if zero_visit_states > 0 {
        println!(
            "  (a zero-visit state's g_star came entirely from smoothing's prior, \
             not real data - see docs/model-spec.md's Smoothing section)"
        );
    }
    println!("states with a P(up):        {states_with_p_up} / {state_count}");
    if states_with_p_up > 0 {
        let directional: Vec<f64> = p_up.iter().filter_map(|p| *p).collect();
        let mean_p = directional.iter().sum::<f64>() / directional.len() as f64;
        let n_above_half = directional.iter().filter(|&&p| p > 0.5).count();
        println!(
            "  P(up) range:              [{:.6}, {:.6}], mean {mean_p:.6}",
            directional.iter().copied().fold(f64::INFINITY, f64::min),
            directional
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max)
        );
        println!(
            "  states leaning up:        {n_above_half} / {} (P(up) > 0.5)",
            directional.len()
        );
    }
    if states_with_p_up < state_count {
        println!(
            "  ({} state(s) have no P(up): training never saw the price move out of them, so \
             there is no directional evidence to give a probability for)",
            state_count - states_with_p_up
        );
    }

    Ok(())
}
