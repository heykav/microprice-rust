//! `microprice predict`: load a saved model and predict the micro-price for
//! one top-of-book snapshot given on the command line.

use std::path::PathBuf;

use clap::Args;

use microprice_calibration::MicroPriceModel;
use microprice_core::{BookValidationPolicy, PriceTicks, Quantity, TopOfBook};

#[derive(Args, Debug)]
pub struct PredictArgs {
    /// Path to a model artifact produced by `microprice train`.
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    bid_price_ticks: i64,
    #[arg(long)]
    bid_qty: u64,
    #[arg(long)]
    ask_price_ticks: i64,
    #[arg(long)]
    ask_qty: u64,
}

pub fn run(args: PredictArgs) -> Result<(), Box<dyn std::error::Error>> {
    let model = MicroPriceModel::load(&args.model)?;
    let book = TopOfBook::new(
        PriceTicks(args.bid_price_ticks),
        Quantity(args.bid_qty),
        PriceTicks(args.ask_price_ticks),
        Quantity(args.ask_qty),
        BookValidationPolicy::RejectCrossedAndLocked,
    )?;
    let estimate = model.predict(&book)?;

    println!("state_id:           {}", estimate.state_id);
    println!("state_observations:  {}", estimate.state_observations);
    println!("mid_ticks:           {:.4}", estimate.mid_ticks);
    println!("weighted_mid_ticks:  {:.4}", estimate.weighted_mid_ticks);
    println!("adjustment_ticks:    {:.6}", estimate.adjustment_ticks);
    println!("microprice_ticks:    {:.6}", estimate.microprice_ticks);
    if estimate.state_observations < 30 {
        println!(
            "warning: this state had only {} training observations - the adjustment \
             estimate for it may be noisy.",
            estimate.state_observations
        );
    }
    Ok(())
}
