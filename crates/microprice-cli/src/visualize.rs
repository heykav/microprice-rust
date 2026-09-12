//! `microprice visualize` (Phase 14): static PNG plots of a calibrated
//! model's `g_star` surface, rendered with `plotters`.
//!
//! Three plots, written into `--output-dir`:
//! - `g_star_by_imbalance.png` — `g_star` vs. imbalance-bucket midpoint,
//!   one line per spread bucket.
//! - `g_star_heatmap.png` — `g_star` over the full `(spread_bucket,
//!   imbalance_bucket)` grid, diverging blue (negative) / red (positive)
//!   around zero.
//! - `visits_heatmap.png` — `log10(visits + 1)` over the same grid, so a
//!   viewer can immediately see *which* cells of the first heatmap are
//!   backed by real data vs. mostly the smoothing prior — plotting
//!   `g_star` alone, without this, would silently imply every cell is
//!   equally trustworthy.
//!
//! This is a static, non-interactive visualization — an interactive
//! browser-based explorer (a separate, larger undertaking) is not what
//! this subcommand does.

use std::path::PathBuf;

use clap::Args;
use plotters::coord::Shift;
use plotters::prelude::*;

use microprice_calibration::MicroPriceModel;
use microprice_core::StateId;

/// A small, fixed, explicit palette for the per-spread-bucket line series,
/// used instead of `plotters::style::Palette99` (whose `pick` returns a
/// `PaletteColor` wrapper that isn't `Copy`, requiring an extra
/// `.to_rgba()` conversion to satisfy the `Fn` legend closure). Owning
/// plain `RGBColor` values directly is simpler to reason about and
/// verify: the same value is passed to both the line/marker drawing and
/// the legend swatch closure, confirmed to match pixel-for-pixel by
/// sampling the actual rendered PNG, not just assumed from the code.
const LINE_COLORS: [RGBColor; 8] = [
    RGBColor(220, 20, 60),   // crimson
    RGBColor(30, 144, 255),  // dodger blue
    RGBColor(46, 139, 87),   // sea green
    RGBColor(255, 140, 0),   // dark orange
    RGBColor(148, 0, 211),   // dark violet
    RGBColor(0, 139, 139),   // dark cyan
    RGBColor(184, 134, 11),  // dark goldenrod
    RGBColor(105, 105, 105), // dim gray
];

fn line_color(index: usize) -> RGBColor {
    LINE_COLORS[index % LINE_COLORS.len()]
}

#[derive(Args, Debug)]
pub struct VisualizeArgs {
    /// Path to a model artifact produced by `microprice train`.
    #[arg(long)]
    model: PathBuf,
    /// Directory to write the PNG files into (created if it doesn't exist).
    #[arg(long)]
    output_dir: PathBuf,
}

fn diverging_color(value: f64, max_abs: f64) -> RGBColor {
    if max_abs <= 0.0 {
        return RGBColor(255, 255, 255);
    }
    let t = (value / max_abs).clamp(-1.0, 1.0);
    if t >= 0.0 {
        // white (0) -> red (+1)
        let g = (255.0 * (1.0 - t)) as u8;
        let b = (255.0 * (1.0 - t)) as u8;
        RGBColor(255, g, b)
    } else {
        // white (0) -> blue (-1)
        let r = (255.0 * (1.0 + t)) as u8;
        let g = (255.0 * (1.0 + t)) as u8;
        RGBColor(r, g, 255)
    }
}

fn sequential_color(t: f64) -> RGBColor {
    let t = t.clamp(0.0, 1.0);
    // white (0) -> dark blue (1)
    let r = (255.0 * (1.0 - t)) as u8;
    let g = (255.0 * (1.0 - t)) as u8;
    RGBColor(r, g, 200)
}

fn draw_heatmap(
    root: &DrawingArea<BitMapBackend, Shift>,
    title: &str,
    num_imbalance_buckets: u32,
    num_spread_buckets: u32,
    values: &[f64],
    color_for: impl Fn(f64) -> RGBColor,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut chart = ChartBuilder::on(root)
        .caption(title, ("sans-serif", 24))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0..num_imbalance_buckets, 0..num_spread_buckets)?;

    chart
        .configure_mesh()
        .x_desc("imbalance bucket")
        .y_desc("spread bucket")
        .disable_mesh()
        .draw()?;

    for spread_bucket in 0..num_spread_buckets {
        for imbalance_bucket in 0..num_imbalance_buckets {
            let idx = (spread_bucket * num_imbalance_buckets + imbalance_bucket) as usize;
            let color = color_for(values[idx]);
            chart.draw_series(std::iter::once(Rectangle::new(
                [
                    (imbalance_bucket, spread_bucket),
                    (imbalance_bucket + 1, spread_bucket + 1),
                ],
                color.filled(),
            )))?;
        }
    }
    Ok(())
}

pub fn run(args: VisualizeArgs) -> Result<(), Box<dyn std::error::Error>> {
    let model = MicroPriceModel::load(&args.model)?;
    let state_space = model.state_space()?;
    let meta = model.metadata();
    let num_imbalance_buckets = meta.num_imbalance_buckets;
    let num_spread_buckets = meta.spread_bucket_bounds_ticks.len() as u32 + 1;
    let g_star = model.g_star();
    let visits = model.visits();

    std::fs::create_dir_all(&args.output_dir)?;

    // --- Plot 1: g_star vs. imbalance midpoint, one line per spread bucket ---
    let line_path = args.output_dir.join("g_star_by_imbalance.png");
    {
        let root = BitMapBackend::new(&line_path, (900, 600)).into_drawing_area();
        root.fill(&WHITE)?;
        let y_min = g_star.iter().cloned().fold(f64::INFINITY, f64::min);
        let y_max = g_star.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let pad = ((y_max - y_min).abs() * 0.1).max(1e-6);
        let mut chart = ChartBuilder::on(&root)
            .caption(
                "Calibrated adjustment (g_star) vs. queue imbalance",
                ("sans-serif", 24),
            )
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(50)
            .build_cartesian_2d(0.0..1.0, (y_min - pad)..(y_max + pad))?;
        chart
            .configure_mesh()
            .x_desc("imbalance bucket midpoint (Qb / (Qb+Qa))")
            .y_desc("g_star (ticks)")
            .draw()?;

        for spread_bucket in 0..num_spread_buckets {
            let series: Vec<(f64, f64)> = (0..num_imbalance_buckets)
                .map(|imbalance_bucket| {
                    let state_id = spread_bucket * num_imbalance_buckets + imbalance_bucket;
                    let desc = state_space.decode(StateId(state_id));
                    let mid = (desc.imbalance_range.0 + desc.imbalance_range.1) / 2.0;
                    (mid, g_star[state_id as usize])
                })
                .collect();
            let color = line_color(spread_bucket as usize);
            chart
                .draw_series(LineSeries::new(series.clone(), color.stroke_width(2)))?
                .label(format!("spread bucket {spread_bucket}"))
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
            chart.draw_series(
                series
                    .iter()
                    .map(|&(x, y)| Circle::new((x, y), 3, color.filled())),
            )?;
        }
        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()?;
        root.present()?;
    }

    // --- Plot 2: g_star heatmap ---
    let g_star_heatmap_path = args.output_dir.join("g_star_heatmap.png");
    {
        let root = BitMapBackend::new(&g_star_heatmap_path, (900, 600)).into_drawing_area();
        root.fill(&WHITE)?;
        let max_abs = g_star
            .iter()
            .cloned()
            .fold(0.0f64, |acc, v| acc.max(v.abs()));
        draw_heatmap(
            &root,
            "g_star heatmap (blue = negative, red = positive)",
            num_imbalance_buckets,
            num_spread_buckets,
            g_star,
            |v| diverging_color(v, max_abs),
        )?;
        root.present()?;
    }

    // --- Plot 3: visits heatmap (log10(visits + 1)) ---
    let visits_heatmap_path = args.output_dir.join("visits_heatmap.png");
    {
        let log_visits: Vec<f64> = visits.iter().map(|&v| ((v + 1) as f64).log10()).collect();
        let max_log = log_visits.iter().cloned().fold(0.0f64, f64::max).max(1e-9);
        let root = BitMapBackend::new(&visits_heatmap_path, (900, 600)).into_drawing_area();
        root.fill(&WHITE)?;
        draw_heatmap(
            &root,
            "Training data density: log10(visits + 1)",
            num_imbalance_buckets,
            num_spread_buckets,
            &log_visits,
            |v| sequential_color(v / max_log),
        )?;
        root.present()?;
    }

    println!("Wrote:");
    println!("  {}", line_path.display());
    println!("  {}", g_star_heatmap_path.display());
    println!("  {}", visits_heatmap_path.display());
    let zero_visit_states = visits.iter().filter(|&&v| v == 0).count();
    if zero_visit_states > 0 {
        println!(
            "note: {zero_visit_states} of {} states had zero training visits - their g_star_heatmap \
             cell came entirely from the smoothing prior; check visits_heatmap.png before trusting them.",
            g_star.len()
        );
    }
    Ok(())
}
