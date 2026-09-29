//! Paired comparison of the calibrated micro-price against the two
//! baselines (naive mid, size-weighted mid) at one fixed event horizon,
//! with sample sizes and bootstrap confidence intervals.
//!
//! This is the evaluation prescribed in `docs/real-data-evaluation.md`. It
//! complements [`crate::evaluate::evaluate`] (which reports point estimates
//! only) by answering the question a point estimate cannot: is the
//! difference between two predictors distinguishable from noise on this
//! sample?
//!
//! ## Definitions
//!
//! For each event `i` with an event `i + horizon` in the slice, and whose
//! book the model can encode:
//!
//! * `target_i` = mid-price of event `i + horizon` (exact, in model units);
//! * each predictor's prediction is the model's `mid`, `weighted_mid` or
//!   `microprice` for event `i`;
//! * squared and absolute errors are taken against `target_i`.
//!
//! "Price-changing" observations are those with `target_i != mid_i`.
//! Directional accuracy is computed on those only, over the subset where the
//! predictor's implied direction (`prediction - mid`) is non-zero; a
//! zero-direction prediction is an abstention, counted in the coverage.
//!
//! ## Uncertainty
//!
//! Differences between predictors are paired per observation. Observations
//! overlap (event `i`'s target is also near event `i+1`'s) and are serially
//! dependent, so an i.i.d. bootstrap would be over-confident. The interval
//! here is a **non-overlapping block bootstrap** (resample whole blocks of
//! `block_len` consecutive observations with replacement, recompute the
//! ratio of sums), reported as the 2.5th-97.5th percentile. Choose
//! `block_len >= horizon`; this is an approximation, not an exact coverage
//! guarantee. Directional accuracy uses a Wilson interval that assumes
//! independence and is therefore optimistic under serial dependence.
//!
//! Units: everything is in the units of the prices in the events. Use
//! [`ComparisonReport::rescaled`] to convert (e.g. model units to ticks).

use microprice_calibration::MicroPriceModel;
use microprice_core::BookEvent;

use crate::error::EvalError;

/// Options for [`compare_predictors`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompareOptions {
    /// Event horizon, `>= 1`.
    pub horizon: usize,
    /// Bootstrap resamples; `0` disables intervals (they become `NaN`).
    pub bootstrap_resamples: usize,
    /// Block length in observations, `>= 1`.
    pub block_len: usize,
    /// Seed for the bootstrap's RNG (deterministic given the same input).
    pub seed: u64,
}

/// Point estimate and a 95% percentile interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    pub estimate: f64,
    pub lo: f64,
    pub hi: f64,
}

impl Interval {
    /// True if the interval is computed and excludes zero.
    pub fn excludes_zero(&self) -> bool {
        self.lo.is_finite() && self.hi.is_finite() && (self.lo > 0.0 || self.hi < 0.0)
    }
}

/// Error summary of one predictor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LossStats {
    pub mae: f64,
    pub mse: f64,
    /// Mean of `prediction - target`.
    pub bias: f64,
}

/// Directional accuracy of one predictor on price-changing observations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectionStats {
    /// Price-changing observations where the prediction had a direction.
    pub n_scored: usize,
    /// Correct among `n_scored`. `NaN` if `n_scored == 0`.
    pub accuracy: f64,
    /// Wilson 95% interval for `accuracy`; `NaN` if `n_scored == 0`.
    pub ci_lo: f64,
    pub ci_hi: f64,
}

/// Result of [`compare_predictors`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComparisonReport {
    pub horizon: usize,
    pub n_evaluated: usize,
    /// Candidates skipped because the book could not be encoded.
    pub n_skipped: usize,
    /// Observations whose target differs from the current mid.
    pub n_price_changing: usize,
    /// Share of price-changing observations that moved up. `NaN` if none.
    pub up_share: f64,
    pub mid: LossStats,
    pub weighted_mid: LossStats,
    pub microprice: LossStats,
    /// `microprice - mid` (negative favours the micro-price).
    pub mae_diff_vs_mid: Interval,
    pub mse_diff_vs_mid: Interval,
    /// `microprice - weighted_mid` (negative favours the micro-price).
    pub mae_diff_vs_weighted_mid: Interval,
    pub mse_diff_vs_weighted_mid: Interval,
    pub microprice_direction: DirectionStats,
    pub weighted_mid_direction: DirectionStats,
    pub block_len: usize,
    pub bootstrap_resamples: usize,
}

impl ComparisonReport {
    /// Multiplies every length-valued quantity by `factor` (MSE-valued ones
    /// by `factor^2`). Counts, accuracies and shares are unchanged. Used to
    /// convert model units to exchange ticks (`factor = 1 / resolution`).
    pub fn rescaled(mut self, factor: f64) -> Self {
        let l = |s: LossStats| LossStats {
            mae: s.mae * factor,
            mse: s.mse * factor * factor,
            bias: s.bias * factor,
        };
        let iv = |i: Interval, k: f64| Interval {
            estimate: i.estimate * k,
            lo: i.lo * k,
            hi: i.hi * k,
        };
        self.mid = l(self.mid);
        self.weighted_mid = l(self.weighted_mid);
        self.microprice = l(self.microprice);
        self.mae_diff_vs_mid = iv(self.mae_diff_vs_mid, factor);
        self.mae_diff_vs_weighted_mid = iv(self.mae_diff_vs_weighted_mid, factor);
        self.mse_diff_vs_mid = iv(self.mse_diff_vs_mid, factor * factor);
        self.mse_diff_vs_weighted_mid = iv(self.mse_diff_vs_weighted_mid, factor * factor);
        self
    }
}

/// Runs the paired comparison. `events` must be the held-out test slice, in
/// chronological order (never data the model was calibrated on).
pub fn compare_predictors(
    model: &MicroPriceModel,
    events: &[BookEvent],
    options: CompareOptions,
) -> Result<ComparisonReport, EvalError> {
    if options.horizon == 0 {
        return Err(EvalError::InvalidHorizon { horizon: 0 });
    }
    if options.block_len == 0 {
        return Err(EvalError::InsufficientEvents {
            reason: "block_len must be >= 1".to_string(),
        });
    }
    if events.len() <= options.horizon {
        return Err(EvalError::InsufficientEvents {
            reason: format!(
                "need more than horizon ({}) events, got {}",
                options.horizon,
                events.len()
            ),
        });
    }

    // Per-observation error vectors, in chronological order.
    let mut e_mid = Vec::new();
    let mut e_wmid = Vec::new();
    let mut e_micro = Vec::new();
    let mut dir_actual = Vec::new();
    let mut dir_micro = Vec::new();
    let mut dir_wmid = Vec::new();
    let mut n_skipped = 0usize;

    for i in 0..events.len() - options.horizon {
        // Exact doubled mid: `(bid + ask) / 2` computed in f64, not the
        // truncating integer `mid_price_ticks` (see the csv module docs).
        let target = exact_mid(&events[i + options.horizon]);
        match model.predict(&events[i].book) {
            Ok(est) => {
                // `est.mid_ticks` is the truncating integer mid; recompute
                // the exact one so all three predictors share one definition.
                let mid = exact_mid(&events[i]);
                let micro = mid + est.adjustment_ticks;
                let wmid = est.weighted_mid_ticks;
                e_mid.push(mid - target);
                e_wmid.push(wmid - target);
                e_micro.push(micro - target);
                dir_actual.push(target - mid);
                dir_micro.push(micro - mid);
                dir_wmid.push(wmid - mid);
            }
            Err(_) => n_skipped += 1,
        }
    }
    let n = e_mid.len();
    if n == 0 {
        return Err(EvalError::InsufficientEvents {
            reason: "every candidate observation failed state encoding".to_string(),
        });
    }

    let stats = |e: &[f64]| LossStats {
        mae: e.iter().map(|x| x.abs()).sum::<f64>() / n as f64,
        mse: e.iter().map(|x| x * x).sum::<f64>() / n as f64,
        bias: e.iter().sum::<f64>() / n as f64,
    };

    let abs_diff = |a: &[f64], b: &[f64]| -> Vec<f64> {
        a.iter().zip(b).map(|(x, y)| x.abs() - y.abs()).collect()
    };
    let sq_diff = |a: &[f64], b: &[f64]| -> Vec<f64> {
        a.iter().zip(b).map(|(x, y)| x * x - y * y).collect()
    };
    let bs = |d: &[f64]| block_bootstrap_mean(d, &options);

    let n_price_changing = dir_actual.iter().filter(|d| **d != 0.0).count();
    let n_up = dir_actual.iter().filter(|d| **d > 0.0).count();
    let up_share = if n_price_changing == 0 {
        f64::NAN
    } else {
        n_up as f64 / n_price_changing as f64
    };

    Ok(ComparisonReport {
        horizon: options.horizon,
        n_evaluated: n,
        n_skipped,
        n_price_changing,
        up_share,
        mid: stats(&e_mid),
        weighted_mid: stats(&e_wmid),
        microprice: stats(&e_micro),
        mae_diff_vs_mid: bs(&abs_diff(&e_micro, &e_mid)),
        mse_diff_vs_mid: bs(&sq_diff(&e_micro, &e_mid)),
        mae_diff_vs_weighted_mid: bs(&abs_diff(&e_micro, &e_wmid)),
        mse_diff_vs_weighted_mid: bs(&sq_diff(&e_micro, &e_wmid)),
        microprice_direction: direction_stats(&dir_micro, &dir_actual),
        weighted_mid_direction: direction_stats(&dir_wmid, &dir_actual),
        block_len: options.block_len,
        bootstrap_resamples: options.bootstrap_resamples,
    })
}

fn exact_mid(e: &BookEvent) -> f64 {
    (e.book.bid_price.0 as f64 + e.book.ask_price.0 as f64) / 2.0
}

fn direction_stats(predicted: &[f64], actual: &[f64]) -> DirectionStats {
    let mut scored = 0usize;
    let mut correct = 0usize;
    for (p, a) in predicted.iter().zip(actual) {
        if *a == 0.0 || *p == 0.0 {
            continue;
        }
        scored += 1;
        if (*p > 0.0) == (*a > 0.0) {
            correct += 1;
        }
    }
    if scored == 0 {
        return DirectionStats {
            n_scored: 0,
            accuracy: f64::NAN,
            ci_lo: f64::NAN,
            ci_hi: f64::NAN,
        };
    }
    let n = scored as f64;
    let p = correct as f64 / n;
    let z = 1.96f64;
    let denom = 1.0 + z * z / n;
    let centre = (p + z * z / (2.0 * n)) / denom;
    let half = z * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt() / denom;
    DirectionStats {
        n_scored: scored,
        accuracy: p,
        ci_lo: centre - half,
        ci_hi: centre + half,
    }
}

/// splitmix64: a tiny, well-known, deterministic generator. Enough for
/// choosing bootstrap block indices; not for anything cryptographic.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Mean of `values` with a non-overlapping block-bootstrap 95% interval.
fn block_bootstrap_mean(values: &[f64], options: &CompareOptions) -> Interval {
    let n = values.len();
    let estimate = values.iter().sum::<f64>() / n as f64;
    if options.bootstrap_resamples == 0 {
        return Interval {
            estimate,
            lo: f64::NAN,
            hi: f64::NAN,
        };
    }
    // (sum, len) per block; the final block may be shorter.
    let blocks: Vec<(f64, usize)> = values
        .chunks(options.block_len)
        .map(|c| (c.iter().sum::<f64>(), c.len()))
        .collect();
    let nb = blocks.len();
    let mut rng = SplitMix64(options.seed);
    let mut means = Vec::with_capacity(options.bootstrap_resamples);
    for _ in 0..options.bootstrap_resamples {
        let mut s = 0.0;
        let mut l = 0usize;
        for _ in 0..nb {
            let (bs, bl) = blocks[(rng.next() % nb as u64) as usize];
            s += bs;
            l += bl;
        }
        means.push(s / l as f64);
    }
    means.sort_by(|a, b| a.total_cmp(b));
    let q = |p: f64| means[(((means.len() - 1) as f64) * p).round() as usize];
    Interval {
        estimate,
        lo: q(0.025),
        hi: q(0.975),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use microprice_calibration::ModelMetadata;
    use microprice_core::{BookValidationPolicy, PriceTicks, Quantity, SymbolId, TopOfBook};

    fn model() -> MicroPriceModel {
        // 2 imbalance buckets, one spread bucket; balanced books -> state 1.
        MicroPriceModel::new(
            ModelMetadata {
                schema_version: microprice_calibration::SCHEMA_VERSION,
                symbol_id: 1,
                num_imbalance_buckets: 2,
                spread_bucket_bounds_ticks: vec![],
                smoothing_alpha: 0.0,
                training_observations: 100,
            },
            vec![0.2, 0.225],
            vec![Some(0.5), Some(0.5)],
            vec![100, 50],
        )
    }

    fn ev(seq: u64, mid: i64) -> BookEvent {
        BookEvent {
            timestamp_ns: seq * 1000,
            sequence: seq,
            symbol: SymbolId(1),
            book: TopOfBook::new(
                PriceTicks(mid - 1),
                Quantity(100),
                PriceTicks(mid + 1),
                Quantity(100),
                BookValidationPolicy::RejectCrossedAndLocked,
            )
            .unwrap(),
        }
    }

    fn opts(horizon: usize, resamples: usize, block_len: usize) -> CompareOptions {
        CompareOptions {
            horizon,
            bootstrap_resamples: resamples,
            block_len,
            seed: 7,
        }
    }

    #[test]
    fn matches_hand_computed_losses_and_paired_differences() {
        // mids 10001, 10002, 10001; adjustment +0.225 in the balanced state.
        // micro errors: 10001.225-10002 = -0.775 ; 10002.225-10001 = +1.225
        // mid errors:   -1 ; +1. weighted mid == mid (balanced queues).
        let events = vec![ev(0, 10001), ev(1, 10002), ev(2, 10001)];
        let r = compare_predictors(&model(), &events, opts(1, 0, 1)).unwrap();
        assert_eq!(r.n_evaluated, 2);
        assert_eq!(r.n_skipped, 0);
        assert!((r.mid.mae - 1.0).abs() < 1e-12);
        assert!((r.mid.mse - 1.0).abs() < 1e-12);
        assert!((r.microprice.mae - 1.0).abs() < 1e-12);
        // (0.775^2 + 1.225^2) / 2 = (0.600625 + 1.500625) / 2
        assert!((r.microprice.mse - 1.050625).abs() < 1e-12);
        assert!((r.mae_diff_vs_mid.estimate - 0.0).abs() < 1e-12);
        assert!((r.mse_diff_vs_mid.estimate - 0.050625).abs() < 1e-12);
        assert!((r.mse_diff_vs_weighted_mid.estimate - 0.050625).abs() < 1e-12);
        assert!((r.microprice.bias - 0.225).abs() < 1e-12);
        // Two price-changing observations, one up and one down.
        assert_eq!(r.n_price_changing, 2);
        assert!((r.up_share - 0.5).abs() < 1e-12);
        // Micro always predicts up: right once of twice.
        assert_eq!(r.microprice_direction.n_scored, 2);
        assert!((r.microprice_direction.accuracy - 0.5).abs() < 1e-12);
        // Weighted mid == mid -> zero implied direction -> abstains always.
        assert_eq!(r.weighted_mid_direction.n_scored, 0);
        assert!(r.weighted_mid_direction.accuracy.is_nan());
        // Intervals disabled with 0 resamples.
        assert!(r.mse_diff_vs_mid.lo.is_nan());
    }

    #[test]
    fn a_half_tick_mid_is_not_truncated() {
        // bid 10000 / ask 10001: exact mid 10000.5 (integer mid would be 10000).
        let e = |seq: u64, bid: i64| BookEvent {
            timestamp_ns: seq,
            sequence: seq,
            symbol: SymbolId(1),
            book: TopOfBook::new(
                PriceTicks(bid),
                Quantity(100),
                PriceTicks(bid + 1),
                Quantity(100),
                BookValidationPolicy::RejectCrossedAndLocked,
            )
            .unwrap(),
        };
        let events = vec![e(0, 10000), e(1, 10000)];
        let r = compare_predictors(&model(), &events, opts(1, 0, 1)).unwrap();
        // Target and mid are both 10000.5 -> the naive mid is exactly right.
        assert_eq!(r.mid.mae, 0.0);
        assert_eq!(r.n_price_changing, 0);
    }

    #[test]
    fn bootstrap_is_deterministic_and_brackets_the_estimate() {
        let events: Vec<_> = (0..400)
            .map(|i| ev(i, 10_000 + ((i * 7) % 5) as i64))
            .collect();
        let a = compare_predictors(&model(), &events, opts(2, 200, 20)).unwrap();
        let b = compare_predictors(&model(), &events, opts(2, 200, 20)).unwrap();
        assert_eq!(a.mse_diff_vs_mid, b.mse_diff_vs_mid);
        assert_eq!(a.mae_diff_vs_mid, b.mae_diff_vs_mid);
        let iv = a.mse_diff_vs_mid;
        assert!(iv.lo <= iv.hi);
        assert!(iv.lo.is_finite() && iv.hi.is_finite());
    }

    #[test]
    fn identical_predictors_have_a_degenerate_zero_interval() {
        // A model whose adjustment is exactly 0 predicts the mid exactly.
        let zero = MicroPriceModel::new(
            ModelMetadata {
                schema_version: microprice_calibration::SCHEMA_VERSION,
                symbol_id: 1,
                num_imbalance_buckets: 2,
                spread_bucket_bounds_ticks: vec![],
                smoothing_alpha: 0.0,
                training_observations: 1,
            },
            vec![0.0, 0.0],
            vec![None, None],
            vec![1, 1],
        );
        let events: Vec<_> = (0..100).map(|i| ev(i, 10_000 + (i % 3) as i64)).collect();
        let r = compare_predictors(&zero, &events, opts(1, 100, 10)).unwrap();
        assert_eq!(r.mse_diff_vs_mid.estimate, 0.0);
        assert_eq!(r.mse_diff_vs_mid.lo, 0.0);
        assert_eq!(r.mse_diff_vs_mid.hi, 0.0);
        assert!(!r.mse_diff_vs_mid.excludes_zero());
    }

    #[test]
    fn rescaling_converts_units_consistently() {
        let events = vec![ev(0, 10001), ev(1, 10002), ev(2, 10001)];
        let r = compare_predictors(&model(), &events, opts(1, 0, 1)).unwrap();
        let s = r.rescaled(0.5);
        assert!((s.mid.mae - 0.5).abs() < 1e-12);
        assert!((s.mid.mse - 0.25).abs() < 1e-12);
        assert_eq!(s.n_evaluated, r.n_evaluated);
        assert_eq!(s.microprice_direction, r.microprice_direction);
    }

    #[test]
    fn invalid_inputs_are_typed_errors() {
        let events = vec![ev(0, 10001), ev(1, 10002)];
        assert!(compare_predictors(&model(), &events, opts(0, 0, 1)).is_err());
        assert!(compare_predictors(&model(), &events, opts(1, 0, 0)).is_err());
        assert!(compare_predictors(&model(), &events, opts(2, 0, 1)).is_err());
    }

    #[test]
    fn wilson_interval_for_a_known_proportion() {
        // 50 of 100 correct -> Wilson 95% is roughly [0.404, 0.596].
        let pred: Vec<f64> = (0..100).map(|_| 1.0).collect();
        let act: Vec<f64> = (0..100).map(|i| if i < 50 { 1.0 } else { -1.0 }).collect();
        let d = direction_stats(&pred, &act);
        assert_eq!(d.n_scored, 100);
        assert!((d.ci_lo - 0.4038).abs() < 1e-3);
        assert!((d.ci_hi - 0.5962).abs() < 1e-3);
    }
}
