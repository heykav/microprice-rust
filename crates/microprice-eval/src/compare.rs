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
    /// Event horizon; `0` for a wall-clock report (see `horizon_ns`) or a
    /// next-mid-change report (see `next_mid_change`).
    pub horizon: usize,
    /// `true` for a report from [`compare_predictors_next_mid_change`]: the
    /// target was the mid of the first later event whose mid differs.
    pub next_mid_change: bool,
    /// `Some(T)` for a wall-clock horizon of `T` nanoseconds, else `None`.
    pub horizon_ns: Option<u64>,
    /// Wall-clock only: candidates dropped because the data ends before
    /// `t_i + T` (the prevailing quote at `t_i + T` is unknown). `0` for
    /// event horizons.
    pub n_unresolved: usize,
    /// Mean number of events between the prediction and its target quote
    /// (`horizon` for event horizons; varies for wall-clock ones).
    pub mean_events_ahead: f64,
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

/// Runs the paired comparison at an **event** horizon. `events` must be the
/// held-out test slice, in chronological order (never data the model was
/// calibrated on).
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
    let pairs: Vec<(usize, usize)> = (0..events.len() - options.horizon)
        .map(|i| (i, i + options.horizon))
        .collect();
    compare_pairs(model, events, &pairs, options, None, 0, false)
}

/// Resolves, for each candidate event `i`, the index of the quote
/// prevailing at wall-clock time `timestamps[i] + horizon_ns` (see
/// [`compare_predictors_wall_clock`] for the exact rule). Returns the
/// `(i, target)` pairs and the number of unresolved candidates.
///
/// `timestamps` must be non-decreasing and `horizon_ns >= 1`.
pub fn resolve_wall_clock_targets(
    timestamps: &[u64],
    horizon_ns: u64,
) -> Result<(Vec<(usize, usize)>, usize), EvalError> {
    if horizon_ns == 0 {
        return Err(EvalError::InsufficientEvents {
            reason: "wall-clock horizon must be >= 1 ns".to_string(),
        });
    }
    if let Some(w) = timestamps.windows(2).find(|w| w[1] < w[0]) {
        return Err(EvalError::InsufficientEvents {
            reason: format!(
                "timestamps must be non-decreasing (saw {} after {})",
                w[1], w[0]
            ),
        });
    }
    let n = timestamps.len();
    let Some(&last_ts) = timestamps.last() else {
        return Ok((Vec::new(), 0));
    };
    let mut pairs = Vec::new();
    let mut unresolved = 0usize;
    let mut j = 0usize;
    for (i, &t) in timestamps.iter().enumerate() {
        // An overflowing deadline can never be reached by any timestamp.
        let Some(deadline) = t.checked_add(horizon_ns) else {
            unresolved += 1;
            continue;
        };
        if last_ts < deadline {
            unresolved += 1;
            continue;
        }
        if j < i {
            j = i;
        }
        // Last event with timestamp <= deadline (inclusive boundary; ties in
        // timestamp resolve to the LAST event at that timestamp).
        while j + 1 < n && timestamps[j + 1] <= deadline {
            j += 1;
        }
        pairs.push((i, j));
    }
    Ok((pairs, unresolved))
}

/// Runs the paired comparison at a **wall-clock** horizon of `horizon_ns`
/// nanoseconds, using each event's `timestamp_ns`.
///
/// ## Rule (fixed here, tested, and documented in
/// `docs/real-data-evaluation.md`)
///
/// For the prediction made at event `i` (time `t_i`), the target is the
/// mid of the quote **prevailing at time `t_i + T`**: the last event whose
/// timestamp is `<= t_i + T`. Consequently:
///
/// * a quote stamped **exactly** `t_i + T` is *included* (boundary is
///   closed on the right); one stamped `t_i + T + 1 ns` is not;
/// * if several events share the timestamp of the prevailing quote, the
///   **last** of them is used (the book's state at the end of that
///   instant);
/// * if no event falls in `(t_i, t_i + T]` the prevailing quote is event
///   `i` itself, the target mid equals the current mid, and the
///   observation is kept (the quote did not change, which is information);
/// * a candidate is **dropped and counted** in `n_unresolved` if the last
///   event in the slice is stamped before `t_i + T`: past the end of data
///   the prevailing quote is unknown, and assuming it persisted would bias
///   toward "no change";
/// * events with the same timestamp as `t_i` that follow `i` count as
///   happening within the window (they are `<= t_i + T`).
///
/// Timestamps must be non-decreasing. `options.horizon` is ignored;
/// `options.block_len == 0` means "auto": `max(1000, 10 * ceil(mean events
/// ahead))`. Wall-clock horizons are an additional, non-pre-registered
/// analysis (`docs/real-data-evaluation.md`, amendments).
pub fn compare_predictors_wall_clock(
    model: &MicroPriceModel,
    events: &[BookEvent],
    horizon_ns: u64,
    options: CompareOptions,
) -> Result<ComparisonReport, EvalError> {
    let ts: Vec<u64> = events.iter().map(|e| e.timestamp_ns).collect();
    let (pairs, unresolved) = resolve_wall_clock_targets(&ts, horizon_ns)?;
    if pairs.is_empty() {
        return Err(EvalError::InsufficientEvents {
            reason: format!(
                "no candidate has data through t + {horizon_ns} ns ({unresolved} unresolved)"
            ),
        });
    }
    let mut options = options;
    if options.block_len == 0 {
        let mean_ahead =
            pairs.iter().map(|(i, j)| (j - i) as f64).sum::<f64>() / pairs.len() as f64;
        options.block_len = 1000.max(10 * mean_ahead.ceil() as usize);
    }
    compare_pairs(
        model,
        events,
        &pairs,
        options,
        Some(horizon_ns),
        unresolved,
        false,
    )
}

/// For each event `i`, the index of the first later event whose exact mid
/// (`(bid + ask) / 2`) differs from event `i`'s. Returns the `(i, target)`
/// pairs and the number of trailing candidates with no later mid change in
/// the slice (dropped, not assumed unchanged). `O(n)`.
pub fn resolve_next_mid_change_targets(events: &[BookEvent]) -> (Vec<(usize, usize)>, usize) {
    let n = events.len();
    // Doubled mids are exact integers, so equality is exact.
    let mid2 = |e: &BookEvent| i128::from(e.book.bid_price.0) + i128::from(e.book.ask_price.0);
    let mut next: Vec<Option<usize>> = vec![None; n];
    for i in (0..n.saturating_sub(1)).rev() {
        next[i] = if mid2(&events[i + 1]) != mid2(&events[i]) {
            Some(i + 1)
        } else {
            // Event i+1 has the same mid as i, so its first change is ours.
            next[i + 1]
        };
    }
    let pairs: Vec<(usize, usize)> = next
        .iter()
        .enumerate()
        .filter_map(|(i, j)| j.map(|j| (i, j)))
        .collect();
    let unresolved = n - pairs.len();
    (pairs, unresolved)
}

/// Runs the paired comparison with the target set to the mid at the
/// **next mid change** after each event (see
/// [`resolve_next_mid_change_targets`]).
///
/// This is the quantity the default `G*` estimates by construction: with
/// `G* = G1 + Q G*`, `mid + G*[s]` is the model's expected mid at the first
/// price change out of state `s` (`docs/model-spec.md`). A fixed event
/// horizon instead scores it against a target that, at short horizons, is
/// usually the unchanged mid. It is an additional analysis, not part of the
/// pre-registered protocol. `options.horizon` is ignored; `block_len == 0`
/// means "auto": `max(1000, 10 * ceil(mean events ahead))`.
pub fn compare_predictors_next_mid_change(
    model: &MicroPriceModel,
    events: &[BookEvent],
    options: CompareOptions,
) -> Result<ComparisonReport, EvalError> {
    let (pairs, unresolved) = resolve_next_mid_change_targets(events);
    if pairs.is_empty() {
        return Err(EvalError::InsufficientEvents {
            reason: "the mid never changes in this slice".to_string(),
        });
    }
    let mut options = options;
    if options.block_len == 0 {
        let mean_ahead =
            pairs.iter().map(|(i, j)| (j - i) as f64).sum::<f64>() / pairs.len() as f64;
        options.block_len = 1000.max(10 * mean_ahead.ceil() as usize);
    }
    compare_pairs(model, events, &pairs, options, None, unresolved, true)
}

/// One row of [`calibration_by_state`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StateCalibration {
    pub state_id: u32,
    /// Scored observations whose book encoded to this state.
    pub n: usize,
    /// Training visits of this state (from the model).
    pub training_visits: u64,
    /// The model's adjustment `G*[state]` (every observation in a state
    /// gets the same prediction).
    pub predicted: f64,
    /// Mean realized move `target_mid - mid` over the observations.
    pub realized_mean: f64,
    /// `sd / sqrt(n)` of the realized moves (`NaN` for `n < 2`). Treats
    /// observations as independent, which overlapping, serially dependent
    /// targets are not: read it as a lower bound on the uncertainty.
    pub realized_naive_se: f64,
}

/// Reliability table: for every state that occurs among the `(candidate,
/// target)` pairs, the model's predicted adjustment next to the mean
/// realized move `exact_mid(target) - exact_mid(candidate)`. A well
/// calibrated model has `realized_mean ~ predicted` in every state. Pairs
/// whose book cannot be encoded are skipped. Sorted by `state_id`.
pub fn calibration_by_state(
    model: &MicroPriceModel,
    events: &[BookEvent],
    pairs: &[(usize, usize)],
) -> Vec<StateCalibration> {
    // (n, sum, sum of squares, predicted) per state.
    let mut acc: std::collections::BTreeMap<u32, (usize, f64, f64, f64)> =
        std::collections::BTreeMap::new();
    for &(i, j) in pairs {
        let Ok(est) = model.predict(&events[i].book) else {
            continue;
        };
        let moved = exact_mid(&events[j]) - exact_mid(&events[i]);
        let e = acc
            .entry(est.state_id)
            .or_insert((0, 0.0, 0.0, est.adjustment_ticks));
        e.0 += 1;
        e.1 += moved;
        e.2 += moved * moved;
    }
    acc.into_iter()
        .map(|(state_id, (n, sum, sumsq, predicted))| {
            let mean = sum / n as f64;
            let se = if n < 2 {
                f64::NAN
            } else {
                let var = ((sumsq - n as f64 * mean * mean) / (n as f64 - 1.0)).max(0.0);
                (var / n as f64).sqrt()
            };
            StateCalibration {
                state_id,
                n,
                training_visits: model.visits().get(state_id as usize).copied().unwrap_or(0),
                predicted,
                realized_mean: mean,
                realized_naive_se: se,
            }
        })
        .collect()
}

/// Shared core: evaluates the `(candidate, target)` index pairs.
fn compare_pairs(
    model: &MicroPriceModel,
    events: &[BookEvent],
    pairs: &[(usize, usize)],
    options: CompareOptions,
    horizon_ns: Option<u64>,
    n_unresolved: usize,
    next_mid_change: bool,
) -> Result<ComparisonReport, EvalError> {
    if options.block_len == 0 {
        return Err(EvalError::InsufficientEvents {
            reason: "block_len must be >= 1".to_string(),
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
    let mut ahead_sum = 0.0f64;

    for &(i, target_idx) in pairs {
        // Exact doubled mid: `(bid + ask) / 2` computed in f64, not the
        // truncating integer `mid_price_ticks` (see the csv module docs).
        let target = exact_mid(&events[target_idx]);
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
                ahead_sum += (target_idx - i) as f64;
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
        horizon: if horizon_ns.is_some() || next_mid_change {
            0
        } else {
            options.horizon
        },
        next_mid_change,
        horizon_ns,
        n_unresolved,
        mean_events_ahead: ahead_sum / n as f64,
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
        .unwrap()
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
    fn next_mid_change_targets_skip_unchanged_mids_and_drop_the_tail() {
        // mids: 10, 10, 11, 11, 11, 10, 10
        let mids = [10, 10, 11, 11, 11, 10, 10];
        let events: Vec<_> = mids
            .iter()
            .enumerate()
            .map(|(i, m)| ev(i as u64, *m))
            .collect();
        let (pairs, unresolved) = resolve_next_mid_change_targets(&events);
        assert_eq!(pairs, vec![(0, 2), (1, 2), (2, 5), (3, 5), (4, 5)]);
        // Events 5 and 6 have no later change in the slice.
        assert_eq!(unresolved, 2);
        assert_eq!(resolve_next_mid_change_targets(&[]), (vec![], 0));
    }

    #[test]
    fn next_mid_change_report_scores_against_the_changed_mid() {
        // mids 10001, 10001, 10002: both candidates target event 2 (10002).
        let events = vec![ev(0, 10001), ev(1, 10001), ev(2, 10002)];
        let r = compare_predictors_next_mid_change(&model(), &events, opts(99, 0, 1)).unwrap();
        assert!(r.next_mid_change);
        assert_eq!(r.horizon, 0);
        assert_eq!(r.n_evaluated, 2);
        assert_eq!(r.n_unresolved, 1);
        assert_eq!(r.n_price_changing, 2);
        assert!((r.mean_events_ahead - 1.5).abs() < 1e-12);
        // mid error -1 twice; micro error 0.225 - 1 = -0.775 twice.
        assert!((r.mid.mse - 1.0).abs() < 1e-12);
        assert!((r.microprice.mae - 0.775).abs() < 1e-12);
        // No change at all -> an error, not an empty report.
        let flat = vec![ev(0, 10001), ev(1, 10001)];
        assert!(compare_predictors_next_mid_change(&model(), &flat, opts(1, 0, 1)).is_err());
    }

    #[test]
    fn calibration_table_groups_realized_moves_by_state() {
        // Balanced books -> state 1 (G* = 0.225). Next-change targets:
        // 10001 -> 10002 (+1) twice, 10002 -> 10001 (-1) once.
        let events = vec![ev(0, 10001), ev(1, 10001), ev(2, 10002), ev(3, 10001)];
        let (pairs, _) = resolve_next_mid_change_targets(&events);
        let table = calibration_by_state(&model(), &events, &pairs);
        assert_eq!(table.len(), 1);
        let row = table[0];
        assert_eq!((row.state_id, row.n, row.training_visits), (1, 3, 50));
        assert!((row.predicted - 0.225).abs() < 1e-12);
        assert!((row.realized_mean - 1.0 / 3.0).abs() < 1e-12);
        // sample sd of {1, 1, -1} = sqrt(4/3); se = sd / sqrt(3) = 2/3.
        assert!((row.realized_naive_se - 2.0 / 3.0).abs() < 1e-12);
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
        )
        .unwrap();
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

    // ------------------------------------------------ wall-clock horizons

    fn ev_at(ts: u64, mid: i64) -> BookEvent {
        BookEvent {
            timestamp_ns: ts,
            sequence: ts,
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

    #[test]
    fn a_quote_exactly_at_the_boundary_is_included() {
        // T = 20: the quote stamped exactly t + 20 is the target.
        let (pairs, unresolved) = resolve_wall_clock_targets(&[0, 10, 20, 30, 40], 20).unwrap();
        assert_eq!(pairs, vec![(0, 2), (1, 3), (2, 4)]);
        // Candidates at 30 and 40 need data through 50 / 60: unresolved.
        assert_eq!(unresolved, 2);
    }

    #[test]
    fn a_quote_one_nanosecond_past_the_boundary_is_excluded() {
        // Event 2 is stamped 21 = t + T + 1 for candidate 0: not included, the
        // prevailing quote at t + 20 is still event 1 (stamped 10).
        let (pairs, _) = resolve_wall_clock_targets(&[0, 10, 21, 30, 41], 20).unwrap();
        assert_eq!(pairs[0], (0, 1));
        // Candidate 1 (t = 10, deadline 30): event 3 (stamped exactly 30) is in.
        assert_eq!(pairs[1], (1, 3));
    }

    #[test]
    fn events_sharing_a_timestamp_resolve_to_the_last_one() {
        let (pairs, unresolved) = resolve_wall_clock_targets(&[0, 5, 5, 5, 20], 5).unwrap();
        // Candidate 0: deadline 5 -> the last of the three events stamped 5.
        assert_eq!(pairs[0], (0, 3));
        // Candidates 1..=3 (t = 5, deadline 10): still the last stamped 5.
        assert_eq!(&pairs[1..], &[(1, 3), (2, 3), (3, 3)]);
        // Candidate 4 (t = 20): data ends at 20 < 25.
        assert_eq!(unresolved, 1);
    }

    #[test]
    fn an_empty_window_keeps_the_observation_with_the_current_quote() {
        let (pairs, unresolved) = resolve_wall_clock_targets(&[0, 100, 200], 50).unwrap();
        assert_eq!(pairs, vec![(0, 0), (1, 1)]);
        assert_eq!(unresolved, 1);
    }

    #[test]
    fn wall_clock_inputs_are_validated() {
        assert!(resolve_wall_clock_targets(&[0, 10], 0).is_err());
        assert!(resolve_wall_clock_targets(&[10, 5], 1).is_err());
        assert_eq!(resolve_wall_clock_targets(&[], 5).unwrap(), (Vec::new(), 0));
        // Deadlines that overflow u64 are unresolved, not wrapped.
        let (pairs, unresolved) = resolve_wall_clock_targets(&[u64::MAX - 1, u64::MAX], 5).unwrap();
        assert!(pairs.is_empty());
        assert_eq!(unresolved, 2);
    }

    #[test]
    fn wall_clock_comparison_matches_hand_computed_errors() {
        // ts:  0      10     20     35
        // mid: 10001  10002  10003  10010      T = 20
        // i=0 -> deadline 20 -> event 2 (10003): mid err -2
        // i=1 -> deadline 30 -> event 2 (ts 20 <= 30 < 35): mid err -1
        // i=2, i=3: deadline 40 > last timestamp 35 -> unresolved.
        let events = vec![
            ev_at(0, 10001),
            ev_at(10, 10002),
            ev_at(20, 10003),
            ev_at(35, 10010),
        ];
        let r = compare_predictors_wall_clock(&model(), &events, 20, opts(1, 0, 1)).unwrap();
        assert_eq!(r.horizon_ns, Some(20));
        assert_eq!(r.horizon, 0);
        assert_eq!(r.n_evaluated, 2);
        assert_eq!(r.n_unresolved, 2);
        assert!((r.mean_events_ahead - 1.5).abs() < 1e-12); // (2 + 1) / 2
        assert!((r.mid.mae - 1.5).abs() < 1e-12);
        assert!((r.mid.mse - 2.5).abs() < 1e-12); // (4 + 1) / 2
                                                  // Balanced books -> +0.225: errors -1.775, -0.775.
        assert!((r.microprice.mae - 1.275).abs() < 1e-12);
        assert!((r.microprice.bias - (-1.275)).abs() < 1e-12);
        assert_eq!(r.n_price_changing, 2);
        assert!((r.up_share - 1.0).abs() < 1e-12);
    }

    #[test]
    fn wall_clock_reduces_to_the_event_horizon_on_a_regular_grid() {
        let events: Vec<_> = (0..300)
            .map(|i| ev_at(i as u64 * 1000, 10_000 + ((i * 7) % 5) as i64))
            .collect();
        let by_events = compare_predictors(&model(), &events, opts(3, 0, 10)).unwrap();
        let by_clock =
            compare_predictors_wall_clock(&model(), &events, 3000, opts(1, 0, 10)).unwrap();
        assert_eq!(by_clock.n_evaluated, by_events.n_evaluated);
        assert_eq!(by_clock.n_unresolved, 3);
        assert_eq!(by_events.n_unresolved, 0);
        assert_eq!(by_events.horizon_ns, None);
        assert!((by_events.mean_events_ahead - 3.0).abs() < 1e-12);
        assert_eq!(by_clock.mid, by_events.mid);
        assert_eq!(by_clock.microprice, by_events.microprice);
        assert_eq!(
            by_clock.mse_diff_vs_mid.estimate,
            by_events.mse_diff_vs_mid.estimate
        );
    }

    #[test]
    fn wall_clock_auto_block_length_and_no_data_error() {
        let events: Vec<_> = (0..50).map(|i| ev_at(i as u64, 10_000)).collect();
        // block_len 0 = auto, allowed only for wall-clock.
        assert!(compare_predictors_wall_clock(&model(), &events, 5, opts(1, 20, 0)).is_ok());
        assert!(compare_predictors(&model(), &events, opts(1, 20, 0)).is_err());
        // Horizon longer than the whole slice: nothing resolvable.
        assert!(compare_predictors_wall_clock(&model(), &events, 1000, opts(1, 0, 1)).is_err());
    }
}
