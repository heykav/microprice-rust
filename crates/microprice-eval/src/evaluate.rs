//! Out-of-sample evaluation of a calibrated [`MicroPriceModel`] against a
//! chronological `BookEvent` stream, compared against the `mid` and
//! `weighted_mid` baselines the model spec itself defines (never just a
//! bare micro-price number in isolation — see the model's own
//! `MicroPriceEstimate`, which already carries all three quantities).
//!
//! **Horizon, made explicit:** each candidate observation at index `i` is
//! compared against the *actual* mid price `horizon` events later
//! (`events[i + horizon]`), not against the next single transition. This is
//! one reasonable, disclosed choice among several valid ones (a
//! next-price-changing-event horizon, or a fixed time horizon, are others)
//! — see `docs/model-spec.md`'s Open Questions for why this isn't the only
//! defensible choice.

use microprice_calibration::MicroPriceModel;
use microprice_core::BookEvent;

use crate::error::EvalError;
use crate::metrics::{brier_score, direction_accuracy, mean_absolute_error, mean_signed_error};

/// The result of evaluating a model against one chronological, held-out
/// event stream at one horizon.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvalReport {
    pub horizon: usize,
    /// Candidate observations actually scored (state encoding succeeded).
    pub n_evaluated: usize,
    /// Candidate observations skipped because the book failed state
    /// encoding (see `microprice_core::StateSpaceConfig::encode`) — not
    /// silently dropped from the report, unlike a bare count of `n_evaluated`
    /// would imply.
    pub n_skipped: usize,
    pub microprice_mae: f64,
    pub mid_mae: f64,
    pub weighted_mid_mae: f64,
    /// Mean signed error of the micro-price prediction (see
    /// `crate::metrics::mean_signed_error`) — this crate's stand-in for
    /// "calibration error": whether the model systematically over- or
    /// under-shoots, distinct from the unsigned MAE above.
    pub microprice_bias: f64,
    /// Fraction of directional (`actual != mid`) observations where the
    /// micro-price's implied direction (`microprice - mid`) matched the
    /// actual realized direction. `NaN` if `n_directional == 0`.
    pub microprice_direction_accuracy: f64,
    /// Same, but for the `weighted_mid` baseline's implied direction
    /// (`weighted_mid - mid`).
    pub weighted_mid_direction_accuracy: f64,
    pub n_directional: usize,
    /// Brier score of the model's per-state `P(up)` against the realized
    /// direction over `horizon`, over the observations that both carry a
    /// `p_up` (state not directionally blank) and actually moved. `NaN`
    /// when `n_probabilistic == 0`.
    pub microprice_brier: f64,
    /// Brier score of the **climatological baseline** — predict this same
    /// sample's own up-rate, constantly — over the identical observations.
    /// This is what makes `microprice_brier` readable: as with `mid_mae`
    /// against `microprice_mae`, a Brier score on its own says nothing
    /// about whether the forecast carries information.
    pub brier_baseline: f64,
    /// `1 - microprice_brier / brier_baseline`. Positive means the model
    /// beat always-predicting-the-base-rate; `0` means it matched an
    /// uninformative forecast; negative means it would have done better
    /// knowing nothing. `NaN` when the baseline is `0`, i.e. every scored
    /// observation moved the same way, so a constant forecast is already
    /// perfect and the ratio is undefined.
    pub brier_skill_score: f64,
    /// Observations contributing to the three Brier fields above.
    pub n_probabilistic: usize,
}

/// Evaluates `model` against every `events[i]` that has an event `horizon`
/// steps ahead of it, comparing the model's prediction at `i` to the
/// actual realized mid price at `i + horizon`.
pub fn evaluate(
    model: &MicroPriceModel,
    events: &[BookEvent],
    horizon: usize,
) -> Result<EvalReport, EvalError> {
    if horizon == 0 {
        return Err(EvalError::InvalidHorizon { horizon });
    }
    if events.len() <= horizon {
        return Err(EvalError::InsufficientEvents {
            reason: format!(
                "need more than horizon ({horizon}) events to evaluate, got {}",
                events.len()
            ),
        });
    }

    let mut microprice_pred = Vec::new();
    let mut mid_pred = Vec::new();
    let mut weighted_mid_pred = Vec::new();
    let mut p_up_pred = Vec::new();
    let mut actual = Vec::new();
    let mut n_skipped = 0usize;

    for i in 0..events.len() - horizon {
        let future_mid = events[i + horizon].book.mid_price_ticks() as f64;
        match model.predict(&events[i].book) {
            Ok(est) => {
                microprice_pred.push(est.microprice_ticks);
                mid_pred.push(est.mid_ticks);
                weighted_mid_pred.push(est.weighted_mid_ticks);
                p_up_pred.push(est.p_up);
                actual.push(future_mid);
            }
            Err(_) => n_skipped += 1,
        }
    }

    if microprice_pred.is_empty() {
        return Err(EvalError::InsufficientEvents {
            reason: "every candidate observation failed state encoding".to_string(),
        });
    }

    let microprice_mae = mean_absolute_error(&microprice_pred, &actual);
    let mid_mae = mean_absolute_error(&mid_pred, &actual);
    let weighted_mid_mae = mean_absolute_error(&weighted_mid_pred, &actual);
    let microprice_bias = mean_signed_error(&microprice_pred, &actual);

    let actual_direction: Vec<f64> = actual
        .iter()
        .zip(mid_pred.iter())
        .map(|(a, mid)| a - mid)
        .collect();
    let microprice_direction: Vec<f64> = microprice_pred
        .iter()
        .zip(mid_pred.iter())
        .map(|(m, mid)| m - mid)
        .collect();
    let weighted_mid_direction: Vec<f64> = weighted_mid_pred
        .iter()
        .zip(mid_pred.iter())
        .map(|(w, mid)| w - mid)
        .collect();

    let (microprice_direction_accuracy, n_directional) =
        direction_accuracy(&microprice_direction, &actual_direction);
    let (weighted_mid_direction_accuracy, _) =
        direction_accuracy(&weighted_mid_direction, &actual_direction);

    // The probabilistic subset: observations where the model actually has a
    // probability to offer AND a direction to be scored against. A flat
    // realization is excluded for the same reason `direction_accuracy`
    // excludes it — "it didn't move" is not an up-or-down outcome, and
    // scoring it as a confident `0.0` would reward a model for predicting
    // something that isn't a coin flip.
    let mut probabilities = Vec::new();
    let mut outcomes = Vec::new();
    for (p_up, direction) in p_up_pred.iter().zip(actual_direction.iter()) {
        let Some(p_up) = p_up else { continue };
        if *direction == 0.0 {
            continue;
        }
        probabilities.push(*p_up);
        outcomes.push(if *direction > 0.0 { 1.0 } else { 0.0 });
    }

    let n_probabilistic = probabilities.len();
    let (microprice_brier, brier_baseline, brier_skill_score) = if n_probabilistic == 0 {
        (f64::NAN, f64::NAN, f64::NAN)
    } else {
        let score = brier_score(&probabilities, &outcomes);
        let base_rate = outcomes.iter().sum::<f64>() / n_probabilistic as f64;
        // Scored through the same function rather than hard-coding the
        // closed form `b * (1 - b)`, so the reference is guaranteed to be
        // computed exactly the way the model's score is.
        let reference = brier_score(&vec![base_rate; n_probabilistic], &outcomes);
        let skill = if reference > 0.0 {
            1.0 - score / reference
        } else {
            f64::NAN
        };
        (score, reference, skill)
    };

    Ok(EvalReport {
        horizon,
        n_evaluated: microprice_pred.len(),
        n_skipped,
        microprice_mae,
        mid_mae,
        weighted_mid_mae,
        microprice_bias,
        microprice_direction_accuracy,
        weighted_mid_direction_accuracy,
        n_directional,
        microprice_brier,
        brier_baseline,
        brier_skill_score,
        n_probabilistic,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use microprice_calibration::ModelMetadata;
    use microprice_core::{BookValidationPolicy, PriceTicks, Quantity, SymbolId, TopOfBook};

    fn toy_model() -> MicroPriceModel {
        // 2 imbalance buckets, 1 catch-all spread bucket -> 2 states.
        // Balanced (I=0.5) books land in state 1 (floor(0.5*2)=1).
        model_with_p_up(vec![Some(0.5), Some(0.5)])
    }

    fn model_with_p_up(p_up: Vec<Option<f64>>) -> MicroPriceModel {
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
            p_up,
            vec![100, 50],
        )
    }

    fn balanced_event(seq: u64, mid: i64) -> BookEvent {
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

    #[test]
    fn matches_hand_computed_mae_and_bias_over_three_events() {
        // mids: 10001, 10002, 10001 - all balanced books, so every
        // prediction lands in state 1 (adjustment 0.225) and weighted_mid
        // always exactly equals mid (balanced qty).
        let events = vec![
            balanced_event(0, 10001),
            balanced_event(1, 10002),
            balanced_event(2, 10001),
        ];
        let model = toy_model();
        let report = evaluate(&model, &events, 1).unwrap();

        assert_eq!(report.n_evaluated, 2);
        assert_eq!(report.n_skipped, 0);
        // i=0: microprice=10001.225 vs actual 10002 -> |err|=0.775
        // i=1: microprice=10002.225 vs actual 10001 -> |err|=1.225
        // mean = 1.0
        assert!((report.microprice_mae - 1.0).abs() < 1e-9);
        // mid and weighted_mid both exactly predict "no change" here, so
        // both baselines are off by exactly 1.0 tick both times.
        assert!((report.mid_mae - 1.0).abs() < 1e-9);
        assert!((report.weighted_mid_mae - 1.0).abs() < 1e-9);
        // signed errors: -0.775, +1.225 -> mean 0.225 (matches the fixed
        // +0.225 adjustment applied regardless of actual direction).
        assert!((report.microprice_bias - 0.225).abs() < 1e-9);

        // Both directional observations are real moves (actual != mid).
        assert_eq!(report.n_directional, 2);
        // microprice always predicts "up" (adjustment is +0.225 in both
        // states here) - correct once (mid rose), wrong once (mid fell).
        assert!((report.microprice_direction_accuracy - 0.5).abs() < 1e-9);

        // Brier: outcomes are 1.0 (mid rose) then 0.0 (mid fell), and
        // state 1 forecasts a flat 0.5 both times, so this is exactly the
        // uninformative case: score 0.25, identical to the climatological
        // base rate (also 0.5, since the two outcomes split evenly), and
        // therefore skill 0 - no information, by construction.
        assert_eq!(report.n_probabilistic, 2);
        assert!((report.microprice_brier - 0.25).abs() < 1e-12);
        assert!((report.brier_baseline - 0.25).abs() < 1e-12);
        assert!((report.brier_skill_score - 0.0).abs() < 1e-12);
    }

    #[test]
    fn brier_scores_only_observations_that_have_a_probability() {
        // State 1 is directionally blank, and every balanced book lands
        // there, so there is nothing probabilistic to score at all.
        let events = vec![
            balanced_event(0, 10001),
            balanced_event(1, 10002),
            balanced_event(2, 10001),
        ];
        let model = model_with_p_up(vec![Some(0.5), None]);
        let report = evaluate(&model, &events, 1).unwrap();

        // The point predictions are unaffected - only the probabilistic
        // score is absent.
        assert_eq!(report.n_evaluated, 2);
        assert_eq!(report.n_directional, 2);
        assert_eq!(report.n_probabilistic, 0);
        assert!(report.microprice_brier.is_nan());
        assert!(report.brier_baseline.is_nan());
        assert!(report.brier_skill_score.is_nan());
    }

    #[test]
    fn brier_excludes_observations_where_the_mid_never_moved() {
        // Horizon 1 over three identical books: both scored observations
        // have a real `p_up` but a zero realized move, so neither is a
        // up-or-down outcome to score against.
        let events = vec![
            balanced_event(0, 10001),
            balanced_event(1, 10001),
            balanced_event(2, 10001),
        ];
        let report = evaluate(&toy_model(), &events, 1).unwrap();
        assert_eq!(report.n_evaluated, 2);
        assert_eq!(report.n_directional, 0);
        assert_eq!(report.n_probabilistic, 0);
        assert!(report.microprice_brier.is_nan());
    }

    #[test]
    fn rejects_a_zero_horizon() {
        let events = vec![balanced_event(0, 10001), balanced_event(1, 10002)];
        let model = toy_model();
        assert!(matches!(
            evaluate(&model, &events, 0),
            Err(EvalError::InvalidHorizon { horizon: 0 })
        ));
    }

    #[test]
    fn rejects_too_few_events_for_the_requested_horizon() {
        let events = vec![balanced_event(0, 10001), balanced_event(1, 10002)];
        let model = toy_model();
        assert!(matches!(
            evaluate(&model, &events, 5),
            Err(EvalError::InsufficientEvents { .. })
        ));
    }

    #[test]
    fn skips_but_does_not_abort_on_an_unencodable_book() {
        // An empty book (both sides zero depth) fails state encoding.
        let empty_book_event = BookEvent {
            timestamp_ns: 500,
            sequence: 1,
            symbol: SymbolId(1),
            book: TopOfBook::new(
                PriceTicks(9999),
                Quantity(0),
                PriceTicks(10001),
                Quantity(0),
                BookValidationPolicy::RejectCrossedAllowLocked,
            )
            .unwrap(),
        };
        let events = vec![
            balanced_event(0, 10001),
            empty_book_event,
            balanced_event(2, 10001),
        ];
        let model = toy_model();
        let report = evaluate(&model, &events, 1).unwrap();
        // i=0 encodes fine; i=1 (the empty book) fails encoding and is
        // skipped, not fatal to the whole evaluation.
        assert_eq!(report.n_evaluated, 1);
        assert_eq!(report.n_skipped, 1);
    }
}
