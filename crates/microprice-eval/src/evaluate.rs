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
use crate::metrics::{direction_accuracy, mean_absolute_error, mean_signed_error};

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
    let mut actual = Vec::new();
    let mut n_skipped = 0usize;

    for i in 0..events.len() - horizon {
        let future_mid = events[i + horizon].book.mid_price_ticks() as f64;
        match model.predict(&events[i].book) {
            Ok(est) => {
                microprice_pred.push(est.microprice_ticks);
                mid_pred.push(est.mid_ticks);
                weighted_mid_pred.push(est.weighted_mid_ticks);
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
