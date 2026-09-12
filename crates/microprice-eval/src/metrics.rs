//! Plain, well-defined error metrics over `(predicted, actual)` pairs.
//!
//! **What's here and what's deliberately not:** mean absolute error, mean
//! signed error (bias), and directional accuracy are all computable
//! directly from the point predictions `MicroPriceModel::predict` already
//! produces. A Brier score needs a *probabilistic* prediction of direction
//! (a `P(price goes up)`), which this model does not produce anywhere —
//! `TransitionCounter` only ever accumulates a *signed delta sum* per
//! state, not separate up/down transition counts, so there is no
//! `P(up)` to score without inventing one from data the calibration
//! pipeline doesn't collect. Rather than fabricate a Brier score from an
//! ad-hoc reinterpretation of `G1`/`G*` as a probability (which they are
//! not — they're signed tick-magnitude expectations), this is left an
//! explicit, disclosed gap; see `docs/model-spec.md`'s Open Questions.

/// Mean of `|predicted - actual|` over all pairs. Panics if the slices are
/// empty or of different lengths — both are caller bugs, not runtime data
/// conditions (see `crate::evaluate` for where the caller-facing validation
/// actually lives).
pub fn mean_absolute_error(predicted: &[f64], actual: &[f64]) -> f64 {
    assert_eq!(predicted.len(), actual.len());
    assert!(!predicted.is_empty());
    predicted
        .iter()
        .zip(actual.iter())
        .map(|(p, a)| (p - a).abs())
        .sum::<f64>()
        / predicted.len() as f64
}

/// Mean of `(predicted - actual)`, i.e. signed bias: positive means the
/// predictions systematically overshoot, negative means they systematically
/// undershoot. Distinct from MAE, which can't tell an unbiased-but-noisy
/// predictor apart from a biased one.
pub fn mean_signed_error(predicted: &[f64], actual: &[f64]) -> f64 {
    assert_eq!(predicted.len(), actual.len());
    assert!(!predicted.is_empty());
    predicted
        .iter()
        .zip(actual.iter())
        .map(|(p, a)| p - a)
        .sum::<f64>()
        / predicted.len() as f64
}

/// Directional accuracy: among pairs where `actual != 0.0` (a real move
/// happened — pairs where nothing moved are excluded rather than counted
/// as free correct/incorrect guesses either way, since "predict no
/// direction" isn't a directional claim to begin with), the fraction where
/// `predicted` and `actual` have the same sign.
///
/// Returns `(accuracy, n_directional)`; `accuracy` is `f64::NAN` if
/// `n_directional == 0` (nothing to score) — callers must check
/// `n_directional` before trusting `accuracy`, exactly like
/// `docs/model-spec.md`'s stance on not inventing a value for an
/// undefined case.
pub fn direction_accuracy(predicted: &[f64], actual: &[f64]) -> (f64, usize) {
    assert_eq!(predicted.len(), actual.len());
    let mut correct = 0usize;
    let mut n_directional = 0usize;
    for (p, a) in predicted.iter().zip(actual.iter()) {
        if *a == 0.0 {
            continue;
        }
        n_directional += 1;
        if p.signum() == a.signum() {
            correct += 1;
        }
    }
    let accuracy = if n_directional == 0 {
        f64::NAN
    } else {
        correct as f64 / n_directional as f64
    };
    (accuracy, n_directional)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mae_matches_a_hand_computed_example() {
        let predicted = [1.0, 2.0, 3.0];
        let actual = [1.5, 2.0, 5.0];
        // |1-1.5| + |2-2| + |3-5| = 0.5 + 0.0 + 2.0 = 2.5 / 3
        assert!((mean_absolute_error(&predicted, &actual) - (2.5 / 3.0)).abs() < 1e-12);
    }

    #[test]
    fn bias_is_signed_unlike_mae() {
        let predicted = [2.0, 2.0];
        let actual = [1.0, 1.0];
        assert!((mean_signed_error(&predicted, &actual) - 1.0).abs() < 1e-12);
        assert!((mean_absolute_error(&predicted, &actual) - 1.0).abs() < 1e-12);

        let predicted2 = [0.0, 2.0];
        let actual2 = [1.0, 1.0];
        // signed errors: -1, +1 -> mean 0 (looks perfectly unbiased)
        assert!((mean_signed_error(&predicted2, &actual2) - 0.0).abs() < 1e-12);
        // but MAE correctly still shows real per-prediction error.
        assert!((mean_absolute_error(&predicted2, &actual2) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn direction_accuracy_ignores_no_move_pairs_and_scores_the_rest() {
        let predicted = [1.0, -1.0, 0.5, -0.2];
        let actual = [2.0, -3.0, 0.0, 0.1]; // last: sign mismatch; third: no move, excluded
        let (acc, n) = direction_accuracy(&predicted, &actual);
        assert_eq!(n, 3); // indices 0,1,3 have actual != 0
        assert!((acc - (2.0 / 3.0)).abs() < 1e-12); // 0 and 1 correct, 3 wrong
    }

    #[test]
    fn direction_accuracy_is_nan_with_no_directional_pairs() {
        let predicted = [1.0, -1.0];
        let actual = [0.0, 0.0];
        let (acc, n) = direction_accuracy(&predicted, &actual);
        assert_eq!(n, 0);
        assert!(acc.is_nan());
    }
}
