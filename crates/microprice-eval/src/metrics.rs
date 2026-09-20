//! Plain, well-defined error metrics over `(predicted, actual)` pairs.
//!
//! **What's here and what's deliberately not:** mean absolute error, mean
//! signed error (bias), and directional accuracy are all computable
//! directly from the point predictions `MicroPriceModel::predict` already
//! produces. A Brier score needs a *probabilistic* prediction of direction
//! (a `P(price goes up)`), which `adjustment_ticks`/`G*` cannot supply —
//! they are signed tick-magnitude expectations, and no rearrangement of a
//! sum of signed deltas recovers an up/down split. So the model now also
//! carries `p_up`, counted at the transition level, and [`brier_score`]
//! scores *that*. It is deliberately not an ad-hoc squashing of `G*` into
//! `[0, 1]`, which would produce a number shaped like a Brier score
//! without being one.

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

/// Brier score: mean squared error of a *probability* forecast against a
/// binary outcome encoded as `0.0`/`1.0`. Lower is better — `0.0` is a
/// perfect forecast, `1.0` is confidently wrong on every observation, and
/// `0.25` is exactly what an uninformative constant `0.5` scores on any
/// sample. That last fact is why this is only readable next to a
/// baseline: `evaluate` reports the climatological reference alongside it.
///
/// Panics if the slices are empty, of different lengths, if any probability
/// falls outside `[0, 1]`, or if any outcome is not exactly `0.0`/`1.0` —
/// all caller bugs, same contract as the rest of this module.
pub fn brier_score(probabilities: &[f64], outcomes: &[f64]) -> f64 {
    assert_eq!(probabilities.len(), outcomes.len());
    assert!(!probabilities.is_empty());
    for p in probabilities {
        assert!(
            p.is_finite() && (0.0..=1.0).contains(p),
            "brier_score probabilities must be finite and within [0, 1], got {p}"
        );
    }
    for o in outcomes {
        assert!(
            *o == 0.0 || *o == 1.0,
            "brier_score outcomes must be exactly 0.0 or 1.0, got {o}"
        );
    }
    probabilities
        .iter()
        .zip(outcomes.iter())
        .map(|(p, o)| (p - o) * (p - o))
        .sum::<f64>()
        / probabilities.len() as f64
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

    #[test]
    fn brier_score_matches_a_hand_computed_example() {
        let probabilities = [0.9, 0.1, 0.5];
        let outcomes = [1.0, 0.0, 0.0];
        // 0.01 + 0.01 + 0.25 = 0.27 / 3
        assert!((brier_score(&probabilities, &outcomes) - 0.27 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn brier_score_of_a_constant_coin_flip_is_always_a_quarter() {
        // The reference point that makes the number readable: with no
        // information at all, a 0.5 forecast scores 0.25 whatever the
        // outcome mix.
        for outcomes in [
            vec![1.0, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0, 0.0],
            vec![1.0, 0.0, 1.0, 0.0],
            vec![1.0, 1.0, 1.0, 0.0],
        ] {
            let probabilities = vec![0.5; outcomes.len()];
            assert!((brier_score(&probabilities, &outcomes) - 0.25).abs() < 1e-12);
        }
    }

    #[test]
    fn brier_score_is_zero_for_a_perfect_forecast_and_one_for_a_confidently_wrong_one() {
        assert_eq!(brier_score(&[1.0, 0.0, 1.0], &[1.0, 0.0, 1.0]), 0.0);
        assert_eq!(brier_score(&[1.0, 0.0, 1.0], &[0.0, 1.0, 0.0]), 1.0);
    }

    #[test]
    fn brier_score_rewards_a_discriminating_forecast_over_a_flat_one() {
        let outcomes = [1.0, 1.0, 0.0, 0.0];
        let discriminating = [0.9, 0.8, 0.2, 0.1];
        let flat = [0.5, 0.5, 0.5, 0.5];
        assert!(brier_score(&discriminating, &outcomes) < brier_score(&flat, &outcomes));
        // And an anti-correlated forecast is worse than knowing nothing.
        let inverted = [0.1, 0.2, 0.8, 0.9];
        assert!(brier_score(&inverted, &outcomes) > brier_score(&flat, &outcomes));
    }

    #[test]
    #[should_panic(expected = "outcomes must be exactly 0.0 or 1.0")]
    fn brier_score_rejects_a_non_binary_outcome() {
        brier_score(&[0.5], &[0.5]);
    }

    #[test]
    #[should_panic(expected = "within [0, 1]")]
    fn brier_score_rejects_an_out_of_range_probability() {
        brier_score(&[1.4], &[1.0]);
    }
}
