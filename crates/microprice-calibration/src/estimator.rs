//! Turns raw [`crate::transitions::TransitionCounter`] statistics into the
//! sub-stochastic transition matrix `Q` and the one-step expected
//! price-change vector `G1` that `crate::solver` needs — see
//! `docs/model-spec.md`'s "Micro-price estimation" section for the exact
//! formulas and "Smoothing" for what `alpha` does to them.
//!
//! **Smoothing normalization, made precise:** each state's `visits[i]`
//! observations are treated as having `state_count + 1` possible
//! outcomes — the `state_count` non-price-changing destinations, plus one
//! *lumped* "price changed" outcome (V1 does not separate up/down for
//! smoothing-denominator purposes, only for the sign already captured in
//! `delta_sum`). Additive smoothing adds `alpha` pseudo-observations to
//! every one of those `state_count + 1` outcomes:
//!
//! ```text
//! Q[i][j] = (count[i][j] + alpha) / (visits[i] + alpha * (state_count + 1))
//! G1[i]   = delta_sum[i] / (visits[i] + alpha)
//! p_up[i] = (up_moves[i] + alpha) / (up_moves[i] + down_moves[i] + 2 * alpha)
//! ```
//!
//! `G1`'s smoothing uses a "prior mean of zero" interpretation (a
//! never-observed state gets `G1 = 0` when `alpha > 0` — "no information"
//! rather than "confidently no adjustment", since it shrinks toward, but
//! isn't forced to, exactly the unsmoothed value as `visits` grows).
//!
//! **`p_up`, and why it isn't derivable from `G1`:** the empirical
//! probability that a step out of state `i` is upward, *conditioned on the
//! price moving at all* — flat transitions are excluded from both
//! numerator and denominator, since a state whose price never moved
//! carries no directional evidence and estimates `None` at `alpha = 0`
//! rather than a fabricated coin flip. Its prior is exactly a coin flip:
//! `alpha` pseudo-observations on each side pulls a directionless state to
//! `0.5`, the probability analogue of `G1`'s zero-mean prior. This is the
//! quantity `docs/model-spec.md` twice flagged as missing — `G1` is a
//! *signed tick-magnitude expectation*, and no rearrangement of a sum of
//! signed deltas recovers the up/down split (a `+2`/`-1` history and a
//! `+1`/`0` history have identical `delta_sum`), so a `P(up)` has to be
//! counted at record time, which `TransitionCounter` now does.

use crate::error::CalibrationError;
use crate::smoothing::SmoothingConfig;
use crate::transitions::TransitionCounter;

/// The estimated `(Q, G1, p_up)` triple for a fully-specified state space,
/// plus the raw `visits` this was estimated from (kept for
/// diagnostics/model metadata, not used by the solver itself).
#[derive(Debug, Clone, PartialEq)]
pub struct EstimatedTransitions {
    pub state_count: u32,
    /// Flattened `state_count x state_count` sub-stochastic matrix.
    pub q: Vec<f64>,
    pub g1: Vec<f64>,
    /// Per-state `P(next move is up | next move is directional)`; `None`
    /// where the state was observed but never moved.
    pub p_up: Vec<Option<f64>>,
    pub visits: Vec<u64>,
}

impl EstimatedTransitions {
    pub fn q(&self, from: u32, to: u32) -> f64 {
        self.q[from as usize * self.state_count as usize + to as usize]
    }

    pub fn p_up(&self, state: u32) -> Option<f64> {
        self.p_up[state as usize]
    }
}

pub fn estimate(
    counter: &TransitionCounter,
    smoothing: SmoothingConfig,
) -> Result<EstimatedTransitions, CalibrationError> {
    let n = counter.state_count() as usize;
    let alpha = smoothing.alpha;
    let mut q = vec![0.0f64; n * n];
    let mut g1 = vec![0.0f64; n];
    let mut p_up = vec![None; n];
    let mut visits = vec![0u64; n];

    for i in 0..n {
        let v = counter.visits(microprice_core::StateId(i as u32));
        visits[i] = v;
        if v == 0 && alpha == 0.0 {
            return Err(CalibrationError::InsufficientObservations { state: i as u32 });
        }

        let denom_q = v as f64 + alpha * (n as f64 + 1.0);
        for j in 0..n {
            let c = counter.count(
                microprice_core::StateId(i as u32),
                microprice_core::StateId(j as u32),
            );
            let value = (c as f64 + alpha) / denom_q;
            if !value.is_finite() {
                return Err(CalibrationError::InsufficientObservations { state: i as u32 });
            }
            q[i * n + j] = value;
        }

        let denom_g1 = v as f64 + alpha;
        let delta_sum = counter.delta_sum(microprice_core::StateId(i as u32));
        let g1_value = delta_sum as f64 / denom_g1;
        if !g1_value.is_finite() {
            return Err(CalibrationError::InsufficientObservations { state: i as u32 });
        }
        g1[i] = g1_value;

        let up = counter.up_moves(microprice_core::StateId(i as u32));
        let down = counter.down_moves(microprice_core::StateId(i as u32));
        let directional = up + down;
        // `directional + 2 * alpha` is zero only when both are zero, i.e.
        // the genuinely undefined case, so the division below is safe.
        p_up[i] = if directional == 0 && alpha == 0.0 {
            None
        } else {
            Some((up as f64 + alpha) / (directional as f64 + 2.0 * alpha))
        };
    }

    Ok(EstimatedTransitions {
        state_count: counter.state_count(),
        q,
        g1,
        p_up,
        visits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use microprice_core::StateId;

    #[test]
    fn zero_alpha_reproduces_plain_maximum_likelihood_estimates() {
        let mut counter = TransitionCounter::new(2);
        counter.record(StateId(0), StateId(0), 0);
        counter.record(StateId(0), StateId(0), 0);
        counter.record(StateId(0), StateId(1), 1);
        counter.record(StateId(1), StateId(0), -1);
        counter.record(StateId(1), StateId(1), 0);

        let est = estimate(&counter, SmoothingConfig::NONE).unwrap();
        // state 0: 3 visits, 2 stayed at 0 (no price change), 1 went to 1 (price changed +1)
        assert!((est.q(0, 0) - 2.0 / 3.0).abs() < 1e-12);
        assert!((est.q(0, 1) - 0.0).abs() < 1e-12);
        assert!((est.g1[0] - (1.0 / 3.0)).abs() < 1e-12);
        // state 1: 2 visits, 1 stayed at 1 (no change), 1 went to 0 (price changed -1)
        assert!((est.q(1, 1) - 0.5).abs() < 1e-12);
        assert!((est.q(1, 0) - 0.0).abs() < 1e-12);
        assert!((est.g1[1] - (-0.5)).abs() < 1e-12);
    }

    #[test]
    fn zero_alpha_with_an_unobserved_state_is_an_error_not_a_guess() {
        let mut counter = TransitionCounter::new(3);
        counter.record(StateId(0), StateId(1), 1);
        // state 2 never visited.
        let result = estimate(&counter, SmoothingConfig::NONE);
        assert!(matches!(
            result,
            Err(CalibrationError::InsufficientObservations { .. })
        ));
    }

    #[test]
    fn positive_alpha_gives_a_neutral_estimate_for_an_unobserved_state() {
        let mut counter = TransitionCounter::new(3);
        counter.record(StateId(0), StateId(1), 1);
        let smoothing = SmoothingConfig::new(1.0).unwrap();
        let est = estimate(&counter, smoothing).unwrap();
        // State 2: zero visits, alpha > 0 -> G1 shrinks exactly to 0 (a
        // "no information" prior, not a confident claim of no movement),
        // and Q is uniform over the (state_count + 1) outcome categories.
        assert_eq!(est.g1[2], 0.0);
        let expected_uniform_q = 1.0 / (3.0 + 1.0);
        for j in 0..3 {
            assert!((est.q(2, j as u32) - expected_uniform_q).abs() < 1e-12);
        }
    }

    #[test]
    fn smoothing_shrinks_toward_but_does_not_erase_a_well_observed_estimate() {
        // A single-state counter: only checking state 0's own shrinkage
        // behavior here, so there's no second, unobserved state to
        // trigger InsufficientObservations under SmoothingConfig::NONE.
        let mut counter = TransitionCounter::new(1);
        for _ in 0..1000 {
            counter.record(StateId(0), StateId(0), 0);
        }
        for _ in 0..1000 {
            counter.record(StateId(0), StateId(0), 2);
        }
        let unsmoothed = estimate(&counter, SmoothingConfig::NONE).unwrap();
        let smoothing = SmoothingConfig::new(1.0).unwrap();
        let smoothed = estimate(&counter, smoothing).unwrap();
        // With 2000 real observations, alpha=1 barely moves the estimate.
        assert!((unsmoothed.g1[0] - smoothed.g1[0]).abs() < 0.01);
        assert!((unsmoothed.q(0, 0) - smoothed.q(0, 0)).abs() < 0.01);
    }

    #[test]
    fn p_up_is_the_up_fraction_among_directional_moves() {
        let mut counter = TransitionCounter::new(2);
        counter.record(StateId(0), StateId(1), 1); // up
        counter.record(StateId(0), StateId(0), 1); // up
        counter.record(StateId(0), StateId(1), -1); // down
        counter.record(StateId(0), StateId(0), 0); // flat: excluded entirely
        counter.record(StateId(1), StateId(0), 1); // up
        counter.record(StateId(1), StateId(0), -1); // down

        let est = estimate(&counter, SmoothingConfig::NONE).unwrap();
        // state 0: 2 up / (2 up + 1 down) - the flat visit is in `visits`
        // (4) but in neither directional bucket.
        assert_eq!(est.p_up(0), Some(2.0 / 3.0));
        assert_eq!(est.p_up(1), Some(0.5));
        assert_eq!(est.visits[0], 4);
    }

    #[test]
    fn p_up_is_none_for_a_visited_state_that_never_moved() {
        let mut counter = TransitionCounter::new(2);
        counter.record(StateId(0), StateId(0), 0);
        counter.record(StateId(0), StateId(0), 0);
        counter.record(StateId(1), StateId(1), 0);
        let est = estimate(&counter, SmoothingConfig::NONE).unwrap();
        // Real visits, zero directional evidence - undefined, not 0.5.
        assert_eq!(est.visits[0], 2);
        assert_eq!(est.p_up(0), None);
        assert_eq!(est.p_up(1), None);
    }

    #[test]
    fn alpha_turns_a_directionless_state_into_an_explicit_coin_flip() {
        let mut counter = TransitionCounter::new(2);
        counter.record(StateId(0), StateId(0), 0);
        counter.record(StateId(1), StateId(1), 0);
        let est = estimate(&counter, SmoothingConfig::new(1.0).unwrap()).unwrap();
        assert_eq!(est.p_up(0), Some(0.5));
        assert_eq!(est.p_up(1), Some(0.5));
    }

    #[test]
    fn alpha_barely_moves_a_well_observed_p_up() {
        let mut counter = TransitionCounter::new(1);
        for _ in 0..600 {
            counter.record(StateId(0), StateId(0), 1);
        }
        for _ in 0..400 {
            counter.record(StateId(0), StateId(0), -1);
        }
        let unsmoothed = estimate(&counter, SmoothingConfig::NONE).unwrap();
        let smoothed = estimate(&counter, SmoothingConfig::new(1.0).unwrap()).unwrap();
        assert!((unsmoothed.p_up(0).unwrap() - 0.6).abs() < 1e-12);
        // (600 + 1) / (1000 + 2), vs 0.6 - barely moved.
        assert!((smoothed.p_up(0).unwrap() - 0.6).abs() < 0.002);
    }
}
