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
//! ```
//!
//! `G1`'s smoothing uses a "prior mean of zero" interpretation (a
//! never-observed state gets `G1 = 0` when `alpha > 0` — "no information"
//! rather than "confidently no adjustment", since it shrinks toward, but
//! isn't forced to, exactly the unsmoothed value as `visits` grows).

use crate::error::CalibrationError;
use crate::smoothing::SmoothingConfig;
use crate::transitions::TransitionCounter;

/// The estimated `(Q, G1)` pair for a fully-specified state space, plus
/// the raw `visits` this was estimated from (kept for diagnostics/model
/// metadata, not used by the solver itself).
#[derive(Debug, Clone, PartialEq)]
pub struct EstimatedTransitions {
    pub state_count: u32,
    /// Flattened `state_count x state_count` sub-stochastic matrix.
    pub q: Vec<f64>,
    pub g1: Vec<f64>,
    pub visits: Vec<u64>,
}

impl EstimatedTransitions {
    pub fn q(&self, from: u32, to: u32) -> f64 {
        self.q[from as usize * self.state_count as usize + to as usize]
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
    }

    Ok(EstimatedTransitions {
        state_count: counter.state_count(),
        q,
        g1,
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
}
