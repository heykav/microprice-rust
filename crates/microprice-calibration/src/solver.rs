//! The micro-price adjustment solver: `G* = G1 + Q @ G*`, solved by
//! fixed-point iteration — never an explicit matrix inverse. See
//! `docs/model-spec.md`'s "Micro-price estimation" section.

use crate::error::CalibrationError;
use crate::estimator::EstimatedTransitions;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolverConfig {
    pub tolerance: f64,
    pub max_iterations: usize,
}

impl SolverConfig {
    pub const DEFAULT: SolverConfig = SolverConfig {
        tolerance: 1e-10,
        max_iterations: 10_000,
    };
}

/// Solves `G* = G1 + Q @ G*` by fixed-point (Jacobi) iteration:
/// `G*_0 = G1`, `G*_{k+1} = G1 + Q @ G*_k`, until the max absolute change
/// across all states drops below `tolerance` or `max_iterations` is
/// exhausted (returned as [`CalibrationError::DidNotConverge`], not a
/// silently truncated result).
pub fn solve(
    est: &EstimatedTransitions,
    config: SolverConfig,
) -> Result<Vec<f64>, CalibrationError> {
    let n = est.state_count as usize;
    let mut g = est.g1.clone();
    let mut next = vec![0.0f64; n];

    let row_dot = |row: &[f64], g: &[f64]| -> f64 {
        row.iter().zip(g.iter()).map(|(qij, gj)| qij * gj).sum()
    };

    for _iter in 0..config.max_iterations {
        let mut max_delta: f64 = 0.0;
        for i in 0..n {
            let acc = est.g1[i] + row_dot(&est.q[i * n..i * n + n], &g);
            next[i] = acc;
            max_delta = max_delta.max((next[i] - g[i]).abs());
        }
        std::mem::swap(&mut g, &mut next);
        if max_delta < config.tolerance {
            return Ok(g);
        }
    }

    // Compute the final delta honestly for the error message rather than
    // reporting a stale/zero value.
    let mut final_delta: f64 = 0.0;
    for i in 0..n {
        let acc = est.g1[i] + row_dot(&est.q[i * n..i * n + n], &g);
        final_delta = final_delta.max((acc - g[i]).abs());
    }
    Err(CalibrationError::DidNotConverge {
        max_iterations: config.max_iterations,
        final_delta,
        tolerance: config.tolerance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_2state() -> EstimatedTransitions {
        // Hand-derived toy example (docs/model-spec.md / section 60):
        // Q = [[0.5, 0.0], [0.4, 0.2]], G1 = [0.1, 0.1].
        // (I - Q)^-1 = [[2.0, 0.0], [1.0, 1.25]]
        // G* = (I-Q)^-1 @ G1 = [0.2, 0.225]
        EstimatedTransitions {
            state_count: 2,
            q: vec![0.5, 0.0, 0.4, 0.2],
            g1: vec![0.1, 0.1],
            visits: vec![1000, 1000],
        }
    }

    #[test]
    fn matches_the_hand_derived_toy_example() {
        let est = toy_2state();
        let g_star = solve(&est, SolverConfig::DEFAULT).unwrap();
        assert!((g_star[0] - 0.2).abs() < 1e-8, "g*[0] = {}", g_star[0]);
        assert!((g_star[1] - 0.225).abs() < 1e-8, "g*[1] = {}", g_star[1]);
    }

    #[test]
    fn zero_transition_matrix_gives_g_star_equal_to_g1() {
        // Q = 0 everywhere: no persistence at all, so G* should reduce to
        // exactly G1 (the recursive term contributes nothing).
        let est = EstimatedTransitions {
            state_count: 3,
            q: vec![0.0; 9],
            g1: vec![0.5, -0.3, 0.0],
            visits: vec![10, 10, 10],
        };
        let g_star = solve(&est, SolverConfig::DEFAULT).unwrap();
        for (gi, g1i) in g_star.iter().zip(est.g1.iter()) {
            assert!((gi - g1i).abs() < 1e-10);
        }
    }

    #[test]
    fn detects_non_convergence_for_a_near_unit_spectral_radius_matrix() {
        // Q = [[0.9999999, 0], [0, 0.9999999]] - each state's continuation
        // probability is essentially 1, so the series G1 + Q G1 + Q^2 G1
        // + ... converges only extremely slowly (spectral radius ~1). A
        // tight tolerance with few iterations must report non-convergence
        // honestly rather than returning a garbage/truncated answer.
        let est = EstimatedTransitions {
            state_count: 2,
            q: vec![0.9999999, 0.0, 0.0, 0.9999999],
            g1: vec![1.0, 1.0],
            visits: vec![10, 10],
        };
        let config = SolverConfig {
            tolerance: 1e-12,
            max_iterations: 5,
        };
        let result = solve(&est, config);
        assert!(matches!(
            result,
            Err(CalibrationError::DidNotConverge { .. })
        ));
    }

    #[test]
    fn converges_for_a_well_conditioned_larger_system() {
        // A slightly denser 4-state system with real off-diagonal mixing,
        // checked only for convergence + finiteness (not a hand-derived
        // value) - the toy_2state test above covers exact-value
        // correctness; this covers "doesn't blow up on a bigger matrix".
        let est = EstimatedTransitions {
            state_count: 4,
            q: vec![
                0.3, 0.1, 0.0, 0.0, 0.1, 0.3, 0.1, 0.0, 0.0, 0.1, 0.3, 0.1, 0.0, 0.0, 0.1, 0.3,
            ],
            g1: vec![0.05, -0.02, 0.01, 0.0],
            visits: vec![100, 100, 100, 100],
        };
        let g_star = solve(&est, SolverConfig::DEFAULT).unwrap();
        for v in &g_star {
            assert!(v.is_finite());
        }
    }
}
