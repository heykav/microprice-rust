//! Numerical diagnostics on a calibrated `G*`: the martingale (one-step
//! drift) check and the mirror-antisymmetry check.
//!
//! # The martingale property, and what it does and does not follow from
//!
//! Write the micro-price in state `s` with mid `M` as `P = M + G(s)`. The
//! model's own one-step kernel from state `i` is: with probability
//! `Q[i][j]` the mid is unchanged and the state becomes `j`; with
//! probability `R[i][j]` the mid changes and the state becomes `j`; the
//! expected mid change is `G1[i]` (over all transitions). Hence
//!
//! ```text
//! E[P_next | i] - P_i  =  G1[i] + sum_j (Q[i][j] + R[i][j]) G[j] - G[i]   (drift_i)
//! ```
//!
//! and `P` is a martingale on the model's own kernel iff `drift_i = 0` for
//! every `i` (i.e. `G = G1 + (Q + R) G`).
//!
//! The default solver, however, enforces `G* = G1 + Q G*` (only the
//! non-price-changing block `Q`). Substituting,
//!
//! ```text
//! drift_i = sum_j R[i][j] G*[j]          (when G* solves the Q-recursion)
//! ```
//!
//! which is **not zero in general**. So the martingale property does *not*
//! hold "by construction" for the default recursion; this module measures
//! how far from it the calibrated model is. [`crate::solver::solve_full_chain`]
//! solves the martingale recursion instead, for which `drift = 0` holds by
//! construction and is checked by the tests. Whether the paper's `G*` is
//! the one or the other is UNVERIFIED here (the paper was unavailable).
//!
//! With smoothing `alpha > 0` the kernel `Q + R` leaks mass
//! `alpha / (visits + alpha (n + 1))` per row; the leaked mass contributes
//! nothing to the expectation (zero-mean prior), which is the same
//! convention the solver uses, so the identity above holds exactly for the
//! smoothed objects.

use crate::error::CalibrationError;
use crate::estimator::EstimatedTransitions;
use crate::transitions::mirror_state;
use microprice_core::StateId;

/// Result of [`martingale_diagnostic`]. All values are in the units of the
/// prices the model was calibrated in (model units; divide by the
/// resolution to get exchange ticks).
#[derive(Debug, Clone, PartialEq)]
pub struct MartingaleDiagnostic {
    /// `max_i |G1[i] + sum_j Q[i][j] G[j] - G[i]|`: residual of the default
    /// solver's own equation. Should be at the solver tolerance for a
    /// `G` produced by [`crate::solver::solve`].
    pub fixed_point_residual: f64,
    /// Per-state one-step drift `E[P_next | i] - P_i` (see module docs).
    pub drift: Vec<f64>,
    /// `max_i |drift_i|`.
    pub max_abs_drift: f64,
    /// State attaining `max_abs_drift`.
    pub max_abs_drift_state: u32,
    /// `sum_i visits_i |drift_i| / sum_i visits_i` (0 if no visits).
    pub visit_weighted_mean_abs_drift: f64,
    /// `max_i |G[i]|`, for scale.
    pub max_abs_g: f64,
}

impl MartingaleDiagnostic {
    /// Returns a copy with every length multiplied by `factor` (e.g.
    /// `1 / resolution` to report in exchange ticks).
    pub fn rescaled(&self, factor: f64) -> Self {
        MartingaleDiagnostic {
            fixed_point_residual: self.fixed_point_residual * factor,
            drift: self.drift.iter().map(|d| d * factor).collect(),
            max_abs_drift: self.max_abs_drift * factor,
            max_abs_drift_state: self.max_abs_drift_state,
            visit_weighted_mean_abs_drift: self.visit_weighted_mean_abs_drift * factor,
            max_abs_g: self.max_abs_g * factor,
        }
    }

    /// One-line human-readable summary, in the diagnostic's own units
    /// (label them with `unit`).
    pub fn summary(&self, unit: &str) -> String {
        format!(
            "fixed-point residual {:.3e} {unit}; one-step micro-price drift: max |E[P'-P|state]| = {:.3e} {unit} \
             (state {}), visit-weighted mean {:.3e} {unit}, vs max |G*| = {:.3e} {unit}",
            self.fixed_point_residual,
            self.max_abs_drift,
            self.max_abs_drift_state,
            self.visit_weighted_mean_abs_drift,
            self.max_abs_g,
        )
    }
}

/// Computes the martingale diagnostic of `g` against the estimated kernel.
pub fn martingale_diagnostic(
    est: &EstimatedTransitions,
    g: &[f64],
) -> Result<MartingaleDiagnostic, CalibrationError> {
    let n = est.state_count as usize;
    if g.len() != n {
        return Err(CalibrationError::DimensionMismatch {
            expected: n,
            actual: g.len(),
        });
    }
    let mut drift = vec![0.0; n];
    let mut fp: f64 = 0.0;
    for i in 0..n {
        let mut qg = 0.0;
        let mut rg = 0.0;
        for (j, gj) in g.iter().enumerate() {
            qg += est.q[i * n + j] * gj;
            rg += est.r[i * n + j] * gj;
        }
        fp = fp.max((est.g1[i] + qg - g[i]).abs());
        drift[i] = est.g1[i] + qg + rg - g[i];
    }
    let mut max_abs_drift: f64 = 0.0;
    let mut max_state = 0u32;
    for (i, d) in drift.iter().enumerate() {
        if d.abs() > max_abs_drift {
            max_abs_drift = d.abs();
            max_state = i as u32;
        }
    }
    let total_visits: u64 = est.visits.iter().sum();
    let weighted = if total_visits == 0 {
        0.0
    } else {
        drift
            .iter()
            .zip(&est.visits)
            .map(|(d, v)| d.abs() * *v as f64)
            .sum::<f64>()
            / total_visits as f64
    };
    Ok(MartingaleDiagnostic {
        fixed_point_residual: fp,
        drift,
        max_abs_drift,
        max_abs_drift_state: max_state,
        visit_weighted_mean_abs_drift: weighted,
        max_abs_g: g.iter().fold(0.0f64, |a, x| a.max(x.abs())),
    })
}

/// `max_s |G[s] + G[sigma(s)]|`: how far `g` is from exact antisymmetry
/// under the mirror map (0 for a model calibrated with symmetrization, up
/// to floating-point rounding).
pub fn antisymmetry_residual(
    g: &[f64],
    num_imbalance_buckets: u32,
) -> Result<f64, CalibrationError> {
    if num_imbalance_buckets == 0 || !g.len().is_multiple_of(num_imbalance_buckets as usize) {
        return Err(CalibrationError::DimensionMismatch {
            expected: num_imbalance_buckets as usize,
            actual: g.len(),
        });
    }
    let mut worst: f64 = 0.0;
    for (s, gs) in g.iter().enumerate() {
        let m = mirror_state(StateId(s as u32), num_imbalance_buckets).0 as usize;
        worst = worst.max((gs + g[m]).abs());
    }
    Ok(worst)
}
