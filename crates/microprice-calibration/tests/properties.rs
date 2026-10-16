//! Property tests for the estimator and solver on randomly generated
//! transition counts (not on market data): row probabilities, the bound on
//! `G*`, exact antisymmetry on mirror-symmetric data, and determinism.

use microprice_calibration::{
    estimate, mirror_state, solve, SmoothingConfig, SolverConfig, TransitionCounter,
};
use microprice_core::StateId;
use proptest::prelude::*;

/// A random transition list over `n_imb * n_spread` states: `(from, to,
/// delta)` with `delta` in `-3..=3` (zero about half the time).
fn transitions() -> impl Strategy<Value = (u32, u32, Vec<(u32, u32, i64)>)> {
    (1u32..=5, 1u32..=3).prop_flat_map(|(n_imb, n_spread)| {
        let n = n_imb * n_spread;
        let one = (0..n, 0..n, prop_oneof![Just(0i64), -3i64..=3]);
        (
            Just(n_imb),
            Just(n_spread),
            proptest::collection::vec(one, 1..400),
        )
    })
}

fn counter_from(n: u32, records: &[(u32, u32, i64)]) -> TransitionCounter {
    let mut c = TransitionCounter::new(n);
    for &(i, j, d) in records {
        c.record(StateId(i), StateId(j), d);
    }
    c
}

fn solver() -> SolverConfig {
    SolverConfig {
        tolerance: 1e-12,
        max_iterations: 200_000,
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Every row of the smoothed model is a probability distribution:
    /// `sum_j Q[i][j] + sum_j R[i][j] + alpha / D[i] == 1`, with every entry
    /// in `[0, 1]`, and `p_up` (when defined) is a probability.
    #[test]
    fn smoothed_rows_are_probability_distributions(
        (n_imb, n_spread, records) in transitions(),
        alpha in 0.01f64..3.0,
    ) {
        let n = (n_imb * n_spread) as usize;
        let c = counter_from(n as u32, &records);
        let est = estimate(&c, SmoothingConfig::new(alpha).unwrap()).unwrap();
        for i in 0..n {
            let row_q = &est.q[i * n..(i + 1) * n];
            let row_r = &est.r[i * n..(i + 1) * n];
            for v in row_q.iter().chain(row_r) {
                prop_assert!((0.0..=1.0).contains(v));
            }
            let d = est.visits[i] as f64 + alpha * (n as f64 + 1.0);
            let total: f64 = row_q.iter().sum::<f64>() + row_r.iter().sum::<f64>() + alpha / d;
            prop_assert!((total - 1.0).abs() < 1e-12, "row {} sums to {}", i, total);
            if let Some(p) = est.p_up[i] {
                prop_assert!((0.0..=1.0).contains(&p));
            }
        }
    }

    /// `G*[i]` is an expected move up to the first price change, so it can
    /// never exceed the largest observed absolute move (and is 0 if no move
    /// was ever observed).
    #[test]
    fn g_star_is_bounded_by_the_largest_observed_move(
        (n_imb, n_spread, records) in transitions(),
        alpha in 0.01f64..3.0,
    ) {
        let n = n_imb * n_spread;
        let c = counter_from(n, &records);
        let max_move = records.iter().map(|r| r.2.unsigned_abs()).max().unwrap_or(0) as f64;
        let est = estimate(&c, SmoothingConfig::new(alpha).unwrap()).unwrap();
        let g = solve(&est, solver()).unwrap();
        for (i, gi) in g.iter().enumerate() {
            prop_assert!(gi.is_finite());
            prop_assert!(gi.abs() <= max_move + 1e-9, "G*[{}] = {} > {}", i, gi, max_move);
        }
    }

    /// The bound in the hardest case for it: every observed move is `+1`
    /// tick, so `0 < G* <= 1` exactly. (The pre-fix `G1` denominator broke
    /// this whenever smoothing mass was comparable to the visit counts.)
    #[test]
    fn g_star_is_bounded_when_every_move_is_one_tick_up(
        n in 1u32..=12,
        edges in proptest::collection::vec((0u32..12, 0u32..12), 1..60),
        alpha in 0.1f64..3.0,
    ) {
        let records: Vec<_> = edges.iter().map(|&(i, j)| (i % n, j % n, 1i64)).collect();
        let est = estimate(&counter_from(n, &records), SmoothingConfig::new(alpha).unwrap()).unwrap();
        let g = solve(&est, solver()).unwrap();
        for gi in &g {
            prop_assert!(*gi >= 0.0 && *gi <= 1.0 + 1e-12, "G* = {}", gi);
        }
    }

    /// On data that is exactly mirror-symmetric (each transition recorded
    /// together with its bid/ask-swapped image `sigma(i) -> sigma(j)`, move
    /// negated), `G*` is antisymmetric and `p_up` mirrors to `1 - p_up`,
    /// without calling `symmetrized`.
    #[test]
    fn mirror_symmetric_data_gives_an_antisymmetric_g_star(
        (n_imb, n_spread, records) in transitions(),
        alpha in 0.01f64..3.0,
    ) {
        let n = n_imb * n_spread;
        let sigma = |s: u32| mirror_state(StateId(s), n_imb).0;
        let mut both = records.clone();
        both.extend(records.iter().map(|&(i, j, d)| (sigma(i), sigma(j), -d)));
        let c = counter_from(n, &both);
        let est = estimate(&c, SmoothingConfig::new(alpha).unwrap()).unwrap();
        let g = solve(&est, solver()).unwrap();
        for s in 0..n {
            let m = sigma(s) as usize;
            prop_assert!((g[s as usize] + g[m]).abs() < 1e-9);
            if let (Some(p), Some(pm)) = (est.p_up[s as usize], est.p_up[m]) {
                prop_assert!((p + pm - 1.0).abs() < 1e-12);
            }
        }
        // And `symmetrized` of an already-symmetric counter changes no
        // estimate: it only doubles every count.
        let est2 = estimate(&c.symmetrized(n_imb).unwrap(), SmoothingConfig::new(alpha * 2.0).unwrap()).unwrap();
        let g2 = solve(&est2, solver()).unwrap();
        for s in 0..n as usize {
            prop_assert!((g[s] - g2[s]).abs() < 1e-9);
        }
    }

    /// Same counts in, bit-identical `G*` out; the record order does not
    /// matter either (counts are order-free).
    #[test]
    fn estimation_and_solve_are_deterministic(
        (n_imb, n_spread, records) in transitions(),
        alpha in 0.01f64..3.0,
    ) {
        let n = n_imb * n_spread;
        let smoothing = SmoothingConfig::new(alpha).unwrap();
        let a = solve(&estimate(&counter_from(n, &records), smoothing).unwrap(), solver()).unwrap();
        let b = solve(&estimate(&counter_from(n, &records), smoothing).unwrap(), solver()).unwrap();
        let mut reversed = records.clone();
        reversed.reverse();
        let c = solve(&estimate(&counter_from(n, &reversed), smoothing).unwrap(), solver()).unwrap();
        prop_assert_eq!(a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), b.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        prop_assert_eq!(a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), c.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    }
}
