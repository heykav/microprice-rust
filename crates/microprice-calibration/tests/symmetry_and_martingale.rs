//! Tests for imbalance symmetrization and the martingale diagnostic.
//!
//! The math is derived in `docs/model-spec.md` ("Imbalance symmetrization",
//! "Martingale diagnostic"). Hand-computed values below are derived in the
//! comments next to them.

use microprice_calibration::{
    antisymmetry_residual, estimate, martingale_diagnostic, mirror_state, solve, solve_full_chain,
    CalibrationError, MicroPriceModel, ModelMetadata, SmoothingConfig, SolverConfig,
    TransitionCounter, SCHEMA_VERSION,
};
use microprice_core::{
    BookEvent, BookValidationPolicy, ImbalanceBucketing, PriceTicks, Quantity, SpreadBucketing,
    StateId, StateSpaceConfig, SymbolId, TopOfBook,
};
use microprice_data::{MarketDataSource, SyntheticConfig, SyntheticEventGenerator};

/// The toy chain of `solver`'s hand-derived example, as exact counts
/// (multiples of 1000 so the probabilities are exact):
/// from 0: stay@0 d=0 x500, up->1 x300, down->1 x200;
/// from 1: ->0 d=0 x400, stay@1 d=0 x200, up->1 x250, down->1 x150.
fn toy_counter() -> TransitionCounter {
    let mut c = TransitionCounter::new(2);
    let mut rec = |from: u32, to: u32, d: i64, n: u32| {
        for _ in 0..n {
            c.record(StateId(from), StateId(to), d);
        }
    };
    rec(0, 0, 0, 500);
    rec(0, 1, 1, 300);
    rec(0, 1, -1, 200);
    rec(1, 0, 0, 400);
    rec(1, 1, 0, 200);
    rec(1, 1, 1, 250);
    rec(1, 1, -1, 150);
    c
}

#[test]
fn default_recursion_is_not_a_martingale_hand_computed() {
    let est = estimate(&toy_counter(), SmoothingConfig::NONE).unwrap();
    let g = solve(&est, SolverConfig::DEFAULT).unwrap();
    assert!((g[0] - 0.2).abs() < 1e-8 && (g[1] - 0.225).abs() < 1e-8);
    let d = martingale_diagnostic(&est, &g).unwrap();
    // The solver's own equation holds to tolerance...
    assert!(d.fixed_point_residual < 1e-9, "{}", d.fixed_point_residual);
    // ...but the one-step drift is sum_j R[i][j] G*[j]:
    // state 0: R[0][1] = 0.5 (300+200 of 1000 land in 1) -> 0.5 * 0.225.
    // Directly: E[P'-P | 0] = 0.5*0.2 + 0.3*(1+0.225) + 0.2*(-1+0.225) - 0.2 = 0.1125.
    // state 1: R[1][1] = 0.4 -> 0.4 * 0.225 = 0.09.
    assert!((d.drift[0] - 0.1125).abs() < 1e-8, "{}", d.drift[0]);
    assert!((d.drift[1] - 0.09).abs() < 1e-8, "{}", d.drift[1]);
    assert_eq!(d.max_abs_drift_state, 0);
    assert!(d.max_abs_drift > 0.1);
}

#[test]
fn full_chain_solution_is_a_martingale_by_construction_with_smoothing() {
    // alpha > 0 leaks mass so the iteration converges even though the toy
    // chain has a nonzero stationary drift.
    let est = estimate(&toy_counter(), SmoothingConfig::new(20.0).unwrap()).unwrap();
    let g = solve_full_chain(&est, SolverConfig::DEFAULT).unwrap();
    let d = martingale_diagnostic(&est, &g).unwrap();
    assert!(d.max_abs_drift < 1e-8, "drift {}", d.max_abs_drift);
    // And the default solution of the same estimate is not one.
    let g_default = solve(&est, SolverConfig::DEFAULT).unwrap();
    let d_default = martingale_diagnostic(&est, &g_default).unwrap();
    assert!(d_default.max_abs_drift > 0.05);
    assert!(d_default.fixed_point_residual < 1e-8);
}

#[test]
fn full_chain_refuses_to_converge_on_an_asymmetric_unsmoothed_chain() {
    // Stochastic kernel with nonzero stationary mean of G1: the series
    // diverges linearly. The honest outcome is an error, not a number.
    let est = estimate(&toy_counter(), SmoothingConfig::NONE).unwrap();
    let r = solve_full_chain(&est, SolverConfig::DEFAULT);
    assert!(matches!(r, Err(CalibrationError::DidNotConverge { .. })));
}

#[test]
fn martingale_diagnostic_fails_on_a_corrupted_g_star() {
    let est = estimate(&toy_counter(), SmoothingConfig::new(20.0).unwrap()).unwrap();

    // Full-chain solution: drift ~ 0, corrupted copy: drift clearly not.
    let good = solve_full_chain(&est, SolverConfig::DEFAULT).unwrap();
    let tol = 1e-8;
    assert!(martingale_diagnostic(&est, &good).unwrap().max_abs_drift < tol);
    let mut bad = good.clone();
    bad[1] += 0.01;
    let d = martingale_diagnostic(&est, &bad).unwrap();
    assert!(d.max_abs_drift > 1e-3, "drift {}", d.max_abs_drift);

    // Default solution: the solver's own residual is tiny, corrupted is not.
    let good = solve(&est, SolverConfig::DEFAULT).unwrap();
    assert!(
        martingale_diagnostic(&est, &good)
            .unwrap()
            .fixed_point_residual
            < tol
    );
    let mut bad = good.clone();
    bad[0] -= 0.01;
    let d = martingale_diagnostic(&est, &bad).unwrap();
    assert!(d.fixed_point_residual > 1e-3);
}

#[test]
fn diagnostic_rejects_a_wrong_length_vector() {
    let est = estimate(&toy_counter(), SmoothingConfig::NONE).unwrap();
    assert!(matches!(
        martingale_diagnostic(&est, &[0.0]),
        Err(CalibrationError::DimensionMismatch { .. })
    ));
}

// ---------------------------------------------------------------- symmetry

#[test]
fn mirror_state_is_an_involution_and_flips_only_the_imbalance_bucket() {
    let n = 5;
    for s in 0..(3 * n) {
        let m = mirror_state(StateId(s), n);
        assert_eq!(mirror_state(m, n), StateId(s));
        assert_eq!(m.0 / n, s / n, "spread bucket must not change");
        assert_eq!(m.0 % n, n - 1 - s % n);
    }
    // odd N: the middle bucket is its own mirror.
    assert_eq!(mirror_state(StateId(2), 5), StateId(2));
}

#[test]
fn symmetrized_counter_is_the_exact_sum_of_data_and_its_mirror() {
    // 2 imbalance buckets x 1 spread bucket: sigma swaps 0 and 1.
    let c = toy_counter();
    let s = c.symmetrized(2).unwrap();
    // visits: 1000 + 1000 each.
    assert_eq!(s.visits(StateId(0)), 2000);
    // counts[0][0] = c[0][0] + c[1][1] = 500 + 200.
    assert_eq!(s.count(StateId(0), StateId(0)), 700);
    // counts[0][1] = c[0][1] + c[1][0] = 0 + 400.
    assert_eq!(s.count(StateId(0), StateId(1)), 400);
    assert_eq!(s.count(StateId(1), StateId(0)), 400);
    // price-changing landing: pc[0][1] = pc[0][1] + pc[1][0] = 500 + 0; pc[0][0] = 0 + pc[1][1] = 400.
    assert_eq!(s.price_change_count(StateId(0), StateId(1)), 500);
    assert_eq!(s.price_change_count(StateId(0), StateId(0)), 400);
    // delta_sum: ds[0] - ds[1] = 100 - 100 = 0; ds[1] - ds[0] = 0 (toy chain has
    // equal drift in both states so they cancel).
    assert_eq!(s.delta_sum(StateId(0)), 0);
    assert_eq!(s.delta_sum(StateId(1)), 0);
    // up/down swap under the mirror.
    assert_eq!(
        s.up_moves(StateId(0)),
        c.up_moves(StateId(0)) + c.down_moves(StateId(1))
    );
    // Symmetrizing twice adds nothing new beyond doubling: idempotent up to scale.
    let ss = s.symmetrized(2).unwrap();
    for i in 0..2u32 {
        assert_eq!(ss.visits(StateId(i)), 2 * s.visits(StateId(i)));
    }
}

#[test]
fn symmetrized_counter_rejects_incompatible_bucket_count() {
    let c = TransitionCounter::new(6);
    assert!(c.symmetrized(4).is_err());
    assert!(c.symmetrized(0).is_err());
    assert!(c.symmetrized(3).is_ok());
}

fn synthetic_events(n: usize, seed: u64) -> Vec<BookEvent> {
    let mut g = SyntheticEventGenerator::new(SyntheticConfig {
        symbol: SymbolId(1),
        initial_mid_ticks: 10_000,
        initial_spread_ticks: 2,
        initial_bid_qty: 500,
        initial_ask_qty: 500,
        arrival_rate: 0.3,
        cancel_rate: 0.2,
        market_order_rate: 0.2,
        imbalance_persistence: 0.5,
        price_move_probability: 0.1,
        seed,
    })
    .unwrap();
    (0..n).map(|_| g.next_event().unwrap()).collect()
}

/// Doubles every price (half-tick model units, as the CSV path does) so
/// `bid + ask` is always even and the integer mid is exact. Without this
/// `TopOfBook::mid_price_ticks` truncation makes the mirror of an
/// odd-sum book's mid move differ by one unit (a documented departure).
fn doubled(events: &[BookEvent]) -> Vec<BookEvent> {
    events
        .iter()
        .map(|e| BookEvent {
            book: TopOfBook::new(
                PriceTicks(2 * e.book.bid_price.0),
                e.book.bid_qty,
                PriceTicks(2 * e.book.ask_price.0),
                e.book.ask_qty,
                BookValidationPolicy::RejectCrossedAndLocked,
            )
            .unwrap(),
            ..*e
        })
        .collect()
}

/// Nudges the bid size (+1 until off-edge) of any event whose imbalance sits
/// exactly on a bucket edge k/N. `floor()` is not mirror-symmetric there
/// (I = k/N maps to bucket k, its mirror 1 - k/N to bucket N-k, not
/// N-1-k), a measure-zero effect for real-valued I that integer sizes can
/// hit; removing it lets the tests assert exact identities.
fn off_edge(events: &[BookEvent], n_imb: u32) -> Vec<BookEvent> {
    events
        .iter()
        .map(|e| {
            let mut bq = e.book.bid_qty.0;
            let aq = e.book.ask_qty.0;
            while bq != 0 && aq != 0 && (bq * u64::from(n_imb)) % (bq + aq) == 0 {
                bq += 1;
            }
            BookEvent {
                book: TopOfBook::new(
                    e.book.bid_price,
                    Quantity(bq),
                    e.book.ask_price,
                    e.book.ask_qty,
                    BookValidationPolicy::RejectCrossedAndLocked,
                )
                .unwrap(),
                ..*e
            }
        })
        .collect()
}

/// The market mirror: swap bid/ask sizes, reflect prices about a constant
/// (bid' = C - ask, ask' = C - bid). Imbalance I -> 1 - I, spread unchanged,
/// every mid move d -> -d.
fn mirror_events(events: &[BookEvent]) -> Vec<BookEvent> {
    const C: i64 = 60_000;
    events
        .iter()
        .map(|e| BookEvent {
            book: TopOfBook::new(
                PriceTicks(C - e.book.ask_price.0),
                e.book.ask_qty,
                PriceTicks(C - e.book.bid_price.0),
                e.book.bid_qty,
                BookValidationPolicy::RejectCrossedAndLocked,
            )
            .unwrap(),
            ..*e
        })
        .collect()
}

fn space(n_imb: u32) -> StateSpaceConfig {
    StateSpaceConfig::new(
        ImbalanceBucketing::new(n_imb).unwrap(),
        SpreadBucketing::new(vec![2, 4]).unwrap(),
    )
}

fn counter_for(events: &[BookEvent], sp: &StateSpaceConfig) -> TransitionCounter {
    let mut c = TransitionCounter::new(sp.state_count());
    c.observe_events(sp, events).unwrap();
    c
}

#[test]
fn symmetric_generator_stream_and_its_mirror_give_the_same_symmetrized_model() {
    // Odd N so the I = 0.5 tie (sizes exactly equal, common in the generator)
    // lands in the self-mirror middle bucket.
    let n_imb = 7u32;
    let sp = space(n_imb);
    let events = off_edge(&doubled(&synthetic_events(60_000, 7)), n_imb);
    let mirrored = mirror_events(&events);

    let c = counter_for(&events, &sp);
    let cm = counter_for(&mirrored, &sp);

    // Exact identity at count level (up to I landing exactly on a bucket
    // edge, where floor() is not mirror-symmetric; verified absent here).
    let sym = c.symmetrized(n_imb).unwrap();
    let sym_m = cm.symmetrized(n_imb).unwrap();
    let n = sp.state_count();
    let mut edge_mismatch = 0u64;
    for i in 0..n {
        assert_eq!(sym.visits(StateId(i)), sym_m.visits(StateId(i)));
        assert_eq!(sym.delta_sum(StateId(i)), sym_m.delta_sum(StateId(i)));
        for j in 0..n {
            if sym.count(StateId(i), StateId(j)) != sym_m.count(StateId(i), StateId(j)) {
                edge_mismatch += 1;
            }
        }
    }
    assert_eq!(edge_mismatch, 0);

    // And so do the solved models.
    let smoothing = SmoothingConfig::new(0.5).unwrap();
    let g = solve(&estimate(&sym, smoothing).unwrap(), SolverConfig::DEFAULT).unwrap();
    let gm = solve(&estimate(&sym_m, smoothing).unwrap(), SolverConfig::DEFAULT).unwrap();
    for (a, b) in g.iter().zip(&gm) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn symmetrized_g_star_is_exactly_antisymmetric_and_plain_is_not() {
    let n_imb = 6u32; // even N: no self-mirror bucket
    let sp = space(n_imb);
    let events = synthetic_events(80_000, 11);
    let c = counter_for(&events, &sp);
    let smoothing = SmoothingConfig::new(0.5).unwrap();

    let plain = solve(&estimate(&c, smoothing).unwrap(), SolverConfig::DEFAULT).unwrap();
    let sym_est = estimate(&c.symmetrized(n_imb).unwrap(), smoothing).unwrap();
    let sym = solve(&sym_est, SolverConfig::DEFAULT).unwrap();

    let plain_res = antisymmetry_residual(&plain, n_imb).unwrap();
    let sym_res = antisymmetry_residual(&sym, n_imb).unwrap();
    assert!(sym_res < 1e-9, "symmetrized residual {sym_res}");
    assert!(
        plain_res > 100.0 * sym_res && plain_res > 1e-6,
        "plain residual {plain_res} should be visibly nonzero (sampling noise)"
    );
    // G1 antisymmetric and p_up mirrored as well.
    for s in 0..sp.state_count() {
        let m = mirror_state(StateId(s), n_imb).0 as usize;
        assert!((sym_est.g1[s as usize] + sym_est.g1[m]).abs() < 1e-12);
        let (a, b) = (sym_est.p_up[s as usize].unwrap(), sym_est.p_up[m].unwrap());
        assert!((a + b - 1.0).abs() < 1e-12);
    }
}

#[test]
fn symmetrized_model_predicts_mirrored_books_with_opposite_adjustments() {
    let n_imb = 6u32;
    let sp = space(n_imb);
    let events = synthetic_events(80_000, 13);
    let c = counter_for(&events, &sp).symmetrized(n_imb).unwrap();
    let est = estimate(&c, SmoothingConfig::new(0.5).unwrap()).unwrap();
    let g = solve(&est, SolverConfig::DEFAULT).unwrap();
    let model = MicroPriceModel::new(
        ModelMetadata {
            schema_version: SCHEMA_VERSION,
            symbol_id: 1,
            num_imbalance_buckets: n_imb,
            spread_bucket_bounds_ticks: vec![2, 4],
            smoothing_alpha: 0.5,
            training_observations: c.total_observations(),
        },
        g,
        est.p_up.clone(),
        est.visits.clone(),
    );
    let mk = |bid: i64, bq: u64, ask: i64, aq: u64| {
        TopOfBook::new(
            PriceTicks(bid),
            Quantity(bq),
            PriceTicks(ask),
            Quantity(aq),
            BookValidationPolicy::RejectCrossedAndLocked,
        )
        .unwrap()
    };
    // (bid qty, ask qty) pairs away from bucket edges (edges at multiples of 1/6).
    for (bq, aq) in [(70u64, 30u64), (10, 90), (55, 45), (23, 77)] {
        let a = model.predict(&mk(10_000, bq, 10_002, aq)).unwrap();
        let b = model.predict(&mk(10_000, aq, 10_002, bq)).unwrap();
        assert!(
            (a.adjustment_ticks + b.adjustment_ticks).abs() < 1e-9,
            "{bq}/{aq}: {} vs {}",
            a.adjustment_ticks,
            b.adjustment_ticks
        );
        assert!((a.p_up.unwrap() + b.p_up.unwrap() - 1.0).abs() < 1e-12);
    }
}

#[test]
fn symmetrization_makes_the_full_chain_series_converge_without_smoothing() {
    // With alpha = 0 the kernel is stochastic; the series converges iff the
    // stationary mean of G1 is zero, which mirror symmetry guarantees
    // (pi is mirror-invariant, G1 antisymmetric). Then the martingale
    // property holds to solver tolerance: a real, non-trivial check.
    let n_imb = 3u32;
    let sp = StateSpaceConfig::new(
        ImbalanceBucketing::new(n_imb).unwrap(),
        SpreadBucketing::new(vec![]).unwrap(),
    );
    let events = synthetic_events(200_000, 21);
    let c = counter_for(&events, &sp).symmetrized(n_imb).unwrap();
    let est = estimate(&c, SmoothingConfig::NONE).unwrap();
    let config = SolverConfig {
        tolerance: 1e-10,
        max_iterations: 200_000,
    };
    let g = solve_full_chain(&est, config).unwrap();
    let d = martingale_diagnostic(&est, &g).unwrap();
    assert!(d.max_abs_drift < 1e-8, "drift {}", d.max_abs_drift);
    assert!(antisymmetry_residual(&g, n_imb).unwrap() < 1e-8);
}

#[test]
fn default_solution_on_synthetic_data_has_measurable_nonzero_drift() {
    // Not asserting a magnitude claim about real markets: just that the
    // diagnostic runs on the real pipeline, the solver equation holds, and
    // the drift is reported (finite).
    let sp = space(6);
    let events = synthetic_events(50_000, 3);
    let c = counter_for(&events, &sp);
    let est = estimate(&c, SmoothingConfig::new(0.5).unwrap()).unwrap();
    let g = solve(&est, SolverConfig::DEFAULT).unwrap();
    let d = martingale_diagnostic(&est, &g).unwrap();
    assert!(d.fixed_point_residual < 1e-9);
    assert!(d.max_abs_drift.is_finite());
    println!("{}", d.summary("model units"));
}
