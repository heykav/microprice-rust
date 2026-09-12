//! Known-truth validation (Phase 6 / "Prompt 6" of the project brief):
//! sample transitions directly from a hand-specified toy Markov chain with
//! an algebraically-known `G*`, feed the samples through the *real*
//! `TransitionCounter` -> `estimate` -> `solve` pipeline, and confirm the
//! estimate converges toward the known truth as sample size grows.
//!
//! This deliberately does **not** use `microprice-data`'s behavioral
//! `SyntheticEventGenerator` (Phase 4): that generator's implied
//! (imbalance-bucket, spread-bucket) transition matrix isn't known in
//! closed form (it emerges from the compound arrival/cancel/market-order
//! process), which makes it the wrong tool for *this* specific claim. A
//! separate, simpler, directly-specified categorical sampler is the
//! correct tool for verifying the estimator/solver are mathematically
//! correct against a chain whose true answer can be checked by hand.

use microprice_calibration::{estimate, solve, SmoothingConfig, SolverConfig, TransitionCounter};
use microprice_core::StateId;
use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// The same 2-state toy chain used in `solver`'s hand-derived unit test:
///
/// From state 0: P(stay@0, Δ=0)=0.5, P(price up, Δ=+1)=0.3, P(price down, Δ=-1)=0.2
/// From state 1: P(go to 0, Δ=0)=0.4, P(stay@1, Δ=0)=0.2, P(up, Δ=+1)=0.25, P(down, Δ=-1)=0.15
///
/// Known Q = [[0.5, 0.0], [0.4, 0.2]], G1 = [0.1, 0.1],
/// known G* = (I - Q)^-1 @ G1 = [0.2, 0.225] (derived by hand, see
/// `solver::tests::matches_the_hand_derived_toy_example`).
struct ToyChain {
    // Each outcome: (destination state if non-price-changing else same
    // state for bookkeeping, delta_ticks, is_price_changing)
    from_0: WeightedIndex<f64>,
    from_1: WeightedIndex<f64>,
}

/// (to_state, delta_ticks) outcomes, in the same order as the weights
/// passed to each `WeightedIndex`. The `to_state` for a *price-changing*
/// outcome (nonzero delta) never affects `Q`/`G1` (see
/// `TransitionCounter::record` - `counts` only increments when
/// `delta == 0`), but it does determine where sampling continues from,
/// which is what actually gives state 1 any visits at all here: sending
/// state 0's price-changing moves to state 1 (rather than looping back to
/// 0) is what makes this a connected, two-state chain instead of one
/// where state 1 is unreachable.
const OUTCOMES_FROM_0: [(u32, i64); 3] = [(0, 0), (1, 1), (1, -1)]; // stay@0/Δ0, up->1, down->1
const OUTCOMES_FROM_1: [(u32, i64); 4] = [(0, 0), (1, 0), (1, 1), (1, -1)]; // ->0/Δ0, stay@1/Δ0, up, down

fn toy_chain() -> ToyChain {
    ToyChain {
        from_0: WeightedIndex::new([0.5, 0.3, 0.2]).unwrap(),
        from_1: WeightedIndex::new([0.4, 0.2, 0.25, 0.15]).unwrap(),
    }
}

fn sample_n(n: u64, seed: u64) -> TransitionCounter {
    let chain = toy_chain();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut counter = TransitionCounter::new(2);
    let mut state = 0u32;
    for _ in 0..n {
        let (to, delta) = if state == 0 {
            OUTCOMES_FROM_0[chain.from_0.sample(&mut rng)]
        } else {
            OUTCOMES_FROM_1[chain.from_1.sample(&mut rng)]
        };
        counter.record(StateId(state), StateId(to), delta);
        state = to;
    }
    counter
}

fn g_star_error(n: u64, seed: u64) -> f64 {
    let known_g_star = [0.2_f64, 0.225_f64];
    let counter = sample_n(n, seed);
    let est = estimate(&counter, SmoothingConfig::new(0.5).unwrap())
        .expect("smoothing > 0 guarantees every state is estimable");
    let g_star = solve(&est, SolverConfig::DEFAULT).expect("well-conditioned toy chain converges");
    let mut max_err: f64 = 0.0;
    for i in 0..2 {
        max_err = max_err.max((g_star[i] - known_g_star[i]).abs());
    }
    max_err
}

#[test]
fn estimated_g_star_converges_toward_the_known_truth_as_sample_size_grows() {
    let err_1k = g_star_error(1_000, 1);
    let err_10k = g_star_error(10_000, 1);
    let err_100k = g_star_error(100_000, 1);

    println!("known-truth |G* - truth|: n=1e3 -> {err_1k:.5}, n=1e4 -> {err_10k:.5}, n=1e5 -> {err_100k:.5}");

    // Not a strict monotonic requirement on every single seed (sampling
    // noise could occasionally make a larger sample look marginally
    // worse), but the *overall trend* across two orders of magnitude must
    // be real shrinkage - and every error must be small in absolute terms
    // by n=1e5.
    assert!(
        err_100k < err_1k,
        "expected error to shrink from n=1e3 ({err_1k}) to n=1e5 ({err_100k})"
    );
    assert!(err_100k < 0.01, "n=1e5 error {err_100k} is too large");
    assert!(err_10k < 0.05, "n=1e4 error {err_10k} is too large");
}

#[test]
fn estimated_g_star_is_close_at_a_moderate_sample_size_across_several_seeds() {
    // Robustness across seeds, not just one lucky draw.
    for seed in [1, 2, 3, 4, 5] {
        let err = g_star_error(50_000, seed);
        assert!(err < 0.03, "seed {seed}: error {err} too large at n=50000");
    }
}
