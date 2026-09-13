//! Benchmarks the calibration pipeline's actual hot paths: streaming
//! transition counting (`TransitionCounter::observe_events`) and the
//! estimate -> solve step, against a realistic-sized synthetic event
//! stream (100,000 events - a middling size for a single calibration run,
//! not a stress-test extreme).
//!
//! This exists to *profile before optimizing* (Phase 15's own explicit
//! requirement — see `docs/model-spec.md`/the project roadmap): run this,
//! look at the real numbers, and only add SIMD/parallelism/sparse-matrix
//! work if a real bottleneck shows up. See `docs/benchmarking.md` for the
//! actual measured result and what it did (and didn't) justify.
//!
//! Run with `cargo bench -p microprice-calibration`.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use microprice_calibration::{estimate, solve, SmoothingConfig, SolverConfig, TransitionCounter};
use microprice_core::{BookEvent, ImbalanceBucketing, SpreadBucketing, StateSpaceConfig};
use microprice_data::{MarketDataSource, SyntheticConfig, SyntheticEventGenerator};

fn generate_events(n: u64, seed: u64) -> Vec<BookEvent> {
    let config = SyntheticConfig {
        symbol: microprice_core::SymbolId(1),
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
    };
    let mut generator = SyntheticEventGenerator::new(config).unwrap();
    (0..n).map(|_| generator.next_event().unwrap()).collect()
}

fn state_space() -> StateSpaceConfig {
    StateSpaceConfig::new(
        ImbalanceBucketing::new(20).unwrap(),
        SpreadBucketing::new(vec![1, 2, 4]).unwrap(),
    )
}

fn bench_observe_events(c: &mut Criterion) {
    let events = generate_events(100_000, 42);
    let state_space = state_space();

    let mut group = c.benchmark_group("observe_events");
    group.bench_with_input(
        BenchmarkId::new("100k_events", events.len()),
        &events,
        |b, events| {
            b.iter(|| {
                let mut counter = TransitionCounter::new(state_space.state_count());
                counter
                    .observe_events(black_box(&state_space), black_box(events))
                    .unwrap();
                black_box(counter.total_observations());
            });
        },
    );
    group.finish();
}

fn bench_estimate_and_solve(c: &mut Criterion) {
    let events = generate_events(100_000, 42);
    let state_space = state_space();
    let mut counter = TransitionCounter::new(state_space.state_count());
    counter.observe_events(&state_space, &events).unwrap();
    let smoothing = SmoothingConfig::new(0.5).unwrap();

    c.bench_function("estimate_80_states", |b| {
        b.iter(|| black_box(estimate(black_box(&counter), smoothing).unwrap()));
    });

    let estimated = estimate(&counter, smoothing).unwrap();
    c.bench_function("solve_80_states", |b| {
        b.iter(|| black_box(solve(black_box(&estimated), SolverConfig::DEFAULT).unwrap()));
    });
}

criterion_group!(benches, bench_observe_events, bench_estimate_and_solve);
criterion_main!(benches);
