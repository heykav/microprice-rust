//! Benchmarks state encoding: `StateSpaceConfig::encode`.
//!
//! Run with `cargo bench -p microprice-core`. Numbers are hardware- and
//! build-dependent; see the README/benchmark docs for the actual measured
//! result on the machine that last ran this, rather than trusting any
//! number written in a comment here.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use microprice_core::state::{ImbalanceBucketing, SpreadBucketing, StateSpaceConfig};
use microprice_core::{BookValidationPolicy, PriceTicks, Quantity, TopOfBook};

fn bench_state_encoding(c: &mut Criterion) {
    // 20 imbalance buckets x 4 spread buckets = 80 states, matching the
    // project brief's own worked example (section 12).
    let config = StateSpaceConfig::new(
        ImbalanceBucketing::new(20).unwrap(),
        SpreadBucketing::new(vec![1, 2, 4]).unwrap(),
    );
    let book = TopOfBook::new(
        PriceTicks(18732),
        Quantity(4200),
        PriceTicks(18733),
        Quantity(1700),
        BookValidationPolicy::RejectCrossedAndLocked,
    )
    .unwrap();

    c.bench_function("state_encode_single", |b| {
        b.iter(|| black_box(config.encode(black_box(&book)).unwrap()));
    });

    c.bench_function("state_encode_batch_1000", |b| {
        b.iter(|| {
            for _ in 0..1000 {
                black_box(config.encode(black_box(&book)).unwrap());
            }
        });
    });
}

criterion_group!(benches, bench_state_encoding);
criterion_main!(benches);
