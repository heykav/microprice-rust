#!/usr/bin/env python3
"""Real end-to-end smoke test for the `microprice_python` PyO3 bindings.

Run against a wheel actually built by `maturin build`/`maturin develop`
and installed into the active interpreter (see docs/python-bindings.md) -
this is not a mock or a stub, it exercises the real Rust calibration
pipeline through the real extension module. Used by CI's
`python-bindings` job so this crate doesn't silently bit-rot outside the
root Cargo workspace's `cargo test --workspace`.
"""

import sys
import tempfile
from pathlib import Path

import microprice_python as mp


def main() -> int:
    model = mp.train_synthetic(
        num_events=50_000,
        num_imbalance_buckets=10,
        spread_bucket_bounds_ticks=[1, 2, 4],
        seed=42,
    )
    meta = model.metadata()
    assert meta["schema_version"] == 1, meta
    assert meta["num_imbalance_buckets"] == 10, meta
    assert meta["spread_bucket_bounds_ticks"] == [1, 2, 4], meta
    assert meta["training_observations"] > 0, meta

    est = model.predict(
        bid_price_ticks=10000, bid_qty=500, ask_price_ticks=10002, ask_qty=500
    )
    assert est["mid_ticks"] == 10001.0, est
    for key in (
        "weighted_mid_ticks",
        "microprice_ticks",
        "adjustment_ticks",
        "state_id",
        "state_observations",
    ):
        assert key in est, est

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "model.bin")
        model.save(path)
        loaded = mp.MicroPriceModel.load(path)
        assert loaded.metadata() == meta, (loaded.metadata(), meta)

    # Error paths must raise, not crash or silently return garbage.
    try:
        model.predict(
            bid_price_ticks=10005, bid_qty=100, ask_price_ticks=10000, ask_qty=100
        )
        raise AssertionError("expected a ValueError for a crossed book")
    except ValueError:
        pass

    try:
        mp.MicroPriceModel.load("/tmp/microprice-python-smoke-test-missing.bin")
        raise AssertionError("expected a ValueError for a missing model file")
    except ValueError:
        pass

    print("microprice_python smoke test: OK -", est)
    return 0


if __name__ == "__main__":
    sys.exit(main())
