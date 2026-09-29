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
    assert meta["schema_version"] == 2, meta
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
        "p_up",
    ):
        assert key in est, est

    # `p_up` is a real probability or None - never a fabricated value, and
    # never outside [0, 1] when present.
    assert est["p_up"] is None or 0.0 <= est["p_up"] <= 1.0, est

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

    check_predict_batch(model)

    print("microprice_python smoke test: OK -", est)
    return 0


def expect_value_error(fn, needle: str) -> None:
    try:
        fn()
    except ValueError as e:
        assert needle in str(e), (needle, str(e))
        return
    raise AssertionError(f"expected ValueError containing {needle!r}")


def check_predict_batch(model) -> None:
    """`predict_batch` must agree row-for-row with scalar `predict`."""
    from array import array

    bid_px = [10000, 10000, 10001, 9999, 10000]
    bid_qty = [500, 900, 10, 250, 1]
    ask_px = [10002, 10002, 10003, 10001, 10001]
    ask_qty = [500, 100, 990, 250, 1000]

    out = model.predict_batch(bid_px, bid_qty, ask_px, ask_qty)
    keys = (
        "mid_ticks",
        "weighted_mid_ticks",
        "microprice_ticks",
        "adjustment_ticks",
        "state_id",
        "state_observations",
        "p_up",
    )
    assert set(out) == set(keys), sorted(out)
    for k in keys:
        assert len(out[k]) == len(bid_px), (k, len(out[k]))
    for i in range(len(bid_px)):
        single = model.predict(bid_px[i], bid_qty[i], ask_px[i], ask_qty[i])
        for k in keys:
            assert out[k][i] == single[k], (i, k, out[k][i], single[k])

    # Other integer-column containers give identical results.
    as_tuples = model.predict_batch(
        tuple(bid_px), tuple(bid_qty), tuple(ask_px), tuple(ask_qty)
    )
    assert as_tuples == out
    as_arrays = model.predict_batch(
        array("q", bid_px), array("Q", bid_qty), array("q", ask_px), array("q", ask_qty)
    )
    assert as_arrays == out
    try:
        import numpy as np
    except ImportError:
        pass  # numpy is optional; the array/list paths above cover the buffer code.
    else:
        as_numpy = model.predict_batch(
            np.array(bid_px, dtype=np.int64),
            np.array(bid_qty, dtype=np.uint64),
            np.array(ask_px, dtype=np.int64),
            np.array(ask_qty, dtype=np.int64),
        )
        assert as_numpy == out

    # Empty batch is valid and returns empty columns.
    empty = model.predict_batch([], [], [], [])
    assert all(empty[k] == [] for k in keys), empty

    # Errors raise ValueError naming the problem; nothing partial is returned.
    expect_value_error(
        lambda: model.predict_batch([10000], [1, 2], [10002], [1]), "lengths differ"
    )
    expect_value_error(  # crossed book at row 1
        lambda: model.predict_batch([10000, 10005], [1, 1], [10002, 10000], [1, 1]),
        "row 1",
    )
    expect_value_error(  # empty book at row 0
        lambda: model.predict_batch([10000], [0], [10002], [0]), "row 0"
    )
    expect_value_error(
        lambda: model.predict_batch([10000], [-5], [10002], [1]), "negative"
    )
    expect_value_error(
        lambda: model.predict_batch([10000.5], [1], [10002], [1]), "bid_price_ticks"
    )


if __name__ == "__main__":
    sys.exit(main())
