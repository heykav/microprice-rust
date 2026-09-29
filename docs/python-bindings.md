# Python bindings (`microprice-python`)

Phase 13. PyO3 bindings exposing `MicroPriceModel` (load/save/predict/
predict_batch/metadata) and `train_synthetic` (the same counting → estimation → solving
pipeline `microprice-cli`'s `train` subcommand runs) to Python.

## Why this crate isn't in the root Cargo workspace

Every other crate in this repository is a member of the root
`microprice-rust` workspace and is built/tested by the root `cargo test
--workspace` CI invocation. `microprice-python` deliberately is not:

A PyO3 extension module built with the `extension-module` feature compiles
as a `cdylib` that expects to be *dynamically loaded into a running Python
process* — it does not link `libpython` itself. `cargo build`/`cargo test`
run directly (the way CI runs every other crate) can't load and exercise
that `cdylib` as an ordinary Rust test binary. The standard, correct way to
build and test a PyO3 extension module is `maturin`, which builds the
wheel and loads it into a real Python interpreter for you. Rather than
special-case the root workspace's CI job for one member, this crate
declares its own `[workspace]` (see its `Cargo.toml`) and is verified
separately, exactly as documented below — a real, disclosed trade-off, not
an oversight.

## Building and verifying it for real

This was actually run on this machine (Python 3.12 via Homebrew, since the
system `/usr/bin/python3` is 3.9.6 and PyO3 needs a Python whose dev
headers/lib are available):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install maturin

cd crates/microprice-python
maturin develop   # builds the extension and installs it into the active venv
```

```python
import microprice_python as mp

model = mp.train_synthetic(
    num_events=200_000,
    num_imbalance_buckets=10,
    spread_bucket_bounds_ticks=[1, 2, 4],
    seed=42,
)
print(model.metadata())
print(model.predict(bid_price_ticks=10000, bid_qty=500, ask_price_ticks=10002, ask_qty=500))

model.save("model.bin")
loaded = mp.MicroPriceModel.load("model.bin")
assert loaded.metadata() == model.metadata()
```

Measured on this machine, this real run (same seed/config as the CLI's own
`docs`/README example) produced numbers **identical** to `microprice-cli`'s
`train`/`predict` on the same inputs — the Rust core, not a reimplementation,
is what both surfaces call:

```text
metadata: {'schema_version': 2, 'symbol_id': 1, 'num_imbalance_buckets': 10,
           'spread_bucket_bounds_ticks': [1, 2, 4], 'smoothing_alpha': 0.5,
           'training_observations': 199999}
predict (balanced):   adjustment_ticks=-0.020089..., microprice_ticks=10000.9799..., p_up=0.485606...
predict (imbalanced): adjustment_ticks=-0.079776..., microprice_ticks=10000.9202..., p_up=0.431818...
```

The `adjustment_ticks`/`microprice_ticks` values here are **byte-identical**
to the ones this doc reported before `p_up` existed (schema version 1 →
2). That is the intended, checkable consequence of `p_up` living outside
the `G*` recursion: the probability is estimated and carried alongside
`G*`, and perturbs it not at all — see
[`model-spec.md`](model-spec.md#micro-price-estimation-v1-as-of-phase-7-8).

Error paths were verified too, not just the happy path: a crossed book
passed to `predict` raises a Python `ValueError` carrying the exact same
message `microprice-core`'s `MicroPriceError` produces
(`"book failed price-ordering validation: bid=... ask=... (policy=...)"`),
and loading a nonexistent model path raises `ValueError` with the
underlying I/O error message — no panics, no silent `None`/garbage
returns.

## Batch prediction: `predict_batch`

`MicroPriceModel.predict_batch(bid_price_ticks, bid_qty, ask_price_ticks,
ask_qty)` is the vectorised form of `predict`: four 1-D integer columns of
equal length in, one dict of equal-length columns out.

```python
import numpy as np
import microprice_python as mp

model = mp.train_synthetic(num_events=200_000, num_imbalance_buckets=10,
                           spread_bucket_bounds_ticks=[1, 2, 4], seed=42)
out = model.predict_batch(
    np.array([10000, 10000, 10001]), np.array([500, 900, 10]),   # bid px, bid qty
    np.array([10002, 10002, 10003]), np.array([500, 100, 990]),  # ask px, ask qty
)
out["microprice_ticks"]   # [..., ..., ...]  (list of floats)
out["p_up"]               # list; None where the state has no directional evidence
```

- **Inputs.** Prices are integer ticks (`int64`); quantities are non-negative
  integers. Any contiguous 1-D `int64`/`uint64` buffer is read directly
  (NumPy arrays, `array.array('q'/'Q')`); otherwise any sequence of Python
  ints (list, tuple) works. NumPy is **not** a dependency of the package or of
  CI; it is used only if you pass it. Floats are rejected, not truncated.
- **Output.** A dict with the same keys as `predict`
  (`mid_ticks`, `weighted_mid_ticks`, `microprice_ticks`, `adjustment_ticks`,
  `state_id`, `state_observations`, `p_up`), each a Python list of length
  `n`. Row `k` equals `predict` on the `k`-th book (checked by the smoke test).
  Wrap columns in `np.asarray` if you want arrays. It is a convenience over
  the Rust batch path, not a claim about speed.
- **Errors.** Mismatched column lengths, non-integer columns, negative
  quantities and invalid books (crossed/locked, both sizes zero) raise
  `ValueError`; row-level problems name the first offending row index
  (`"row 3: ..."`). No partial result is returned.
- **Empty input** returns empty columns.

Tests live in `docs/python_bindings_smoke_test.py` (`check_predict_batch`),
which CI's `python-bindings` job runs against a freshly built wheel: row-wise
agreement with `predict`, list/tuple/`array.array` (and NumPy when installed)
inputs, empty batches, and each error path.

## What isn't wired up yet

- Only synthetic-data training is exposed (`train_synthetic`) — Parquet
  ingestion (`microprice-data`'s `parquet-ingestion` feature) is not yet
  exposed to Python. Wiring it up is straightforward (the same pattern as
  `train_synthetic`, reading events via
  `microprice_data::read_events_from_parquet` instead of the synthetic
  generator) but hasn't been done.
- No PyPI publishing/CI wheel-building workflow exists yet; `maturin
  develop` (editable, local) is the only verified installation path so
  far.
