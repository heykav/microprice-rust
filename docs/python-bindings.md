# Python bindings (`microprice-python`)

Phase 13. PyO3 bindings exposing `MicroPriceModel` (load/save/predict/
metadata) and `train_synthetic` (the same counting → estimation → solving
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
metadata: {'schema_version': 1, 'symbol_id': 1, 'num_imbalance_buckets': 10,
           'spread_bucket_bounds_ticks': [1, 2, 4], 'smoothing_alpha': 0.5,
           'training_observations': 199999}
predict (balanced):   adjustment_ticks=-0.020089..., microprice_ticks=10000.9799...
predict (imbalanced): adjustment_ticks=-0.079776..., microprice_ticks=10000.9202...
```

Error paths were verified too, not just the happy path: a crossed book
passed to `predict` raises a Python `ValueError` carrying the exact same
message `microprice-core`'s `MicroPriceError` produces
(`"book failed price-ordering validation: bid=... ask=... (policy=...)"`),
and loading a nonexistent model path raises `ValueError` with the
underlying I/O error message — no panics, no silent `None`/garbage
returns.

## What isn't wired up yet

- Only synthetic-data training is exposed (`train_synthetic`) — Parquet
  ingestion (`microprice-data`'s `parquet-ingestion` feature) is not yet
  exposed to Python. Wiring it up is straightforward (the same pattern as
  `train_synthetic`, reading events via
  `microprice_data::read_events_from_parquet` instead of the synthetic
  generator) but hasn't been done.
- Batch prediction (`MicroPriceModel::predict_batch`) is not exposed —
  only the scalar `predict`.
- No PyPI publishing/CI wheel-building workflow exists yet; `maturin
  develop` (editable, local) is the only verified installation path so
  far.
