//! PyO3 bindings for MicroPrice-Rust (Phase 13).
//!
//! Exposes `MicroPriceModel` (load/save/predict/metadata) and
//! `train_synthetic` (the same counting -> estimation -> solving pipeline
//! `microprice-cli`'s `train` subcommand runs, against the same synthetic
//! generator - real-data ingestion from Python is only as available as it
//! is from Rust, i.e. Parquet via `microprice-data`'s `parquet-ingestion`
//! feature, not yet exposed here).
//!
//! Built and verified via `maturin build`/`maturin develop`, **not** the
//! root workspace's `cargo test` - see this crate's `Cargo.toml` for why
//! it's deliberately its own workspace, and `docs/python-bindings.md` for
//! the exact verification steps actually run.

use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use microprice_calibration::{
    estimate, solve, MicroPriceModel, ModelMetadata, SmoothingConfig, SolverConfig,
    TransitionCounter, SCHEMA_VERSION,
};
use microprice_core::{
    BookValidationPolicy, ImbalanceBucketing, PriceTicks, Quantity, SpreadBucketing,
    StateSpaceConfig, SymbolId, TopOfBook,
};
use microprice_data::{MarketDataSource, SyntheticConfig, SyntheticEventGenerator};

fn to_py_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// A calibrated micro-price model, loadable from (and savable to) the same
/// `bincode` artifact format `microprice-cli`/`microprice-calibration` use.
#[pyclass(name = "MicroPriceModel")]
struct PyMicroPriceModel {
    inner: MicroPriceModel,
}

#[pymethods]
impl PyMicroPriceModel {
    /// Loads a model artifact previously saved by this class or by
    /// `microprice train`.
    #[staticmethod]
    fn load(path: PathBuf) -> PyResult<Self> {
        let inner = MicroPriceModel::load(&path).map_err(to_py_err)?;
        Ok(PyMicroPriceModel { inner })
    }

    /// Saves this model to `path` (plus a `<path>.json` metadata sidecar).
    fn save(&self, path: PathBuf) -> PyResult<()> {
        self.inner.save(&path).map_err(to_py_err)
    }

    /// Predicts the micro-price for one top-of-book snapshot, returning a
    /// dict with `mid_ticks`, `weighted_mid_ticks`, `microprice_ticks`,
    /// `adjustment_ticks`, `state_id`, and `state_observations` - the full
    /// `MicroPriceEstimate`, not just a bare number.
    fn predict(
        &self,
        py: Python<'_>,
        bid_price_ticks: i64,
        bid_qty: u64,
        ask_price_ticks: i64,
        ask_qty: u64,
    ) -> PyResult<Py<PyDict>> {
        let book = TopOfBook::new(
            PriceTicks(bid_price_ticks),
            Quantity(bid_qty),
            PriceTicks(ask_price_ticks),
            Quantity(ask_qty),
            BookValidationPolicy::RejectCrossedAndLocked,
        )
        .map_err(to_py_err)?;
        let est = self.inner.predict(&book).map_err(to_py_err)?;

        let dict = PyDict::new(py);
        dict.set_item("mid_ticks", est.mid_ticks)?;
        dict.set_item("weighted_mid_ticks", est.weighted_mid_ticks)?;
        dict.set_item("microprice_ticks", est.microprice_ticks)?;
        dict.set_item("adjustment_ticks", est.adjustment_ticks)?;
        dict.set_item("state_id", est.state_id)?;
        dict.set_item("state_observations", est.state_observations)?;
        Ok(dict.into())
    }

    /// This model's metadata (schema version, bucketing configuration,
    /// smoothing alpha, training observation count) as a dict.
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let meta = self.inner.metadata();
        let dict = PyDict::new(py);
        dict.set_item("schema_version", meta.schema_version)?;
        dict.set_item("symbol_id", meta.symbol_id)?;
        dict.set_item("num_imbalance_buckets", meta.num_imbalance_buckets)?;
        dict.set_item(
            "spread_bucket_bounds_ticks",
            meta.spread_bucket_bounds_ticks.clone(),
        )?;
        dict.set_item("smoothing_alpha", meta.smoothing_alpha)?;
        dict.set_item("training_observations", meta.training_observations)?;
        Ok(dict.into())
    }
}

/// Generates a synthetic `BookEvent` stream and runs the full
/// counting -> estimation -> solving pipeline against it, returning a
/// calibrated `MicroPriceModel` - the same pipeline `microprice-cli`'s
/// `train` subcommand runs, exposed directly to Python. Synthetic data is
/// the only data source available from Python today, same as from the
/// CLI (see this crate's module docs).
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    num_events,
    num_imbalance_buckets = 20,
    spread_bucket_bounds_ticks = vec![],
    smoothing_alpha = 0.5,
    seed = 42,
    symbol_id = 1,
    initial_mid_ticks = 10_000,
    initial_spread_ticks = 2,
    initial_bid_qty = 500,
    initial_ask_qty = 500,
    arrival_rate = 0.3,
    cancel_rate = 0.2,
    market_order_rate = 0.2,
    imbalance_persistence = 0.5,
    price_move_probability = 0.1,
))]
fn train_synthetic(
    num_events: u64,
    num_imbalance_buckets: u32,
    spread_bucket_bounds_ticks: Vec<i64>,
    smoothing_alpha: f64,
    seed: u64,
    symbol_id: u32,
    initial_mid_ticks: i64,
    initial_spread_ticks: i64,
    initial_bid_qty: u64,
    initial_ask_qty: u64,
    arrival_rate: f64,
    cancel_rate: f64,
    market_order_rate: f64,
    imbalance_persistence: f64,
    price_move_probability: f64,
) -> PyResult<PyMicroPriceModel> {
    let imbalance = ImbalanceBucketing::new(num_imbalance_buckets).map_err(to_py_err)?;
    let spread = SpreadBucketing::new(spread_bucket_bounds_ticks.clone()).map_err(to_py_err)?;
    let state_space = StateSpaceConfig::new(imbalance, spread);

    let synth_config = SyntheticConfig {
        symbol: SymbolId(symbol_id),
        initial_mid_ticks,
        initial_spread_ticks,
        initial_bid_qty,
        initial_ask_qty,
        arrival_rate,
        cancel_rate,
        market_order_rate,
        imbalance_persistence,
        price_move_probability,
        seed,
    };
    let mut generator = SyntheticEventGenerator::new(synth_config).map_err(to_py_err)?;
    let events: Vec<_> = (0..num_events)
        .map(|_| {
            generator
                .next_event()
                .expect("SyntheticEventGenerator::next_event never returns None")
        })
        .collect();

    let mut counter = TransitionCounter::new(state_space.state_count());
    counter
        .observe_events(&state_space, &events)
        .map_err(to_py_err)?;

    let smoothing = SmoothingConfig::new(smoothing_alpha).map_err(to_py_err)?;
    let estimated = estimate(&counter, smoothing).map_err(to_py_err)?;
    let g_star = solve(&estimated, SolverConfig::DEFAULT).map_err(to_py_err)?;

    let metadata = ModelMetadata {
        schema_version: SCHEMA_VERSION,
        symbol_id,
        num_imbalance_buckets,
        spread_bucket_bounds_ticks,
        smoothing_alpha,
        training_observations: counter.total_observations(),
    };
    let inner = MicroPriceModel::new(metadata, g_star, estimated.visits.clone());
    Ok(PyMicroPriceModel { inner })
}

#[pymodule]
fn microprice_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMicroPriceModel>()?;
    m.add_function(wrap_pyfunction!(train_synthetic, m)?)?;
    Ok(())
}
