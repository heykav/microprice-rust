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

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use microprice_calibration::{
    estimate, solve, MicroPriceEstimate, MicroPriceModel, ModelMetadata, SmoothingConfig,
    SolverConfig, TransitionCounter, SCHEMA_VERSION,
};
use microprice_core::{
    BookValidationPolicy, ImbalanceBucketing, PriceTicks, Quantity, SpreadBucketing,
    StateSpaceConfig, SymbolId, TopOfBook,
};
use microprice_data::{MarketDataSource, SyntheticConfig, SyntheticEventGenerator};

fn to_py_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Extracts a one-dimensional integer column from a Python object.
///
/// Accepts anything exposing a contiguous 1-D `int64` or `uint64` buffer
/// (NumPy arrays, `array.array('q'/'Q')`, ...), which is read without
/// per-element Python overhead, and otherwise falls back to any Python
/// sequence of integers (`list`, `tuple`, ...). Floats are rejected rather
/// than truncated. `what` names the column in error messages.
fn extract_int_column(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<i64>> {
    if let Ok(buf) = PyBuffer::<i64>::get(obj) {
        if buf.dimensions() == 1 {
            return buf.to_vec(obj.py());
        }
    }
    if let Ok(buf) = PyBuffer::<u64>::get(obj) {
        if buf.dimensions() == 1 {
            return buf
                .to_vec(obj.py())?
                .into_iter()
                .map(|v| {
                    i64::try_from(v)
                        .map_err(|_| PyValueError::new_err(format!("{what}: value {v} too large")))
                })
                .collect();
        }
    }
    obj.extract::<Vec<i64>>().map_err(|e| {
        PyValueError::new_err(format!(
            "{what} must be a 1-D int64/uint64 buffer (e.g. a NumPy array) or a sequence of integers: {e}"
        ))
    })
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
    /// `adjustment_ticks`, `state_id`, `state_observations`, and `p_up` -
    /// the full `MicroPriceEstimate`, not just a bare number.
    ///
    /// `p_up` is `None` when training never observed a directional move out
    /// of this book's state; it is a real probability otherwise. That's
    /// Python `None`, deliberately not `0.5` - the model does not invent a
    /// coin flip for a state it has no evidence about.
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
        dict.set_item("p_up", est.p_up)?;
        Ok(dict.into())
    }

    /// Vectorised `predict`: one call for many books. Each argument is a
    /// 1-D array of the same length - a NumPy `int64`/`uint64` array, an
    /// `array.array`, or a plain list/tuple of ints. Prices are integer
    /// ticks; quantities must be non-negative.
    ///
    /// Returns a dict of equal-length lists keyed like `predict`'s dict
    /// (`mid_ticks`, `weighted_mid_ticks`, `microprice_ticks`,
    /// `adjustment_ticks`, `state_id`, `state_observations`, `p_up`), where
    /// `p_up` entries are `None` for states with no directional evidence.
    /// Row `k` is exactly what `predict` returns for the `k`-th book.
    ///
    /// Raises `ValueError` if the lengths differ, an argument is not an
    /// integer column, or any book is invalid (crossed/locked, both sizes
    /// zero); the message names the first offending row index. Nothing is
    /// returned for a partially valid batch.
    fn predict_batch(
        &self,
        py: Python<'_>,
        bid_price_ticks: &Bound<'_, PyAny>,
        bid_qty: &Bound<'_, PyAny>,
        ask_price_ticks: &Bound<'_, PyAny>,
        ask_qty: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyDict>> {
        let bp = extract_int_column(bid_price_ticks, "bid_price_ticks")?;
        let bq = extract_int_column(bid_qty, "bid_qty")?;
        let ap = extract_int_column(ask_price_ticks, "ask_price_ticks")?;
        let aq = extract_int_column(ask_qty, "ask_qty")?;
        let n = bp.len();
        if bq.len() != n || ap.len() != n || aq.len() != n {
            return Err(PyValueError::new_err(format!(
                "column lengths differ: bid_price_ticks={}, bid_qty={}, ask_price_ticks={}, ask_qty={}",
                n,
                bq.len(),
                ap.len(),
                aq.len()
            )));
        }
        let to_qty = |v: i64, what: &str, k: usize| -> PyResult<Quantity> {
            u64::try_from(v)
                .map(Quantity)
                .map_err(|_| PyValueError::new_err(format!("row {k}: {what} is negative ({v})")))
        };
        let mut books = Vec::with_capacity(n);
        for k in 0..n {
            let book = TopOfBook::new(
                PriceTicks(bp[k]),
                to_qty(bq[k], "bid_qty", k)?,
                PriceTicks(ap[k]),
                to_qty(aq[k], "ask_qty", k)?,
                BookValidationPolicy::RejectCrossedAndLocked,
            )
            .map_err(|e| PyValueError::new_err(format!("row {k}: {e}")))?;
            books.push(book);
        }
        let blank = MicroPriceEstimate {
            mid_ticks: 0.0,
            weighted_mid_ticks: 0.0,
            microprice_ticks: 0.0,
            adjustment_ticks: 0.0,
            state_id: 0,
            state_observations: 0,
            p_up: None,
        };
        let mut out = vec![blank; n];
        if self.inner.predict_batch(&books, &mut out).is_err() {
            // Locate the first failing row for a useful message.
            for (k, b) in books.iter().enumerate() {
                if let Err(e) = self.inner.predict(b) {
                    return Err(PyValueError::new_err(format!("row {k}: {e}")));
                }
            }
            return Err(PyValueError::new_err("batch prediction failed"));
        }
        let dict = PyDict::new(py);
        dict.set_item(
            "mid_ticks",
            out.iter().map(|e| e.mid_ticks).collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "weighted_mid_ticks",
            out.iter().map(|e| e.weighted_mid_ticks).collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "microprice_ticks",
            out.iter().map(|e| e.microprice_ticks).collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "adjustment_ticks",
            out.iter().map(|e| e.adjustment_ticks).collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "state_id",
            out.iter().map(|e| e.state_id).collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "state_observations",
            out.iter().map(|e| e.state_observations).collect::<Vec<_>>(),
        )?;
        dict.set_item("p_up", PyList::new(py, out.iter().map(|e| e.p_up))?)?;
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
    let inner = MicroPriceModel::new(
        metadata,
        g_star,
        estimated.p_up.clone(),
        estimated.visits.clone(),
    );
    Ok(PyMicroPriceModel { inner })
}

#[pymodule]
fn microprice_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMicroPriceModel>()?;
    m.add_function(wrap_pyfunction!(train_synthetic, m)?)?;
    Ok(())
}
