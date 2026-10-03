//! Transition estimation, the micro-price solver, and model artifacts.
//!
//! [`transitions::TransitionCounter`] (streaming
//! transition counting), [`estimator`] (turns counts into `Q`/`G1` with
//! configurable smoothing), [`solver`] (the fixed-point `G*` solve), and
//! [`model::MicroPriceModel`] (the serializable, allocation-free-inference
//! trained artifact). See `docs/model-spec.md` for the exact math.
//!
//! ```
//! use microprice_calibration::{
//!     estimate, solve, ModelMetadata, MicroPriceModel, SmoothingConfig, SolverConfig,
//!     TransitionCounter, SCHEMA_VERSION,
//! };
//! use microprice_core::{
//!     BookEvent, BookValidationPolicy, ImbalanceBucketing, PriceTicks, Quantity,
//!     SpreadBucketing, StateSpaceConfig, SymbolId, TopOfBook,
//! };
//!
//! let space = StateSpaceConfig::new(ImbalanceBucketing::new(2)?, SpreadBucketing::new(vec![])?)?;
//! let event = |seq: u64, bid: i64, bq: u64, aq: u64| -> Result<BookEvent, Box<dyn std::error::Error>> {
//!     Ok(BookEvent {
//!         timestamp_ns: seq,
//!         sequence: seq,
//!         symbol: SymbolId(1),
//!         book: TopOfBook::new(
//!             PriceTicks(bid), Quantity(bq), PriceTicks(bid + 2), Quantity(aq),
//!             BookValidationPolicy::RejectCrossedAndLocked,
//!         )?,
//!     })
//! };
//! // A few hand-made events, only to show the API; far too few to mean anything.
//! let events = vec![
//!     event(0, 100, 10, 90)?, event(1, 100, 60, 40)?, event(2, 101, 10, 90)?,
//!     event(3, 101, 60, 40)?, event(4, 100, 10, 90)?, event(5, 100, 60, 40)?,
//! ];
//!
//! let mut counter = TransitionCounter::new(space.state_count());
//! counter.observe_events(&space, &events)?;
//! let estimated = estimate(&counter, SmoothingConfig::new(0.5)?)?;
//! let g_star = solve(&estimated, SolverConfig::DEFAULT)?;
//! let model = MicroPriceModel::new(
//!     ModelMetadata {
//!         schema_version: SCHEMA_VERSION,
//!         symbol_id: 1,
//!         num_imbalance_buckets: 2,
//!         spread_bucket_bounds_ticks: vec![],
//!         smoothing_alpha: 0.5,
//!         training_observations: counter.total_observations(),
//!     },
//!     g_star,
//!     estimated.p_up.clone(),
//!     estimated.visits.clone(),
//! )?;
//! let estimate = model.predict(&events[1].book)?;
//! assert_eq!(estimate.mid_ticks, 101.0);
//! assert_eq!(estimate.microprice_ticks, estimate.mid_ticks + estimate.adjustment_ticks);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![forbid(unsafe_code)]
// Enforces the README claim "no `unwrap()` in library code" (tests exempt).
#![cfg_attr(not(test), deny(clippy::unwrap_used))]

pub mod diagnostics;
pub mod error;
pub mod estimator;
pub mod model;
pub mod smoothing;
pub mod solver;
pub mod transitions;

pub use diagnostics::{antisymmetry_residual, martingale_diagnostic, MartingaleDiagnostic};
pub use error::CalibrationError;
pub use estimator::{estimate, EstimatedTransitions};
pub use model::{MicroPriceEstimate, MicroPriceModel, ModelMetadata, SCHEMA_VERSION};
pub use smoothing::SmoothingConfig;
pub use solver::{solve, solve_full_chain, SolverConfig};
pub use transitions::{mirror_state, TransitionCounter};
