//! Transition estimation, the micro-price solver, and model artifacts.
//!
//! As of Phases 5-9: [`transitions::TransitionCounter`] (streaming
//! transition counting), [`estimator`] (turns counts into `Q`/`G1` with
//! configurable smoothing), [`solver`] (the fixed-point `G*` solve), and
//! [`model::MicroPriceModel`] (the serializable, allocation-free-inference
//! trained artifact). See `docs/model-spec.md` for the exact math.

#![forbid(unsafe_code)]

pub mod error;
pub mod estimator;
pub mod model;
pub mod smoothing;
pub mod solver;
pub mod transitions;

pub use error::CalibrationError;
pub use estimator::{estimate, EstimatedTransitions};
pub use model::{MicroPriceEstimate, MicroPriceModel, ModelMetadata, SCHEMA_VERSION};
pub use smoothing::SmoothingConfig;
pub use solver::{solve, SolverConfig};
pub use transitions::TransitionCounter;
