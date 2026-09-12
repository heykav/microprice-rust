//! Market data source adapters.
//!
//! As of Phase 4: a `MarketDataSource` trait and a deterministic synthetic
//! event generator (the project's development dataset — see the module
//! docs on [`synthetic`] for exactly what it does and doesn't claim). The
//! CSV/Parquet adapters are Phase 13.

#![forbid(unsafe_code)]

pub mod error;
pub mod source;
pub mod synthetic;

pub use error::DataError;
pub use source::MarketDataSource;
pub use synthetic::{SyntheticConfig, SyntheticEventGenerator};
