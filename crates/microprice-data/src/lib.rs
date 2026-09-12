//! Market data source adapters.
//!
//! A `MarketDataSource` trait, a deterministic synthetic event generator
//! (the project's development dataset — see the module docs on
//! [`synthetic`] for exactly what it does and doesn't claim), and (behind
//! the `parquet-ingestion` Cargo feature, off by default, so not an
//! intra-doc link here — this crate's default `cargo doc` build has the
//! module compiled out) Parquet ingestion. See that module's own docs
//! (`src/parquet.rs`, built with `--features parquet-ingestion`) for the
//! schema and why the feature defaults off. A CSV adapter is not yet
//! implemented.

#![forbid(unsafe_code)]

pub mod error;
#[cfg(feature = "parquet-ingestion")]
pub mod parquet;
pub mod source;
pub mod synthetic;

pub use error::DataError;
#[cfg(feature = "parquet-ingestion")]
pub use parquet::{read_events_from_parquet, write_events_to_parquet};
pub use source::MarketDataSource;
pub use synthetic::{SyntheticConfig, SyntheticEventGenerator};
