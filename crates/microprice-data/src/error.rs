//! Typed errors for `microprice-data`.

use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum DataError {
    /// A [`crate::synthetic::SyntheticConfig`] had an invalid rate (outside
    /// `[0.0, 1.0]`), rates summing above `1.0`, or a sub-1-tick initial
    /// spread. Raised at construction, never mid-generation.
    #[error("invalid synthetic generator configuration: {reason}")]
    InvalidSyntheticConfig { reason: String },

    /// A file, schema, or encoding failure while reading/writing a Parquet
    /// file (feature `parquet-ingestion`) — everything from a missing file
    /// to a missing/mistyped column, wrapped so callers only need to match
    /// one error type.
    #[cfg(feature = "parquet-ingestion")]
    #[error("Parquet I/O error: {0}")]
    ParquetIo(String),

    /// A row in an ingested Parquet file failed `TopOfBook` validation
    /// (e.g. a crossed book under a policy that rejects crossed markets) —
    /// reported with its row index rather than silently dropped or
    /// aborting the whole file read with no indication of which row was
    /// bad.
    #[cfg(feature = "parquet-ingestion")]
    #[error("row {row} failed book validation: {reason}")]
    InvalidRow { row: usize, reason: String },
}
