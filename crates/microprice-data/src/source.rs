//! The generic market-data adapter interface every data source (synthetic,
//! CSV, Parquet, ...) implements, so nothing downstream of it needs to know
//! which one it's reading from.

use microprice_core::BookEvent;

/// A source of chronologically-ordered [`BookEvent`]s.
///
/// Implementors are expected (not yet enforced by this trait) to emit
/// events in non-decreasing `sequence` order — consumers such as the
/// Phase 5 transition counter are explicitly specified to detect
/// out-of-order input themselves rather than trust the source blindly.
pub trait MarketDataSource {
    /// Returns the next event, or `None` once the source is exhausted.
    fn next_event(&mut self) -> Option<BookEvent>;
}
