//! Exchange-agnostic market data event types: a symbol identifier and a
//! timestamped top-of-book snapshot.

use crate::book::TopOfBook;

/// An opaque, dense integer identifier for a tradeable instrument.
///
/// Deliberately not a `String`/ticker: a `u32` costs nothing to copy, hash,
/// or compare on the hot path, and keeping "which venue/ticker does `42`
/// mean" as an external mapping (a symbol table, owned by whatever adapter
/// assigns the IDs) is what keeps this crate exchange-agnostic — it never
/// needs to know what a real ticker string looks like on any particular
/// venue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SymbolId(pub u32);

/// A single top-of-book observation: a symbol, a timestamp, a sequence
/// number, and the book state itself.
///
/// `sequence` is separate from `timestamp_ns` deliberately: real feeds can
/// (and do) emit multiple events sharing a timestamp at nanosecond
/// resolution, or occasionally emit out-of-order timestamps across
/// multiplexed sources — `sequence` is the unambiguous per-symbol
/// happens-before ordering a consumer should actually trust, with
/// `timestamp_ns` carried alongside for wall-clock-relative work (event
/// sampling by wall-clock horizon, per `docs/model-spec.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BookEvent {
    pub timestamp_ns: u64,
    pub sequence: u64,
    pub symbol: SymbolId,
    pub book: TopOfBook,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::book::BookValidationPolicy;
    use crate::price::PriceTicks;
    use crate::quantity::Quantity;

    #[test]
    fn symbol_id_equality_and_ordering_are_by_value() {
        assert_eq!(SymbolId(7), SymbolId(7));
        assert!(SymbolId(1) < SymbolId(2));
    }

    #[test]
    fn book_event_is_constructible_and_carries_its_fields() {
        let book = TopOfBook::new(
            PriceTicks(10000),
            Quantity(500),
            PriceTicks(10001),
            Quantity(300),
            BookValidationPolicy::RejectCrossedAndLocked,
        )
        .unwrap();
        let event = BookEvent {
            timestamp_ns: 1_700_000_000_000_000_000,
            sequence: 42,
            symbol: SymbolId(1),
            book,
        };
        assert_eq!(event.sequence, 42);
        assert_eq!(event.symbol, SymbolId(1));
        assert_eq!(event.book, book);
    }
}
