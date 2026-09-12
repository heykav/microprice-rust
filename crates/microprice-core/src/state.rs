//! The `StateId` primitive.
//!
//! Phase 1 introduces only the opaque type. Mapping a [`crate::book::TopOfBook`]
//! to a `StateId` — bucket counts, bucketing method, dimensionality beyond
//! imbalance/spread — is Phase 3's state-encoder concern and is deliberately
//! not implemented or guessed at here; see `docs/model-spec.md`'s "State"
//! section and its open questions.

/// An opaque, deterministic, contiguous index into some (as-yet-undefined,
/// Phase 3) discretization of order-book state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StateId(pub u32);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equality_and_ordering_are_by_value() {
        assert_eq!(StateId(3), StateId(3));
        assert!(StateId(1) < StateId(2));
    }
}
