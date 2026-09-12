//! Chronological train/test splitting.
//!
//! **Deliberately not a random split.** A model must be evaluated only
//! against data that comes *after* everything it was calibrated on — a
//! random shuffle would leak future information into training (the same
//! mistake as shuffling a time series before cross-validation) and produce
//! an optimistic, meaningless out-of-sample number. `events` is assumed to
//! already be in chronological (`sequence`-increasing) order, which is the
//! same assumption `microprice_calibration::TransitionCounter::observe_events`
//! makes and enforces.

use microprice_core::BookEvent;

use crate::error::EvalError;

/// Splits `events` at `train_fraction` of its length: `events[..split_at]`
/// for training, `events[split_at..]` for out-of-sample evaluation.
///
/// `train_fraction` must be strictly between `0.0` and `1.0`, and the split
/// must leave at least one event on each side — a split that would starve
/// either side is a configuration error, not something to silently clamp.
pub fn chronological_split(
    events: &[BookEvent],
    train_fraction: f64,
) -> Result<(&[BookEvent], &[BookEvent]), EvalError> {
    if events.is_empty() {
        return Err(EvalError::EmptyDataset);
    }
    if !(train_fraction > 0.0 && train_fraction < 1.0) {
        return Err(EvalError::InvalidTrainFraction { train_fraction });
    }
    let split_at = ((events.len() as f64) * train_fraction).round() as usize;
    if split_at == 0 || split_at >= events.len() {
        return Err(EvalError::InsufficientEvents {
            reason: format!(
                "train_fraction {train_fraction} on {} events would leave one side empty \
                 (computed split point {split_at})",
                events.len()
            ),
        });
    }
    Ok(events.split_at(split_at))
}

#[cfg(test)]
mod tests {
    use super::*;
    use microprice_core::{BookValidationPolicy, PriceTicks, Quantity, SymbolId, TopOfBook};

    fn event(seq: u64) -> BookEvent {
        BookEvent {
            timestamp_ns: seq * 1000,
            sequence: seq,
            symbol: SymbolId(1),
            book: TopOfBook::new(
                PriceTicks(10_000),
                Quantity(500),
                PriceTicks(10_002),
                Quantity(500),
                BookValidationPolicy::RejectCrossedAndLocked,
            )
            .unwrap(),
        }
    }

    #[test]
    fn splits_at_the_requested_fraction() {
        let events: Vec<_> = (0..100).map(event).collect();
        let (train, test) = chronological_split(&events, 0.7).unwrap();
        assert_eq!(train.len(), 70);
        assert_eq!(test.len(), 30);
        // Chronological, not shuffled: the boundary is contiguous.
        assert_eq!(train.last().unwrap().sequence, 69);
        assert_eq!(test.first().unwrap().sequence, 70);
    }

    #[test]
    fn rejects_an_empty_dataset() {
        assert_eq!(chronological_split(&[], 0.5), Err(EvalError::EmptyDataset));
    }

    #[test]
    fn rejects_a_fraction_outside_the_open_unit_interval() {
        let events: Vec<_> = (0..10).map(event).collect();
        assert!(matches!(
            chronological_split(&events, 0.0),
            Err(EvalError::InvalidTrainFraction { .. })
        ));
        assert!(matches!(
            chronological_split(&events, 1.0),
            Err(EvalError::InvalidTrainFraction { .. })
        ));
        assert!(matches!(
            chronological_split(&events, 1.5),
            Err(EvalError::InvalidTrainFraction { .. })
        ));
    }

    #[test]
    fn a_fraction_that_would_starve_one_side_of_a_tiny_dataset_is_an_error() {
        let events: Vec<_> = (0..2).map(event).collect();
        // 0.99 of 2 events rounds to 2 -> test side would be empty.
        assert!(matches!(
            chronological_split(&events, 0.99),
            Err(EvalError::InsufficientEvents { .. })
        ));
    }

    #[test]
    fn a_single_event_dataset_always_starves_one_side() {
        let events = [event(0)];
        assert!(matches!(
            chronological_split(&events, 0.5),
            Err(EvalError::InsufficientEvents { .. })
        ));
    }
}
