//! Parquet ingestion (Phase 12), gated behind the `parquet-ingestion`
//! Cargo feature — arrow/parquet pull in a substantial dependency tree
//! that only matters to callers actually reading/writing Parquet files, so
//! the base crate (and everyone who only wants the synthetic generator)
//! doesn't pay for it. Not enabled by default; see the crate's `Cargo.toml`.
//!
//! **Schema, made explicit** (one row per [`BookEvent`]): this is this
//! crate's own column convention, not a standard exchange format —
//! disclosed here rather than assumed obvious.
//!
//! | column            | type    |
//! |-------------------|---------|
//! | `timestamp_ns`    | UInt64  |
//! | `sequence`        | UInt64  |
//! | `symbol_id`       | UInt32  |
//! | `bid_price_ticks` | Int64   |
//! | `bid_qty`         | UInt64  |
//! | `ask_price_ticks` | Int64   |
//! | `ask_qty`         | UInt64  |
//!
//! Every row read back is passed through [`TopOfBook::new`] with the
//! caller-supplied [`BookValidationPolicy`] — a row that fails validation
//! (e.g. a crossed book under a rejecting policy) is reported as a typed
//! [`DataError::InvalidRow`] carrying its row index, not silently dropped.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, Int64Array, RecordBatch, UInt32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;

use microprice_core::{BookEvent, BookValidationPolicy, PriceTicks, Quantity, SymbolId, TopOfBook};

use crate::error::DataError;

fn schema() -> Schema {
    Schema::new(vec![
        Field::new("timestamp_ns", DataType::UInt64, false),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("symbol_id", DataType::UInt32, false),
        Field::new("bid_price_ticks", DataType::Int64, false),
        Field::new("bid_qty", DataType::UInt64, false),
        Field::new("ask_price_ticks", DataType::Int64, false),
        Field::new("ask_qty", DataType::UInt64, false),
    ])
}

/// Writes `events` to a Parquet file at `path`, one row per event, in the
/// schema documented on this module. Does not itself re-validate the
/// books (they are already-constructed, already-valid `TopOfBook`s by the
/// time they're `BookEvent`s) — validation happens on the *read* side,
/// where untrusted bytes actually enter the system.
pub fn write_events_to_parquet(
    events: &[BookEvent],
    path: impl AsRef<Path>,
) -> Result<(), DataError> {
    let schema = Arc::new(schema());

    let timestamp_ns: UInt64Array = events.iter().map(|e| e.timestamp_ns).collect();
    let sequence: UInt64Array = events.iter().map(|e| e.sequence).collect();
    let symbol_id: UInt32Array = events.iter().map(|e| e.symbol.0).collect();
    let bid_price_ticks: Int64Array = events.iter().map(|e| e.book.bid_price.0).collect();
    let bid_qty: UInt64Array = events.iter().map(|e| e.book.bid_qty.0).collect();
    let ask_price_ticks: Int64Array = events.iter().map(|e| e.book.ask_price.0).collect();
    let ask_qty: UInt64Array = events.iter().map(|e| e.book.ask_qty.0).collect();

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(timestamp_ns),
            Arc::new(sequence),
            Arc::new(symbol_id),
            Arc::new(bid_price_ticks),
            Arc::new(bid_qty),
            Arc::new(ask_price_ticks),
            Arc::new(ask_qty),
        ],
    )
    .map_err(|e| DataError::ParquetIo(e.to_string()))?;

    let file = File::create(path.as_ref()).map_err(|e| DataError::ParquetIo(e.to_string()))?;
    let mut writer = ArrowWriter::try_new(file, schema, None)
        .map_err(|e| DataError::ParquetIo(e.to_string()))?;
    writer
        .write(&batch)
        .map_err(|e| DataError::ParquetIo(e.to_string()))?;
    writer
        .close()
        .map_err(|e| DataError::ParquetIo(e.to_string()))?;
    Ok(())
}

fn column<'a, T: 'static>(batch: &'a RecordBatch, name: &str) -> Result<&'a T, DataError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| DataError::ParquetIo(format!("missing column {name:?}")))?
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| DataError::ParquetIo(format!("column {name:?} had an unexpected type")))
}

/// Reads every row of the Parquet file at `path` into a `Vec<BookEvent>`,
/// in file (row-group, then row) order, validating each row's book against
/// `policy`.
///
/// A row that fails book validation aborts the read with
/// [`DataError::InvalidRow`] (naming the row index) rather than silently
/// skipping bad rows — unlike `TransitionCounter::observe_events`'s
/// skip-on-encode-failure behavior, this is a data *ingestion* boundary,
/// where a malformed row is a real data-quality problem worth stopping for,
/// not a normal, expected in-domain occurrence.
pub fn read_events_from_parquet(
    path: impl AsRef<Path>,
    policy: BookValidationPolicy,
) -> Result<Vec<BookEvent>, DataError> {
    let file = File::open(path.as_ref()).map_err(|e| DataError::ParquetIo(e.to_string()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| DataError::ParquetIo(e.to_string()))?
        .build()
        .map_err(|e| DataError::ParquetIo(e.to_string()))?;

    let mut events = Vec::new();
    let mut row_index = 0usize;
    for batch_result in reader {
        let batch = batch_result.map_err(|e| DataError::ParquetIo(e.to_string()))?;
        let timestamp_ns = column::<UInt64Array>(&batch, "timestamp_ns")?;
        let sequence = column::<UInt64Array>(&batch, "sequence")?;
        let symbol_id = column::<UInt32Array>(&batch, "symbol_id")?;
        let bid_price_ticks = column::<Int64Array>(&batch, "bid_price_ticks")?;
        let bid_qty = column::<UInt64Array>(&batch, "bid_qty")?;
        let ask_price_ticks = column::<Int64Array>(&batch, "ask_price_ticks")?;
        let ask_qty = column::<UInt64Array>(&batch, "ask_qty")?;

        for i in 0..batch.num_rows() {
            let book = TopOfBook::new(
                PriceTicks(bid_price_ticks.value(i)),
                Quantity(bid_qty.value(i)),
                PriceTicks(ask_price_ticks.value(i)),
                Quantity(ask_qty.value(i)),
                policy,
            )
            .map_err(|e| DataError::InvalidRow {
                row: row_index,
                reason: e.to_string(),
            })?;
            events.push(BookEvent {
                timestamp_ns: timestamp_ns.value(i),
                sequence: sequence.value(i),
                symbol: SymbolId(symbol_id.value(i)),
                book,
            });
            row_index += 1;
        }
    }
    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;
    use microprice_core::BookValidationPolicy;

    fn event(seq: u64, mid: i64) -> BookEvent {
        BookEvent {
            timestamp_ns: seq * 1000,
            sequence: seq,
            symbol: SymbolId(3),
            book: TopOfBook::new(
                PriceTicks(mid - 1),
                Quantity(100 + seq),
                PriceTicks(mid + 1),
                Quantity(200 + seq),
                BookValidationPolicy::RejectCrossedAndLocked,
            )
            .unwrap(),
        }
    }

    #[test]
    fn write_then_read_round_trips_every_event_exactly() {
        let events: Vec<_> = (0..250)
            .map(|i| event(i, 10_000 + (i as i64 % 7)))
            .collect();
        let dir =
            std::env::temp_dir().join(format!("microprice-parquet-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("events.parquet");

        write_events_to_parquet(&events, &path).unwrap();
        let read_back =
            read_events_from_parquet(&path, BookValidationPolicy::RejectCrossedAndLocked).unwrap();

        assert_eq!(events, read_back);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn read_reports_the_row_index_of_a_book_that_fails_validation() {
        // Hand-build a batch with one crossed book (bid > ask) at row 2,
        // bypassing write_events_to_parquet (which can only ever write
        // already-valid BookEvents) so the read-side validation path is
        // actually exercised against real bad bytes.
        let dir = std::env::temp_dir().join(format!(
            "microprice-parquet-bad-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.parquet");

        let schema = Arc::new(schema());
        let timestamp_ns: UInt64Array = vec![0u64, 1, 2].into_iter().collect();
        let sequence: UInt64Array = vec![0u64, 1, 2].into_iter().collect();
        let symbol_id: UInt32Array = vec![1u32, 1, 1].into_iter().collect();
        // Row 2: bid (10005) > ask (10000) - crossed.
        let bid_price_ticks: Int64Array = vec![9999i64, 9999, 10005].into_iter().collect();
        let bid_qty: UInt64Array = vec![100u64, 100, 100].into_iter().collect();
        let ask_price_ticks: Int64Array = vec![10001i64, 10001, 10000].into_iter().collect();
        let ask_qty: UInt64Array = vec![100u64, 100, 100].into_iter().collect();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(timestamp_ns),
                Arc::new(sequence),
                Arc::new(symbol_id),
                Arc::new(bid_price_ticks),
                Arc::new(bid_qty),
                Arc::new(ask_price_ticks),
                Arc::new(ask_qty),
            ],
        )
        .unwrap();
        let file = File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let result = read_events_from_parquet(&path, BookValidationPolicy::RejectCrossedAndLocked);
        assert_eq!(
            result,
            Err(DataError::InvalidRow {
                row: 2,
                reason: "book failed price-ordering validation: bid=10005 ask=10000 \
                         (policy=RejectCrossedAndLocked)"
                    .to_string(),
            })
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn read_fails_cleanly_on_a_nonexistent_file() {
        let result = read_events_from_parquet(
            "/tmp/microprice-does-not-exist-12345.parquet",
            BookValidationPolicy::RejectCrossedAndLocked,
        );
        assert!(matches!(result, Err(DataError::ParquetIo(_))));
    }
}
