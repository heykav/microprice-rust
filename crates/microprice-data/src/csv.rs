//! Level-1 (best bid / best ask) CSV ingestion.
//!
//! Turns a CSV of top-of-book quotes into the same [`BookEvent`] stream the
//! synthetic generator and the Parquet reader produce, so calibration and
//! evaluation need no changes to consume real data.
//!
//! ```
//! use std::io::Cursor;
//! use microprice_data::csv::{read_csv_events, CsvIngestConfig};
//!
//! // A made-up two-row file in the Binance bookTicker layout (format demo only).
//! let csv = "update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,transaction_time,event_time\n\
//!            1,100.0,2.0,100.1,1.0,1700000000000,1700000000001\n\
//!            2,100.0,3.0,100.1,1.0,1700000000010,1700000000011\n";
//! let config = CsvIngestConfig::binance_book_ticker(0.1); // tick size is required
//! let ingest = read_csv_events(Cursor::new(csv), &config, None)?;
//! assert_eq!(ingest.events.len(), 2);
//! // Prices are integer half-tick units: 100.0 / 0.1 = 1000 ticks -> 2000 units.
//! assert_eq!(ingest.events[0].book.bid_price.0, 2000);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Schemas
//!
//! A schema is a set of column references (by header name or by position)
//! plus a timestamp unit. Two vendor presets are provided:
//!
//! * [`CsvIngestConfig::binance_book_ticker`] - Binance USD-M futures daily
//!   `bookTicker` files from `data.binance.vision`: header row
//!   `update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,transaction_time,event_time`,
//!   timestamps in milliseconds since the Unix epoch (`event_time` is used).
//! * [`CsvIngestConfig::lobster_orderbook`] - the LOBSTER `orderbook` file
//!   (no header; the first four columns are ask price, ask size, bid price,
//!   bid size of level 1; prices are dollars x 10,000). LOBSTER keeps
//!   timestamps in the paired `message` file (first column, seconds after
//!   midnight), so use [`read_lobster_files`].
//!
//! **Both vendor mappings are UNVERIFIED against the vendors' own
//! documentation** (the development sandbox could not reach either site);
//! they follow the column layouts as described to the author. If a file does
//! not match, the reader fails with a typed error naming the missing column
//! rather than guessing. See `docs/real-data-evaluation.md`.
//!
//! ## Price and quantity conversion
//!
//! Prices become integer [`PriceTicks`] *model units*: `price / tick_size`
//! must land on an integer number of exchange ticks (else the row is invalid as
//! off-grid), which is then multiplied by `resolution`. With
//! `resolution = 2` (the default) one model unit is half an exchange tick,
//! which keeps the mid-price `(bid + ask) / 2` exactly representable even
//! when the spread is an odd number of ticks. (The core crate's
//! `TopOfBook::mid_price_ticks` truncates, which on a one-tick spread would
//! silently make "mid" equal the bid.) Quantities are scaled by
//! `10^qty_decimals` and rounded to `u64`; imbalance is scale-free, so this
//! only affects rounding.
//!
//! ## Invalid rows
//!
//! By default any invalid row (unparsable field, crossed/locked book, both
//! sides empty, off-grid price, decreasing timestamp, vendor "empty level"
//! sentinel) aborts the read with an error naming the line. With
//! [`InvalidRowPolicy::Skip`] such rows are dropped and counted per reason in
//! [`CsvIngest::skipped`], so nothing is discarded silently.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use microprice_core::{BookEvent, BookValidationPolicy, PriceTicks, Quantity, SymbolId, TopOfBook};

use crate::error::DataError;

/// A CSV column, addressed by header name or by zero-based position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ColumnRef {
    Name(String),
    Index(usize),
}

impl ColumnRef {
    fn named(s: &str) -> Self {
        ColumnRef::Name(s.to_string())
    }
}

/// Unit of the timestamp column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimestampUnit {
    Seconds,
    Milliseconds,
    Microseconds,
    Nanoseconds,
}

impl TimestampUnit {
    fn ns_per_unit(self) -> u64 {
        match self {
            TimestampUnit::Seconds => 1_000_000_000,
            TimestampUnit::Milliseconds => 1_000_000,
            TimestampUnit::Microseconds => 1_000,
            TimestampUnit::Nanoseconds => 1,
        }
    }
}

/// Which columns hold what.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsvSchema {
    /// Whether the first non-blank line is a header. Named columns require it.
    pub has_header: bool,
    /// `None` means timestamps are supplied externally (LOBSTER).
    pub timestamp: Option<ColumnRef>,
    pub timestamp_unit: TimestampUnit,
    pub bid_price: ColumnRef,
    pub bid_qty: ColumnRef,
    pub ask_price: ColumnRef,
    pub ask_qty: ColumnRef,
}

/// What to do with a row that cannot become a valid [`BookEvent`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvalidRowPolicy {
    /// Abort with [`DataError::CsvRow`] naming the line.
    Error,
    /// Drop the row and count it in [`CsvIngest::skipped`].
    Skip,
}

/// Why a row was invalid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SkipReason {
    Unparsable,
    /// Vendor "no order at this level" sentinel, or a non-positive price.
    EmptyLevel,
    OffGridPrice,
    CrossedOrLocked,
    /// Zero size on both sides (imbalance is undefined).
    EmptyBook,
    /// Timestamp earlier than the previous accepted row's.
    TimestampDecreased,
}

impl SkipReason {
    pub fn as_str(self) -> &'static str {
        match self {
            SkipReason::Unparsable => "unparsable field",
            SkipReason::EmptyLevel => "empty level / non-positive price",
            SkipReason::OffGridPrice => "price not on the tick grid",
            SkipReason::CrossedOrLocked => "crossed or locked book",
            SkipReason::EmptyBook => "zero size on both sides",
            SkipReason::TimestampDecreased => "timestamp decreased",
        }
    }
}

/// Full ingestion configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct CsvIngestConfig {
    pub schema: CsvSchema,
    pub symbol: SymbolId,
    /// Raw price divided by this gives a currency price (LOBSTER: 10,000).
    pub price_divisor: f64,
    /// Currency price of one exchange tick (e.g. `0.1` for BTCUSDT perp).
    pub tick_size: f64,
    /// Model units per exchange tick; `>= 1`. `2` gives exact mid-prices.
    pub resolution: u32,
    /// Decimal digits of quantity preserved (`qty * 10^decimals`, rounded).
    pub qty_decimals: u32,
    pub on_invalid_row: InvalidRowPolicy,
    /// Drop a row whose L1 (prices and sizes) equals the previous accepted
    /// row's. Off by default; see the methodology doc for why it matters.
    pub drop_unchanged: bool,
    /// Stop after this many accepted events (a chronological prefix).
    pub max_events: Option<usize>,
}

impl CsvIngestConfig {
    /// Binance USD-M futures daily `bookTicker` file. UNVERIFIED mapping.
    pub fn binance_book_ticker(tick_size: f64) -> Self {
        CsvIngestConfig {
            schema: CsvSchema {
                has_header: true,
                timestamp: Some(ColumnRef::named("event_time")),
                timestamp_unit: TimestampUnit::Milliseconds,
                bid_price: ColumnRef::named("best_bid_price"),
                bid_qty: ColumnRef::named("best_bid_qty"),
                ask_price: ColumnRef::named("best_ask_price"),
                ask_qty: ColumnRef::named("best_ask_qty"),
            },
            symbol: SymbolId(1),
            price_divisor: 1.0,
            tick_size,
            resolution: 2,
            qty_decimals: 8,
            on_invalid_row: InvalidRowPolicy::Error,
            drop_unchanged: false,
            max_events: None,
        }
    }

    /// LOBSTER `orderbook` level 1. Timestamps come from the `message` file,
    /// so this config is meant for [`read_lobster_files`]. `tick_size` is in
    /// dollars (`0.01` for US equities). UNVERIFIED mapping.
    pub fn lobster_orderbook(tick_size: f64) -> Self {
        CsvIngestConfig {
            schema: CsvSchema {
                has_header: false,
                timestamp: None,
                timestamp_unit: TimestampUnit::Seconds,
                ask_price: ColumnRef::Index(0),
                ask_qty: ColumnRef::Index(1),
                bid_price: ColumnRef::Index(2),
                bid_qty: ColumnRef::Index(3),
            },
            symbol: SymbolId(1),
            price_divisor: 10_000.0,
            tick_size,
            resolution: 2,
            qty_decimals: 0,
            on_invalid_row: InvalidRowPolicy::Error,
            drop_unchanged: false,
            max_events: None,
        }
    }

    /// A generic named-column schema with a header row.
    pub fn generic(
        timestamp: &str,
        timestamp_unit: TimestampUnit,
        bid_price: &str,
        bid_qty: &str,
        ask_price: &str,
        ask_qty: &str,
        tick_size: f64,
    ) -> Self {
        let mut c = Self::binance_book_ticker(tick_size);
        c.schema = CsvSchema {
            has_header: true,
            timestamp: Some(ColumnRef::named(timestamp)),
            timestamp_unit,
            bid_price: ColumnRef::named(bid_price),
            bid_qty: ColumnRef::named(bid_qty),
            ask_price: ColumnRef::named(ask_price),
            ask_qty: ColumnRef::named(ask_qty),
        };
        c
    }

    fn validate(&self) -> Result<(), DataError> {
        if !(self.tick_size.is_finite() && self.tick_size > 0.0) {
            return Err(schema_err("tick_size must be finite and > 0"));
        }
        if !(self.price_divisor.is_finite() && self.price_divisor > 0.0) {
            return Err(schema_err("price_divisor must be finite and > 0"));
        }
        if self.resolution == 0 {
            return Err(schema_err("resolution must be >= 1"));
        }
        if self.qty_decimals > 12 {
            return Err(schema_err("qty_decimals must be <= 12"));
        }
        Ok(())
    }
}

/// The result of an ingestion run.
#[derive(Debug, Clone, PartialEq)]
pub struct CsvIngest {
    /// Accepted events, in file order (`sequence` = index in this vector).
    pub events: Vec<BookEvent>,
    /// Data rows seen (excluding header and blank lines).
    pub rows_read: usize,
    /// Rows dropped under [`InvalidRowPolicy::Skip`], by reason.
    pub skipped: BTreeMap<SkipReason, usize>,
    /// Rows dropped because `drop_unchanged` was set.
    pub unchanged_dropped: usize,
    /// True if reading stopped early at `max_events`.
    pub truncated_at_max_events: bool,
}

fn schema_err(reason: &str) -> DataError {
    DataError::CsvSchema {
        reason: reason.to_string(),
    }
}

/// Reads a CSV file of L1 quotes.
pub fn read_csv_file(
    path: impl AsRef<Path>,
    config: &CsvIngestConfig,
) -> Result<CsvIngest, DataError> {
    let path = path.as_ref();
    let file = File::open(path)
        .map_err(|e| DataError::CsvIo(format!("cannot open {}: {e}", path.display())))?;
    read_csv_events(BufReader::new(file), config, None)
}

/// Reads a LOBSTER `orderbook` file paired with its `message` file (whose
/// first column supplies the timestamps, seconds after midnight). Row `i` of
/// one corresponds to row `i` of the other; differing row counts are an error.
pub fn read_lobster_files(
    orderbook_path: impl AsRef<Path>,
    message_path: impl AsRef<Path>,
    config: &CsvIngestConfig,
) -> Result<CsvIngest, DataError> {
    let ob = orderbook_path.as_ref();
    let msg = message_path.as_ref();
    let msg_file = File::open(msg)
        .map_err(|e| DataError::CsvIo(format!("cannot open {}: {e}", msg.display())))?;
    let times = read_message_timestamps(BufReader::new(msg_file), config.schema.timestamp_unit)?;
    let ob_file = File::open(ob)
        .map_err(|e| DataError::CsvIo(format!("cannot open {}: {e}", ob.display())))?;
    read_csv_events(BufReader::new(ob_file), config, Some(&times))
}

/// Parses the first column of a LOBSTER `message` stream into nanoseconds.
pub fn read_message_timestamps<R: BufRead>(
    reader: R,
    unit: TimestampUnit,
) -> Result<Vec<u64>, DataError> {
    let mut out = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| DataError::CsvIo(e.to_string()))?;
        if line.trim().is_empty() {
            continue;
        }
        let first = line.split(',').next().unwrap_or("").trim();
        let ts = parse_timestamp(first, unit).ok_or_else(|| DataError::CsvRow {
            row: i + 1,
            reason: format!("message file: unparsable timestamp {first:?}"),
        })?;
        out.push(ts);
    }
    Ok(out)
}

/// Core reader. `external_timestamps`, when given, supplies the timestamp of
/// the i-th data row (required when the schema has no timestamp column).
pub fn read_csv_events<R: BufRead>(
    reader: R,
    config: &CsvIngestConfig,
    external_timestamps: Option<&[u64]>,
) -> Result<CsvIngest, DataError> {
    config.validate()?;
    if config.schema.timestamp.is_none() && external_timestamps.is_none() {
        return Err(schema_err(
            "schema has no timestamp column and no external timestamps were supplied",
        ));
    }

    let mut lines = reader.lines().enumerate();
    let cols = if config.schema.has_header {
        // First non-blank line is the header.
        loop {
            let Some((i, line)) = lines.next() else {
                return Err(schema_err("empty file: expected a header row"));
            };
            let line = line.map_err(|e| DataError::CsvIo(e.to_string()))?;
            if line.trim().is_empty() {
                continue;
            }
            let header: Vec<String> = split_fields(&line).map(str::to_string).collect();
            break Resolved::from_header(&config.schema, &header, i + 1)?;
        }
    } else {
        Resolved::from_positions(&config.schema)?
    };

    let mut out = CsvIngest {
        events: Vec::new(),
        rows_read: 0,
        skipped: BTreeMap::new(),
        unchanged_dropped: 0,
        truncated_at_max_events: false,
    };
    let mut prev_ts: Option<u64> = None;
    let mut prev_book: Option<TopOfBook> = None;
    let mut data_row = 0usize;

    for (i, line) in lines {
        let line = line.map_err(|e| DataError::CsvIo(e.to_string()))?;
        if line.trim().is_empty() {
            continue;
        }
        let line_no = i + 1;
        let this_row = data_row;
        data_row += 1;
        out.rows_read += 1;

        let ts_override = external_timestamps.map(|t| t.get(this_row).copied());
        let parsed = parse_row(&line, &cols, config, ts_override);
        let (ts, book) = match parsed.and_then(|(ts, book)| {
            if let Some(p) = prev_ts {
                if ts < p {
                    return Err((
                        SkipReason::TimestampDecreased,
                        format!("timestamp {ts} ns is earlier than previous {p} ns"),
                    ));
                }
            }
            Ok((ts, book))
        }) {
            Ok(v) => v,
            Err((reason, detail)) => match config.on_invalid_row {
                InvalidRowPolicy::Error => {
                    return Err(DataError::CsvRow {
                        row: line_no,
                        reason: format!("{}: {detail}", reason.as_str()),
                    })
                }
                InvalidRowPolicy::Skip => {
                    *out.skipped.entry(reason).or_insert(0) += 1;
                    continue;
                }
            },
        };

        if config.drop_unchanged && prev_book == Some(book) {
            out.unchanged_dropped += 1;
            prev_ts = Some(ts);
            continue;
        }
        if let Some(max) = config.max_events {
            if out.events.len() >= max {
                out.truncated_at_max_events = true;
                break;
            }
        }
        out.events.push(BookEvent {
            timestamp_ns: ts,
            sequence: out.events.len() as u64,
            symbol: config.symbol,
            book,
        });
        prev_ts = Some(ts);
        prev_book = Some(book);
    }

    if let Some(times) = external_timestamps {
        if times.len() != data_row && !out.truncated_at_max_events {
            return Err(DataError::CsvSchema {
                reason: format!(
                    "orderbook has {data_row} data rows but the message file has {} rows",
                    times.len()
                ),
            });
        }
    }
    Ok(out)
}

struct Resolved {
    timestamp: Option<usize>,
    bid_price: usize,
    bid_qty: usize,
    ask_price: usize,
    ask_qty: usize,
}

impl Resolved {
    fn from_header(schema: &CsvSchema, header: &[String], line: usize) -> Result<Self, DataError> {
        let find = |c: &ColumnRef| -> Result<usize, DataError> {
            match c {
                ColumnRef::Index(i) => Ok(*i),
                ColumnRef::Name(n) => header
                    .iter()
                    .position(|h| h.eq_ignore_ascii_case(n))
                    .ok_or_else(|| DataError::CsvSchema {
                        reason: format!(
                            "column {n:?} not found in header on line {line}: {header:?}"
                        ),
                    }),
            }
        };
        Ok(Resolved {
            timestamp: schema.timestamp.as_ref().map(find).transpose()?,
            bid_price: find(&schema.bid_price)?,
            bid_qty: find(&schema.bid_qty)?,
            ask_price: find(&schema.ask_price)?,
            ask_qty: find(&schema.ask_qty)?,
        })
    }

    fn from_positions(schema: &CsvSchema) -> Result<Self, DataError> {
        let idx = |c: &ColumnRef| -> Result<usize, DataError> {
            match c {
                ColumnRef::Index(i) => Ok(*i),
                ColumnRef::Name(n) => Err(DataError::CsvSchema {
                    reason: format!("column {n:?} is named but the schema has no header row"),
                }),
            }
        };
        Ok(Resolved {
            timestamp: schema.timestamp.as_ref().map(idx).transpose()?,
            bid_price: idx(&schema.bid_price)?,
            bid_qty: idx(&schema.bid_qty)?,
            ask_price: idx(&schema.ask_price)?,
            ask_qty: idx(&schema.ask_qty)?,
        })
    }
}

fn split_fields(line: &str) -> impl Iterator<Item = &str> {
    line.trim_end_matches(['\r', '\n'])
        .split(',')
        .map(|f| f.trim().trim_matches('"'))
}

type RowError = (SkipReason, String);

fn parse_row(
    line: &str,
    cols: &Resolved,
    config: &CsvIngestConfig,
    external_ts: Option<Option<u64>>,
) -> Result<(u64, TopOfBook), RowError> {
    let fields: Vec<&str> = split_fields(line).collect();
    let get = |idx: usize, what: &str| -> Result<&str, RowError> {
        fields.get(idx).copied().ok_or_else(|| {
            (
                SkipReason::Unparsable,
                format!(
                    "missing {what} column (index {idx}); row has {} fields",
                    fields.len()
                ),
            )
        })
    };

    let ts = match (external_ts, cols.timestamp) {
        (Some(Some(t)), _) => t,
        (Some(None), _) => {
            return Err((
                SkipReason::Unparsable,
                "no timestamp available for this row (message file is shorter)".to_string(),
            ))
        }
        (None, Some(c)) => {
            let raw = get(c, "timestamp")?;
            parse_timestamp(raw, config.schema.timestamp_unit)
                .ok_or_else(|| (SkipReason::Unparsable, format!("bad timestamp {raw:?}")))?
        }
        (None, None) => {
            return Err((
                SkipReason::Unparsable,
                "no timestamp source configured".to_string(),
            ))
        }
    };

    let bid_raw = parse_f64(get(cols.bid_price, "bid price")?, "bid price")?;
    let ask_raw = parse_f64(get(cols.ask_price, "ask price")?, "ask price")?;
    let bid_qty_raw = parse_f64(get(cols.bid_qty, "bid size")?, "bid size")?;
    let ask_qty_raw = parse_f64(get(cols.ask_qty, "ask size")?, "ask size")?;

    const SENTINEL: f64 = 9_999_999_999.0;
    if bid_raw <= 0.0 || ask_raw <= 0.0 || bid_raw.abs() >= SENTINEL || ask_raw.abs() >= SENTINEL {
        return Err((
            SkipReason::EmptyLevel,
            format!("bid {bid_raw} / ask {ask_raw}"),
        ));
    }
    if bid_qty_raw < 0.0 || ask_qty_raw < 0.0 {
        return Err((SkipReason::Unparsable, "negative size".to_string()));
    }

    let bid_units = to_units(bid_raw, config)?;
    let ask_units = to_units(ask_raw, config)?;
    let qty_scale = 10f64.powi(config.qty_decimals as i32);
    let bid_qty = to_qty(bid_qty_raw, qty_scale)?;
    let ask_qty = to_qty(ask_qty_raw, qty_scale)?;

    if bid_qty == 0 && ask_qty == 0 {
        return Err((SkipReason::EmptyBook, "both sizes are zero".to_string()));
    }
    let book = TopOfBook::new(
        PriceTicks(bid_units),
        Quantity(bid_qty),
        PriceTicks(ask_units),
        Quantity(ask_qty),
        BookValidationPolicy::RejectCrossedAndLocked,
    )
    .map_err(|e| (SkipReason::CrossedOrLocked, e.to_string()))?;
    Ok((ts, book))
}

fn parse_f64(s: &str, what: &str) -> Result<f64, RowError> {
    match s.parse::<f64>() {
        Ok(v) if v.is_finite() => Ok(v),
        _ => Err((SkipReason::Unparsable, format!("bad {what} {s:?}"))),
    }
}

fn to_units(raw: f64, config: &CsvIngestConfig) -> Result<i64, RowError> {
    let ticks = raw / config.price_divisor / config.tick_size;
    let rounded = ticks.round();
    if (ticks - rounded).abs() > 1e-4 || rounded.abs() > 1e15 {
        return Err((
            SkipReason::OffGridPrice,
            format!(
                "price {raw} is {ticks} ticks at tick_size {} (not an integer)",
                config.tick_size
            ),
        ));
    }
    Ok(rounded as i64 * i64::from(config.resolution))
}

fn to_qty(raw: f64, scale: f64) -> Result<u64, RowError> {
    let v = (raw * scale).round();
    if !(0.0..1.8e19).contains(&v) {
        return Err((SkipReason::Unparsable, format!("size {raw} out of range")));
    }
    Ok(v as u64)
}

fn parse_timestamp(s: &str, unit: TimestampUnit) -> Option<u64> {
    let per = unit.ns_per_unit();
    if let Ok(v) = s.parse::<u64>() {
        return v.checked_mul(per);
    }
    let f: f64 = s.parse().ok()?;
    if !f.is_finite() || f < 0.0 {
        return None;
    }
    let ns = (f * per as f64).round();
    if ns >= 1.8e19 {
        return None;
    }
    Some(ns as u64)
}

#[cfg(test)]
mod tests {
    //! FORMAT TESTS ONLY. Every fixture below is a tiny, hand-written string
    //! shaped like a vendor file, with made-up numbers. None of it is market
    //! data and no result may be derived from it.

    use super::*;
    use std::io::Cursor;

    const BINANCE_FIXTURE: &str = "\
update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,transaction_time,event_time
1001,100.0,2.500,100.1,1.000,1700000000123,1700000000125
1002,100.0,2.000,100.1,1.000,1700000000130,1700000000131
1003,100.1,0.500,100.2,3.000,1700000000200,1700000000201
";

    fn binance() -> CsvIngestConfig {
        CsvIngestConfig::binance_book_ticker(0.1)
    }

    fn read(s: &str, c: &CsvIngestConfig) -> Result<CsvIngest, DataError> {
        read_csv_events(Cursor::new(s.to_string()), c, None)
    }

    #[test]
    fn binance_fixture_parses_into_half_tick_units() {
        let r = read(BINANCE_FIXTURE, &binance()).unwrap();
        assert_eq!(r.rows_read, 3);
        assert_eq!(r.events.len(), 3);
        let e0 = r.events[0];
        // 100.0 / 0.1 = 1000 ticks; x resolution 2 = 2000 units.
        assert_eq!(e0.book.bid_price, PriceTicks(2000));
        assert_eq!(e0.book.ask_price, PriceTicks(2002));
        assert_eq!(e0.book.bid_qty, Quantity(250_000_000));
        assert_eq!(e0.book.ask_qty, Quantity(100_000_000));
        // event_time (ms) is used, converted to ns.
        assert_eq!(e0.timestamp_ns, 1_700_000_000_125_000_000);
        assert_eq!(e0.sequence, 0);
        assert_eq!(r.events[2].sequence, 2);
        // A one-tick spread has an exactly representable mid (2001 units).
        assert_eq!(e0.book.mid_price_ticks(), 2001);
    }

    #[test]
    fn column_order_in_the_header_does_not_matter() {
        let csv = "\
event_time,best_ask_qty,best_ask_price,best_bid_qty,best_bid_price,update_id,transaction_time
1700000000125,1.0,100.1,2.5,100.0,1,1
";
        let r = read(csv, &binance()).unwrap();
        assert_eq!(r.events[0].book.bid_price, PriceTicks(2000));
        assert_eq!(r.events[0].book.ask_qty, Quantity(100_000_000));
    }

    #[test]
    fn missing_column_is_a_schema_error_naming_it() {
        let csv = "update_id,best_bid_price,best_bid_qty,best_ask_price,transaction_time,event_time\n1,1,1,1,1,1\n";
        match read(csv, &binance()) {
            Err(DataError::CsvSchema { reason }) => assert!(reason.contains("best_ask_qty")),
            other => panic!("expected CsvSchema error, got {other:?}"),
        }
    }

    #[test]
    fn crossed_book_aborts_with_the_line_number_by_default() {
        let csv = "\
update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,transaction_time,event_time
1,100.0,1.0,100.1,1.0,1,1000
2,100.2,1.0,100.1,1.0,2,1001
";
        match read(csv, &binance()) {
            Err(DataError::CsvRow { row, reason }) => {
                assert_eq!(row, 3);
                assert!(reason.contains("crossed"));
            }
            other => panic!("expected CsvRow error, got {other:?}"),
        }
    }

    #[test]
    fn skip_policy_drops_and_counts_each_reason() {
        let csv = "\
update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,transaction_time,event_time
1,100.0,1.0,100.1,1.0,1,1000
2,100.2,1.0,100.1,1.0,2,1001
3,100.0,0,100.1,0,3,1002
4,100.05,1.0,100.1,1.0,4,1003
5,abc,1.0,100.1,1.0,5,1004
6,100.0,1.0,100.1,1.0,6,900
7,100.0,2.0,100.1,1.0,7,1005
";
        let mut c = binance();
        c.on_invalid_row = InvalidRowPolicy::Skip;
        let r = read(csv, &c).unwrap();
        assert_eq!(r.rows_read, 7);
        assert_eq!(r.events.len(), 2);
        assert_eq!(r.skipped[&SkipReason::CrossedOrLocked], 1);
        assert_eq!(r.skipped[&SkipReason::EmptyBook], 1);
        assert_eq!(r.skipped[&SkipReason::OffGridPrice], 1);
        assert_eq!(r.skipped[&SkipReason::Unparsable], 1);
        assert_eq!(r.skipped[&SkipReason::TimestampDecreased], 1);
        // Accepted + skipped accounts for every row read.
        let skipped: usize = r.skipped.values().sum();
        assert_eq!(r.events.len() + skipped, r.rows_read);
        // Sequence numbers stay dense and strictly increasing.
        assert_eq!(r.events[1].sequence, 1);
    }

    #[test]
    fn drop_unchanged_removes_repeated_l1_and_counts_it() {
        let csv = "\
update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,transaction_time,event_time
1,100.0,1.0,100.1,1.0,1,1000
2,100.0,1.0,100.1,1.0,2,1001
3,100.0,2.0,100.1,1.0,3,1002
";
        let mut c = binance();
        c.drop_unchanged = true;
        let r = read(csv, &c).unwrap();
        assert_eq!(r.events.len(), 2);
        assert_eq!(r.unchanged_dropped, 1);
    }

    #[test]
    fn max_events_takes_a_chronological_prefix() {
        let mut c = binance();
        c.max_events = Some(2);
        let r = read(BINANCE_FIXTURE, &c).unwrap();
        assert_eq!(r.events.len(), 2);
        assert!(r.truncated_at_max_events);
        assert_eq!(r.events[1].timestamp_ns, 1_700_000_000_131_000_000);
    }

    const LOBSTER_ORDERBOOK: &str = "\
5859400,200,5853300,100,5859800,200,5853000,100
5859400,200,5853300,100,5859800,200,5853000,100
5859400,200,5853300,300,5859800,200,5853000,100
";
    const LOBSTER_MESSAGE: &str = "\
34200.017459617,1,16113575,18,5853300,1
34200.189607785,3,16113575,18,5853300,1
34201.000000000,1,16113600,200,5853300,1
";

    #[test]
    fn lobster_fixture_uses_message_timestamps_and_level_one_columns() {
        // tick = $0.01 -> 100 raw units; prices are $585.94 / $585.33.
        let c = CsvIngestConfig::lobster_orderbook(0.01);
        let times = read_message_timestamps(
            Cursor::new(LOBSTER_MESSAGE.to_string()),
            TimestampUnit::Seconds,
        )
        .unwrap();
        assert_eq!(times[0], 34_200_017_459_617);
        let r =
            read_csv_events(Cursor::new(LOBSTER_ORDERBOOK.to_string()), &c, Some(&times)).unwrap();
        assert_eq!(r.events.len(), 3);
        let b = r.events[0].book;
        // ask 5859400/10000 = 585.94 -> 58594 ticks -> x2; bid 585.33 -> 58533.
        assert_eq!(b.ask_price, PriceTicks(58594 * 2));
        assert_eq!(b.ask_qty, Quantity(200));
        assert_eq!(b.bid_price, PriceTicks(58533 * 2));
        assert_eq!(b.bid_qty, Quantity(100));
        assert_eq!(r.events[0].timestamp_ns, 34_200_017_459_617);
        assert_eq!(r.events[2].timestamp_ns, 34_201_000_000_000);
        assert_eq!(r.events[2].book.bid_qty, Quantity(300));
    }

    #[test]
    fn lobster_sentinel_levels_are_empty_level_rows() {
        let ob = "9999999999,0,-9999999999,0\n5859400,200,5853300,100\n";
        let msg = "34200.1,1,1,1,1,1\n34200.2,1,1,1,1,1\n";
        let mut c = CsvIngestConfig::lobster_orderbook(0.01);
        c.on_invalid_row = InvalidRowPolicy::Skip;
        let times =
            read_message_timestamps(Cursor::new(msg.to_string()), TimestampUnit::Seconds).unwrap();
        let r = read_csv_events(Cursor::new(ob.to_string()), &c, Some(&times)).unwrap();
        assert_eq!(r.events.len(), 1);
        assert_eq!(r.skipped[&SkipReason::EmptyLevel], 1);
        // The surviving row keeps ITS OWN timestamp (row 2), not row 1's.
        assert_eq!(r.events[0].timestamp_ns, 34_200_200_000_000);
    }

    #[test]
    fn lobster_row_count_mismatch_is_an_error() {
        let c = CsvIngestConfig::lobster_orderbook(0.01);
        let times = vec![1u64];
        let r = read_csv_events(Cursor::new(LOBSTER_ORDERBOOK.to_string()), &c, Some(&times));
        assert!(r.is_err());
    }

    #[test]
    fn generic_schema_with_seconds_timestamps() {
        let csv = "ts,bp,bq,ap,aq\n1.5,10.00,5,10.01,7\n";
        let c =
            CsvIngestConfig::generic("ts", TimestampUnit::Seconds, "bp", "bq", "ap", "aq", 0.01);
        let mut c = c;
        c.qty_decimals = 0;
        let r = read(csv, &c).unwrap();
        assert_eq!(r.events[0].timestamp_ns, 1_500_000_000);
        assert_eq!(r.events[0].book.bid_price, PriceTicks(2000));
        assert_eq!(r.events[0].book.ask_qty, Quantity(7));
    }

    #[test]
    fn invalid_config_is_rejected_up_front() {
        let mut c = binance();
        c.tick_size = 0.0;
        assert!(matches!(
            read(BINANCE_FIXTURE, &c),
            Err(DataError::CsvSchema { .. })
        ));
        let mut c = binance();
        c.resolution = 0;
        assert!(read(BINANCE_FIXTURE, &c).is_err());
    }

    #[test]
    fn missing_file_is_a_typed_io_error() {
        let r = read_csv_file("/nonexistent/definitely-not-here.csv", &binance());
        assert!(matches!(r, Err(DataError::CsvIo(_))));
    }
}
