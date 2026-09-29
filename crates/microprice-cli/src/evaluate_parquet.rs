//! `microprice evaluate-parquet`: the same calibration and evaluation as
//! `evaluate-csv`, reading Level-1 quotes from a Parquet file in this
//! project's schema (documented in `docs/parquet-input.md` and on
//! `microprice_data::parquet`).
//!
//! Compiled only with the `parquet` Cargo feature, which forwards to
//! `microprice-data/parquet-ingestion`; without it the subcommand exists but
//! only explains how to enable it, so the default build stays free of the
//! arrow/parquet dependency tree.

#[cfg(feature = "parquet")]
mod enabled {
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    use clap::Args;

    use microprice_core::BookValidationPolicy;
    use microprice_data::csv::CsvIngest;
    use microprice_data::read_events_from_parquet;

    use crate::evaluate_csv::{evaluate_and_report, file_name_of, EvalCommon, SourceInfo};

    #[derive(Args, Debug)]
    pub struct EvaluateParquetArgs {
        /// Parquet file in the schema of docs/parquet-input.md (integer
        /// price units; `--resolution` of them make one exchange tick).
        #[arg(long)]
        input: PathBuf,

        /// Use only the first N rows (a chronological prefix).
        #[arg(long)]
        max_events: Option<usize>,

        /// Optional, display only: currency price of one exchange tick. The
        /// Parquet schema stores integer units, so nothing is validated
        /// against it.
        #[arg(long)]
        tick_size: Option<f64>,

        #[command(flatten)]
        common: EvalCommon,
    }

    pub fn run(args: EvaluateParquetArgs) -> Result<(), Box<dyn std::error::Error>> {
        eprintln!("Reading {} ...", args.input.display());
        // Books are validated with the strict policy; the first bad row
        // aborts the read and is named by index (no skip mode, unlike CSV).
        let mut events =
            read_events_from_parquet(&args.input, BookValidationPolicy::RejectCrossedAndLocked)?;
        let rows_read = events.len();

        // Calibration requires strictly increasing `sequence`, and the
        // no-leakage guard and wall-clock horizons require non-decreasing
        // timestamps; check both here so a bad file fails naming its row.
        for (i, w) in events.windows(2).enumerate() {
            if w[1].sequence <= w[0].sequence {
                return Err(format!(
                    "row {}: sequence {} does not increase past {}",
                    i + 1,
                    w[1].sequence,
                    w[0].sequence
                )
                .into());
            }
            if w[1].timestamp_ns < w[0].timestamp_ns {
                return Err(format!(
                    "row {}: timestamp_ns {} is earlier than the previous row's {}",
                    i + 1,
                    w[1].timestamp_ns,
                    w[0].timestamp_ns
                )
                .into());
            }
        }

        let truncated = match args.max_events {
            Some(n) if n < events.len() => {
                events.truncate(n);
                true
            }
            _ => false,
        };

        let odd_sums = events
            .iter()
            .filter(|e| (e.book.bid_price.0 + e.book.ask_price.0) % 2 != 0)
            .count();
        if odd_sums > 0 && args.common.resolution == 1 {
            eprintln!(
                "warning: {odd_sums} rows have an odd bid+ask sum, so their mid falls between \
                 integer units and the model truncates it; write the file in half-tick units \
                 and use --resolution 2 (docs/parquet-input.md)"
            );
        }

        let ingest = CsvIngest {
            events,
            rows_read,
            skipped: BTreeMap::new(),
            unchanged_dropped: 0,
            truncated_at_max_events: truncated,
        };
        let source = SourceInfo {
            command: "evaluate-parquet",
            file_name: file_name_of(&args.input),
            format: "Parquet (microprice schema)".to_string(),
            tick_size: args.tick_size,
        };
        evaluate_and_report(&args.common, &source, &ingest)
    }
}

#[cfg(feature = "parquet")]
pub use enabled::{run, EvaluateParquetArgs};

#[cfg(not(feature = "parquet"))]
mod disabled {
    use clap::Args;

    /// Accepts and ignores any arguments so the user gets the explanation
    /// below rather than an argument-parsing error.
    #[derive(Args, Debug)]
    #[command(disable_help_flag = true)]
    pub struct EvaluateParquetArgs {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        _rest: Vec<String>,
    }

    pub fn run(_args: EvaluateParquetArgs) -> Result<(), Box<dyn std::error::Error>> {
        Err(
            "this build of microprice does not include Parquet support; rebuild with \
             `cargo build --release -p microprice-cli --features parquet` \
             (see docs/parquet-input.md)"
                .into(),
        )
    }
}

#[cfg(not(feature = "parquet"))]
pub use disabled::{run, EvaluateParquetArgs};
