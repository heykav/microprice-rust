//! End-to-end tests of `microprice evaluate-csv`.
//!
//! FORMAT/PLUMBING TESTS ONLY. The input files are written on the fly from
//! the deterministic *synthetic* generator, laid out in the Binance
//! `bookTicker` and LOBSTER column formats. They prove the file -> ingest ->
//! calibrate -> evaluate -> report path runs and fails cleanly; the numbers
//! in the reports they produce mean nothing about real markets.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::{Command, Output};

use microprice_core::SymbolId;
use microprice_data::{MarketDataSource, SyntheticConfig, SyntheticEventGenerator};

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("microprice-cli-test-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn synthetic_events(n: usize) -> Vec<microprice_core::BookEvent> {
    let mut g = SyntheticEventGenerator::new(SyntheticConfig {
        symbol: SymbolId(1),
        initial_mid_ticks: 10_000,
        initial_spread_ticks: 2,
        initial_bid_qty: 500,
        initial_ask_qty: 500,
        arrival_rate: 0.3,
        cancel_rate: 0.2,
        market_order_rate: 0.2,
        imbalance_persistence: 0.5,
        price_move_probability: 0.1,
        seed: 1,
    })
    .unwrap();
    (0..n).map(|_| g.next_event().unwrap()).collect()
}

/// Binance layout, tick 0.1 (price = ticks / 10).
fn binance_csv(n: usize) -> String {
    let mut s = String::from(
        "update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,transaction_time,event_time\n",
    );
    for (i, e) in synthetic_events(n).iter().enumerate() {
        let ms = 1_700_000_000_000u64 + i as u64 * 10;
        let _ = writeln!(
            s,
            "{},{:.1},{}.000,{:.1},{}.000,{},{}",
            i,
            e.book.bid_price.0 as f64 / 10.0,
            e.book.bid_qty.0,
            e.book.ask_price.0 as f64 / 10.0,
            e.book.ask_qty.0,
            ms,
            ms
        );
    }
    s
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_microprice"))
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn binance_layout_end_to_end_writes_a_report_that_states_its_sample_sizes() {
    let dir = scratch("binance");
    let csv = dir.join("SYNTH-bookTicker.csv");
    std::fs::write(&csv, binance_csv(6000)).unwrap();
    let report = dir.join("out").join("report.md");
    let out = run(&[
        "evaluate-csv",
        "--input",
        csv.to_str().unwrap(),
        "--format",
        "binance-bookticker",
        "--tick-size",
        "0.1",
        "--horizons",
        "1,5",
        "--primary-horizon",
        "5",
        "--bootstrap-resamples",
        "50",
        "--block-len",
        "50",
        "--report-md",
        report.to_str().unwrap(),
    ]);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = std::fs::read_to_string(&report).unwrap();
    assert!(text.contains("# Real-data evaluation report"));
    assert!(text.contains("events accepted: 6000"));
    assert!(text.contains("4200 train events, 1800 test events"));
    assert!(text.contains("### Horizon 1 events"));
    assert!(text.contains("### Horizon 5 events"));
    assert!(text.contains("Pre-registered decision (primary horizon 5)"));
    assert!(text.contains("naive mid"));
    assert!(text.contains("weighted mid"));
    // The report echoes stdout.
    assert!(String::from_utf8_lossy(&out.stdout).contains("Pre-registered decision"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn lobster_layout_end_to_end() {
    let dir = scratch("lobster");
    let events = synthetic_events(3000);
    let mut ob = String::new();
    let mut msg = String::new();
    for (i, e) in events.iter().enumerate() {
        // Prices in dollars x 10,000 with a $0.01 tick: ticks * 100.
        let _ = writeln!(
            ob,
            "{},{},{},{}",
            e.book.ask_price.0 * 100,
            e.book.ask_qty.0,
            e.book.bid_price.0 * 100,
            e.book.bid_qty.0
        );
        let _ = writeln!(
            msg,
            "{}.{:09},1,{},1,{},1",
            34200 + i / 100,
            (i % 100) * 10_000_000,
            i,
            1
        );
    }
    let ob_path = dir.join("SYNTH_orderbook_1.csv");
    let msg_path = dir.join("SYNTH_message_1.csv");
    std::fs::write(&ob_path, ob).unwrap();
    std::fs::write(&msg_path, msg).unwrap();
    let out = run(&[
        "evaluate-csv",
        "--input",
        ob_path.to_str().unwrap(),
        "--format",
        "lobster",
        "--lobster-messages",
        msg_path.to_str().unwrap(),
        "--tick-size",
        "0.01",
        "--horizons",
        "1",
        "--primary-horizon",
        "1",
        "--bootstrap-resamples",
        "20",
        "--block-len",
        "20",
    ]);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("events accepted: 3000"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn lobster_without_messages_fails_with_a_clear_error() {
    let dir = scratch("lobster-nomsg");
    let ob = dir.join("ob.csv");
    std::fs::write(&ob, "1,1,1,1\n").unwrap();
    let out = run(&[
        "evaluate-csv",
        "--input",
        ob.to_str().unwrap(),
        "--format",
        "lobster",
        "--tick-size",
        "0.01",
    ]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("--lobster-messages"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn tick_size_is_required_not_guessed() {
    let out = run(&["evaluate-csv", "--input", "/nonexistent.csv"]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("tick-size"));
}

#[test]
fn missing_file_and_bad_rows_fail_without_panicking() {
    let out = run(&[
        "evaluate-csv",
        "--input",
        "/nonexistent/x.csv",
        "--tick-size",
        "0.1",
    ]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("error:"));

    let dir = scratch("badrows");
    let csv = dir.join("bad.csv");
    let mut body = binance_csv(200);
    body.push_str("999,100.0,1.0,99.9,1.0,1,1700000009999\n"); // crossed
    std::fs::write(&csv, body).unwrap();
    let base = [
        "evaluate-csv",
        "--input",
        csv.to_str().unwrap(),
        "--format",
        "binance-bookticker",
        "--tick-size",
        "0.1",
        "--horizons",
        "1",
        "--primary-horizon",
        "1",
        "--bootstrap-resamples",
        "0",
    ];
    // Default: abort, naming the line.
    let out = run(&base);
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("line 202") && err.contains("crossed"), "{err}");
    // With --skip-invalid-rows: proceeds and reports the skip.
    let mut with_skip = base.to_vec();
    with_skip.push("--skip-invalid-rows");
    let out = run(&with_skip);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        String::from_utf8_lossy(&out.stdout).contains("rows skipped (crossed or locked book): 1")
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn primary_horizon_must_be_among_the_horizons() {
    let out = run(&[
        "evaluate-csv",
        "--input",
        "/nonexistent.csv",
        "--tick-size",
        "0.1",
        "--horizons",
        "1,2",
        "--primary-horizon",
        "10",
    ]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("primary-horizon"));
}
