//! End-to-end checks of the `--symmetrize` option and of the martingale /
//! antisymmetry diagnostics in CLI output. Synthetic data only; the numbers
//! printed here mean nothing about real markets.

use std::process::{Command, Output};

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_microprice"))
        .args(args)
        .output()
        .unwrap()
}

fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("mp-cli-sym-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Extracts `x` from a line containing `... = x <unit>` / `residual: x`.
fn residual_after(text: &str, key: &str) -> f64 {
    let line = text
        .lines()
        .find(|l| l.contains(key))
        .unwrap_or_else(|| panic!("no line with {key:?} in:\n{text}"));
    let rest = &line[line.find(key).unwrap() + key.len()..];
    let tok = rest
        .split_whitespace()
        .find(|t| t.parse::<f64>().is_ok())
        .unwrap_or_else(|| panic!("no number after {key:?} in {line:?}"));
    tok.parse().unwrap()
}

#[test]
fn train_symmetrize_flag_yields_an_antisymmetric_model_and_default_does_not() {
    let dir = tmp("train");
    let mut residuals = Vec::new();
    for sym in [false, true] {
        let model = dir.join(if sym { "sym.bin" } else { "plain.bin" });
        let mut args = vec![
            "train",
            "--output",
            model.to_str().unwrap(),
            "--num-events",
            "60000",
            "--num-imbalance-buckets",
            "6",
            "--spread-bucket-bounds",
            "1,2,4",
        ];
        if sym {
            args.push("--symmetrize");
        }
        let out = run(&args);
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
        assert!(stdout.contains("Martingale diagnostic:"));
        assert!(stdout.contains(if sym {
            "Symmetrized calibration: yes"
        } else {
            "Symmetrized calibration: no"
        }));

        let insp = run(&["inspect", "--model", model.to_str().unwrap()]);
        assert!(insp.status.success());
        let text = String::from_utf8_lossy(&insp.stdout).into_owned();
        residuals.push(residual_after(&text, "antisymmetry residual:"));
    }
    let (plain, sym) = (residuals[0], residuals[1]);
    assert!(sym < 1e-9, "symmetrized residual {sym}");
    assert!(plain > 1e-6, "default should not be antisymmetric: {plain}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn evaluate_csv_report_labels_symmetrized_runs_as_exploratory() {
    let dir = tmp("csv");
    // Minimal generic CSV: a two-level oscillation with varying sizes.
    let mut csv = String::from("timestamp,bid_price,bid_qty,ask_price,ask_qty\n");
    for i in 0..4000u64 {
        let up = (i / 7) % 2 == 0;
        let bid = if up { 100.0 } else { 100.1 };
        let (bq, aq) = (1 + (i * 37) % 90, 1 + (i * 53) % 90);
        csv.push_str(&format!(
            "{},{:.1},{},{:.1},{}\n",
            1_700_000_000_000 + i * 5,
            bid,
            bq,
            bid + 0.1,
            aq
        ));
    }
    let path = dir.join("q.csv");
    std::fs::write(&path, csv).unwrap();
    let base = |extra: &[&'static str]| {
        let mut a = vec![
            "evaluate-csv",
            "--input",
            path.to_str().unwrap(),
            "--tick-size",
            "0.1",
            "--horizons",
            "1",
            "--primary-horizon",
            "1",
            "--bootstrap-resamples",
            "0",
        ];
        a.extend_from_slice(extra);
        run(&a)
    };
    let plain = String::from_utf8_lossy(&base(&[]).stdout).into_owned();
    assert!(plain.contains("pre-registered configuration (no symmetrization)"));
    assert!(plain.contains("martingale diagnostic"));
    assert!(!plain.contains("EXPLORATORY"));

    let out = base(&["--symmetrize"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let sym = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(sym.contains("imbalance-symmetrized (EXPLORATORY"));
    assert!(sym.contains("must not be reported as the pre-registered result"));
    assert!(residual_after(&sym, "antisymmetry residual max|G*[s] + G*[mirror(s)]|:") < 1e-9);
    let _ = std::fs::remove_dir_all(&dir);
}
