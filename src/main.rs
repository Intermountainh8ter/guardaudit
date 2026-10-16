//! guardaudit CLI — scaffold.
//!
//! `score` reads a (predictions, labels) JSON and prints Brier, ECE at
//! 10 and 20 bins, and precision/recall at the configured threshold,
//! each with a 95% percentile-bootstrap CI resampling examples with
//! replacement. The pre-registered gate is on CI bounds, not points,
//! so the CLI prints the point AND the interval on every row.

use clap::{Parser, Subcommand};
use guardaudit::{bootstrap_ci, brier, ece, precision_at, recall_at};

#[derive(Parser)]
#[command(
    name = "guardaudit",
    version,
    about = "Safety-classifier calibration audit"
)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Score a (predictions, labels) JSON file: compute Brier, ECE at 10
    /// and 20 bins, and precision/recall at the configured threshold,
    /// each with a 95% bootstrap CI over examples.
    Score {
        data: std::path::PathBuf,
        #[arg(long, default_value_t = 0.5)]
        threshold: f64,
        #[arg(long, default_value_t = 2000)]
        bootstrap: usize,
        #[arg(long, default_value_t = 1234)]
        seed: u64,
    },
    Version,
}

#[derive(serde::Deserialize)]
struct Data {
    predictions: Vec<f64>,
    labels: Vec<bool>,
}

fn row(label: &str, point: f64, lo: f64, hi: f64) {
    println!("{label:14} = {point:.4}  [{lo:.4}, {hi:.4}]");
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Cmd::Score {
            data,
            threshold,
            bootstrap,
            seed,
        } => {
            let d: Data = serde_json::from_str(&std::fs::read_to_string(&data)?)?;
            let p = &d.predictions;
            let l = &d.labels;
            let (lo, hi) = bootstrap_ci(p, l, bootstrap, seed, brier);
            row("brier", brier(p, l), lo, hi);
            let (lo, hi) = bootstrap_ci(p, l, bootstrap, seed, |a, b| ece(a, b, 10));
            row("ece(10)", ece(p, l, 10), lo, hi);
            let (lo, hi) = bootstrap_ci(p, l, bootstrap, seed, |a, b| ece(a, b, 20));
            row("ece(20)", ece(p, l, 20), lo, hi);
            let (lo, hi) =
                bootstrap_ci(p, l, bootstrap, seed, |a, b| precision_at(a, b, threshold));
            row(
                &format!("precision@{threshold:.2}"),
                precision_at(p, l, threshold),
                lo,
                hi,
            );
            let (lo, hi) = bootstrap_ci(p, l, bootstrap, seed, |a, b| recall_at(a, b, threshold));
            row(
                &format!("recall@{threshold:.2}"),
                recall_at(p, l, threshold),
                lo,
                hi,
            );
        }
        Cmd::Version => println!("guardaudit {}", env!("CARGO_PKG_VERSION")),
    }
    Ok(())
}
