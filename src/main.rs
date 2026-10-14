//! guardaudit CLI — scaffold.

use clap::{Parser, Subcommand};
use guardaudit::{brier, ece, precision_at, recall_at};

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
    /// Score a (predictions, labels) JSON file: compute Brier, ECE, and
    /// precision/recall at the configured threshold.
    Score {
        data: std::path::PathBuf,
        #[arg(long, default_value_t = 10)]
        bins: usize,
        #[arg(long, default_value_t = 0.5)]
        threshold: f64,
    },
    Version,
}

#[derive(serde::Deserialize)]
struct Data {
    predictions: Vec<f64>,
    labels: Vec<bool>,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Cmd::Score {
            data,
            bins,
            threshold,
        } => {
            let d: Data = serde_json::from_str(&std::fs::read_to_string(&data)?)?;
            let b = brier(&d.predictions, &d.labels);
            let e = ece(&d.predictions, &d.labels, bins);
            let p = precision_at(&d.predictions, &d.labels, threshold);
            let r = recall_at(&d.predictions, &d.labels, threshold);
            println!("brier     = {b:.4}");
            println!("ece({bins}) = {e:.4}");
            println!("precision@{threshold:.2} = {p:.4}");
            println!("recall@{threshold:.2}    = {r:.4}");
        }
        Cmd::Version => println!("guardaudit {}", env!("CARGO_PKG_VERSION")),
    }
    Ok(())
}
