//! Core calibration primitives for safety-classifier auditing.
//!
//! Each classifier emits a score in [0,1] interpreted as "probability
//! this input is unsafe." Ground truth labels are binary. We compute:
//! Brier score, expected calibration error (ECE) with equal-width
//! binning, and per-threshold precision/recall, each with a 95%
//! percentile-bootstrap CI resampling examples. The pre-registered
//! calibration gate is on the UPPER bound of the ECE CI; the
//! discrimination gate is on the LOWER bound of the recall CI.

use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;

/// Brier score: mean squared error between predicted probability and
/// binary label. Lower is better. 0.0 is perfect.
pub fn brier(predictions: &[f64], labels: &[bool]) -> f64 {
    assert_eq!(predictions.len(), labels.len());
    assert!(!predictions.is_empty());
    predictions
        .iter()
        .zip(labels.iter())
        .map(|(p, l)| {
            let y = if *l { 1.0 } else { 0.0 };
            (p - y).powi(2)
        })
        .sum::<f64>()
        / predictions.len() as f64
}

/// Expected calibration error with equal-width binning in [0,1].
/// Standard quantity for calibration plots; smaller is better.
pub fn ece(predictions: &[f64], labels: &[bool], n_bins: usize) -> f64 {
    assert_eq!(predictions.len(), labels.len());
    assert!(!predictions.is_empty());
    assert!(n_bins >= 1);
    let n = predictions.len();
    let mut counts = vec![0usize; n_bins];
    let mut sum_pred = vec![0f64; n_bins];
    let mut sum_label = vec![0f64; n_bins];
    for (p, l) in predictions.iter().zip(labels.iter()) {
        let bin = ((p * n_bins as f64).floor() as usize).min(n_bins - 1);
        counts[bin] += 1;
        sum_pred[bin] += p;
        sum_label[bin] += if *l { 1.0 } else { 0.0 };
    }
    let mut ece = 0f64;
    for b in 0..n_bins {
        if counts[b] == 0 {
            continue;
        }
        let avg_pred = sum_pred[b] / counts[b] as f64;
        let avg_label = sum_label[b] / counts[b] as f64;
        ece += (counts[b] as f64 / n as f64) * (avg_pred - avg_label).abs();
    }
    ece
}

/// Precision at a given threshold: `TP / (TP + FP)`.
pub fn precision_at(predictions: &[f64], labels: &[bool], threshold: f64) -> f64 {
    let mut tp = 0usize;
    let mut fp = 0usize;
    for (p, l) in predictions.iter().zip(labels.iter()) {
        if *p >= threshold {
            if *l {
                tp += 1;
            } else {
                fp += 1;
            }
        }
    }
    if tp + fp == 0 {
        return 0.0;
    }
    tp as f64 / (tp + fp) as f64
}

/// Recall at a given threshold: `TP / (TP + FN)`.
pub fn recall_at(predictions: &[f64], labels: &[bool], threshold: f64) -> f64 {
    let mut tp = 0usize;
    let mut fn_ = 0usize;
    for (p, l) in predictions.iter().zip(labels.iter()) {
        if *l {
            if *p >= threshold {
                tp += 1;
            } else {
                fn_ += 1;
            }
        }
    }
    if tp + fn_ == 0 {
        return 0.0;
    }
    tp as f64 / (tp + fn_) as f64
}

/// Paired percentile bootstrap of a scalar statistic over examples.
///
/// Resamples indices with replacement `iters` times and recomputes
/// `stat` on the resampled (predictions, labels) pair. Returns the
/// 2.5 and 97.5 percentiles.
pub fn bootstrap_ci<F>(
    predictions: &[f64],
    labels: &[bool],
    iters: usize,
    seed: u64,
    stat: F,
) -> (f64, f64)
where
    F: Fn(&[f64], &[bool]) -> f64,
{
    assert_eq!(predictions.len(), labels.len());
    assert!(!predictions.is_empty());
    assert!(iters >= 2);
    let n = predictions.len();
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    let mut rp = Vec::with_capacity(n);
    let mut rl = Vec::with_capacity(n);
    let mut vals = Vec::with_capacity(iters);
    for _ in 0..iters {
        rp.clear();
        rl.clear();
        for _ in 0..n {
            let k = (rand::Rng::gen::<u32>(&mut rng) as usize) % n;
            rp.push(predictions[k]);
            rl.push(labels[k]);
        }
        vals.push(stat(&rp, &rl));
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo = vals[(iters as f64 * 0.025).floor() as usize];
    let hi = vals[((iters as f64 * 0.975).ceil() as usize).min(iters - 1)];
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brier_zero_for_perfect_calibration() {
        let p = vec![1.0, 0.0, 1.0, 0.0];
        let l = vec![true, false, true, false];
        assert!((brier(&p, &l)).abs() < 1e-12);
    }

    #[test]
    fn brier_quarter_for_half_right() {
        let p = vec![0.5, 0.5, 0.5, 0.5];
        let l = vec![true, false, true, false];
        assert!((brier(&p, &l) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn ece_zero_for_perfectly_calibrated_bins() {
        let p = vec![0.1, 0.1, 0.9, 0.9];
        let l = vec![false, false, true, true];
        assert!(ece(&p, &l, 10) < 0.1);
    }

    #[test]
    fn precision_recall_at_half() {
        let p = vec![0.9, 0.6, 0.4, 0.1];
        let l = vec![true, false, true, false];
        // threshold 0.5 flags index 0 (TP) and 1 (FP) -> precision 0.5
        // positives in truth are 0 and 2; flagged positives: {0} -> recall 0.5
        assert!((precision_at(&p, &l, 0.5) - 0.5).abs() < 1e-12);
        assert!((recall_at(&p, &l, 0.5) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn bootstrap_ci_contains_point_estimate_for_brier() {
        let p = vec![0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4];
        let l = vec![true, false, true, false, true, false, true, false];
        let point = brier(&p, &l);
        let (lo, hi) = bootstrap_ci(&p, &l, 2000, 1234, brier);
        assert!(lo <= point && point <= hi, "lo={lo} point={point} hi={hi}");
        assert!(hi - lo < 0.5);
    }

    #[test]
    fn bootstrap_ci_is_zero_width_for_degenerate_perfect_input() {
        let p = vec![1.0, 1.0, 1.0, 1.0];
        let l = vec![true, true, true, true];
        let (lo, hi) = bootstrap_ci(&p, &l, 500, 42, brier);
        assert_eq!(lo, 0.0);
        assert_eq!(hi, 0.0);
    }

    #[test]
    fn bootstrap_ci_is_deterministic_under_seed() {
        let p = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let l = vec![false, false, false, true, true, true, true, true, true];
        let a = bootstrap_ci(&p, &l, 500, 7, brier);
        let b = bootstrap_ci(&p, &l, 500, 7, brier);
        assert_eq!(a, b);
    }

    #[test]
    fn bootstrap_ci_for_ece_at_two_bin_counts() {
        let p = vec![0.1, 0.1, 0.9, 0.9, 0.1, 0.9, 0.1, 0.9];
        let l = vec![false, false, true, true, false, true, false, true];
        let (lo10, hi10) = bootstrap_ci(&p, &l, 500, 1, |pr, la| ece(pr, la, 10));
        let (lo20, hi20) = bootstrap_ci(&p, &l, 500, 1, |pr, la| ece(pr, la, 20));
        assert!(lo10 >= 0.0 && hi10 <= 1.0);
        assert!(lo20 >= 0.0 && hi20 <= 1.0);
    }
}
