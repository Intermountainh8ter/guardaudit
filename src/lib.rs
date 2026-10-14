//! Core calibration primitives for safety-classifier auditing.
//!
//! Each classifier emits a score in [0,1] interpreted as "probability
//! this input is unsafe." Ground truth labels are binary. We compute:
//! Brier score, expected calibration error (ECE) with equal-width
//! binning, and per-threshold precision/recall.

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
}
