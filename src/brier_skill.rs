//! Brier skill against the constant-prevalence predictor, which ignores the input.

use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;

/// Brier score of predicting the labels' own prevalence for every example: p(1-p).
pub fn prevalence_brier(labels: &[bool]) -> f64 {
    let p = labels.iter().filter(|&&l| l).count() as f64 / labels.len() as f64;
    p * (1.0 - p)
}

/// Percentile bootstrap of (classifier Brier - constant Brier) and of the classifier Brier,
/// both on the same resamples. Returns (diff ci, brier ci).
fn paired_ci(scores: &[f64], labels: &[bool], iters: usize, seed: u64) -> ((f64, f64), (f64, f64)) {
    let n = labels.len();
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    let (mut diffs, mut bs) = (Vec::with_capacity(iters), Vec::with_capacity(iters));
    let (mut rp, mut rl) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for _ in 0..iters {
        rp.clear();
        rl.clear();
        for _ in 0..n {
            let k = (rng.gen::<u32>() as usize) % n;
            rp.push(scores[k]);
            rl.push(labels[k]);
        }
        let b = crate::brier(&rp, &rl);
        bs.push(b);
        diffs.push(b - prevalence_brier(&rl));
    }
    let q = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        (v[(iters as f64 * 0.025).floor() as usize], v[((iters as f64 * 0.975).ceil() as usize).min(iters - 1)])
    };
    (q(&mut diffs), q(&mut bs))
}

/// Report for one fixture. `rows` are (classifier name, scores).
pub fn fixture_skill(rows: &[(String, Vec<f64>)], labels: &[bool], iters: usize, seed: u64) -> serde_json::Value {
    let n = labels.len();
    let base = prevalence_brier(labels);
    let mut worse = 0usize;
    let mut cl = Vec::new();
    for (name, s) in rows {
        let b = crate::brier(s, labels);
        let (d_ci, b_ci) = paired_ci(s, labels, iters, seed);
        let is_worse = b > base;
        if is_worse { worse += 1; }
        cl.push(serde_json::json!({
            "classifier": name, "brier": b, "brier_ci95": [b_ci.0, b_ci.1],
            "diff_vs_prevalence": b - base, "diff_ci95": [d_ci.0, d_ci.1],
            "brier_skill": 1.0 - b / base, "worse_than_prevalence": is_worse,
        }));
    }
    let n_unsafe = labels.iter().filter(|&&l| l).count();
    serde_json::json!({
        "n": n, "n_unsafe": n_unsafe,
        "prevalence": n_unsafe as f64 / n as f64,
        "prevalence_brier": base,
        "classifiers": cl,
        "classifiers_worse_than_prevalence": worse,
        "classifiers_audited": rows.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prevalence_brier_is_p_times_one_minus_p() {
        let l = [true, true, true, false];
        assert!((prevalence_brier(&l) - 0.1875).abs() < 1e-12);
        assert_eq!(prevalence_brier(&[true; 5]), 0.0);
    }

    #[test]
    fn constant_prevalence_scores_equal_baseline() {
        let l: Vec<bool> = (0..20).map(|i| i < 15).collect();
        let s = vec![0.75; 20];
        assert!((crate::brier(&s, &l) - prevalence_brier(&l)).abs() < 1e-12);
    }

    #[test]
    fn counts_rows_worse_than_prevalence() {
        let l: Vec<bool> = (0..20).map(|i| i < 15).collect();
        let good: Vec<f64> = l.iter().map(|&x| if x { 0.9 } else { 0.1 }).collect();
        let bad: Vec<f64> = l.iter().map(|&x| if x { 0.1 } else { 0.9 }).collect();
        let r = fixture_skill(&[("good".into(), good), ("bad".into(), bad)], &l, 200, 1);
        assert_eq!(r["classifiers_worse_than_prevalence"], 1);
        assert_eq!(r["classifiers_audited"], 2);
        assert_eq!(r["classifiers"][0]["worse_than_prevalence"], false);
    }

    #[test]
    fn seeded_ci_is_deterministic() {
        let l: Vec<bool> = (0..30).map(|i| i % 3 != 0).collect();
        let s: Vec<f64> = (0..30).map(|i| (i % 7) as f64 / 7.0).collect();
        assert_eq!(paired_ci(&s, &l, 100, 5), paired_ci(&s, &l, 100, 5));
    }
}
