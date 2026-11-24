//! Constant-responder floor: F1 of always-flag, never-flag and a seeded
//! random responder on a fixture, against the F1 of audited classifier rows.

use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;

/// F1 of the positive (unsafe) class; 0 when there are no true positives.
pub fn f1(flags: &[bool], labels: &[bool]) -> f64 {
    let (mut tp, mut fp, mut fneg) = (0.0, 0.0, 0.0);
    for (&f, &l) in flags.iter().zip(labels) {
        match (f, l) {
            (true, true) => tp += 1.0,
            (true, false) => fp += 1.0,
            (false, true) => fneg += 1.0,
            _ => {}
        }
    }
    if tp == 0.0 { 0.0 } else { 2.0 * tp / (2.0 * tp + fp + fneg) }
}

pub fn f1_at(scores: &[f64], labels: &[bool], threshold: f64) -> f64 {
    let flags: Vec<bool> = scores.iter().map(|&s| s >= threshold).collect();
    f1(&flags, labels)
}

/// Seeded fair-coin responder.
pub fn random_flags(n: usize, seed: u64) -> Vec<bool> {
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    (0..n).map(|_| rng.gen::<bool>()).collect()
}

/// Percentile bootstrap CI over example indices for a flag-vector F1.
pub fn f1_ci(flags: &[bool], labels: &[bool], iters: usize, seed: u64) -> (f64, f64) {
    let n = flags.len();
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    let mut vals = Vec::with_capacity(iters);
    let (mut rf, mut rl) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for _ in 0..iters {
        rf.clear();
        rl.clear();
        for _ in 0..n {
            let k = (rng.gen::<u32>() as usize) % n;
            rf.push(flags[k]);
            rl.push(labels[k]);
        }
        vals.push(f1(&rf, &rl));
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    (vals[(iters as f64 * 0.025).floor() as usize], vals[((iters as f64 * 0.975).ceil() as usize).min(iters - 1)])
}

/// Floor report for one fixture. `rows` are (classifier name, scores).
pub fn fixture_floor(rows: &[(String, Vec<f64>)], labels: &[bool], threshold: f64, iters: usize, seed: u64) -> serde_json::Value {
    let n = labels.len();
    let always = vec![true; n];
    let never = vec![false; n];
    let rand_f = random_flags(n, seed);
    let consts = [("always_flag", &always), ("never_flag", &never), ("seeded_random", &rand_f)];
    let mut floor = 0.0f64;
    let mut cj = serde_json::Map::new();
    for (name, fl) in consts {
        let v = f1(fl, labels);
        floor = floor.max(v);
        let (lo, hi) = f1_ci(fl, labels, iters, seed);
        cj.insert(name.into(), serde_json::json!({"f1": v, "ci95": [lo, hi]}));
    }
    let mut below = 0usize;
    let mut cl = Vec::new();
    for (name, s) in rows {
        let flags: Vec<bool> = s.iter().map(|&x| x >= threshold).collect();
        let v = f1(&flags, labels);
        let (lo, hi) = f1_ci(&flags, labels, iters, seed);
        let is_below = v < floor;
        if is_below { below += 1; }
        cl.push(serde_json::json!({"classifier": name, "f1": v, "ci95": [lo, hi], "below_floor": is_below}));
    }
    serde_json::json!({
        "n": n,
        "n_unsafe": labels.iter().filter(|&&l| l).count(),
        "constant": cj,
        "best_constant_f1": floor,
        "classifiers": cl,
        "classifiers_below_floor": below,
        "classifiers_audited": rows.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f1_basics() {
        let l = [true, true, false, false];
        assert_eq!(f1(&[true; 4], &l), 2.0 * 2.0 / (4.0 + 2.0));
        assert_eq!(f1(&[false; 4], &l), 0.0);
        assert_eq!(f1(&[true, true, false, false], &l), 1.0);
    }

    #[test]
    fn random_is_seeded() {
        assert_eq!(random_flags(50, 3), random_flags(50, 3));
        assert_ne!(random_flags(50, 3), random_flags(50, 4));
    }

    #[test]
    fn floor_counts_rows_below_always_flag() {
        let l: Vec<bool> = (0..20).map(|i| i < 15).collect();
        let weak: Vec<f64> = (0..20).map(|i| if i < 3 { 0.9 } else { 0.1 }).collect();
        let perfect: Vec<f64> = l.iter().map(|&x| if x { 0.9 } else { 0.1 }).collect();
        let r = fixture_floor(&[("weak".into(), weak), ("perfect".into(), perfect)], &l, 0.5, 200, 1);
        assert_eq!(r["classifiers_below_floor"], 1);
        assert_eq!(r["classifiers_audited"], 2);
    }
}
