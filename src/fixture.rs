//! Fixture assembly and the inter-rater check.
//!
//! A fixture is a JSONL file of `{text, unsafe_label, source, id}` built from
//! published prompt lists, kept OUT of the repository (the texts are harmful
//! requests). What the repository keeps is the manifest: one SHA-256 per
//! text with its label and source, so a reader can check that a scored
//! example is the example claimed without the text ever entering git.

use crate::http::LabeledExample;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FixtureItem {
    pub text: String,
    pub unsafe_label: bool,
    pub source: String,
    pub id: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ManifestItem {
    pub sha256: String,
    pub unsafe_label: bool,
    pub source: String,
    pub id: String,
}

/// One source list: a JSONL file whose `field` holds the prompt, every row carrying `label`.
pub struct Source {
    pub path: PathBuf,
    pub field: &'static str,
    pub label: bool,
    pub name: &'static str,
}

pub fn sha256_hex(s: &str) -> String {
    use std::fmt::Write;
    let digest = sha2_digest(s.as_bytes());
    let mut out = String::with_capacity(64);
    for b in digest { let _ = write!(out, "{b:02x}"); }
    out
}

fn sha2_digest(bytes: &[u8]) -> [u8; 32] {
    // Minimal SHA-256 (FIPS 180-4) so the crate adds no dependency for one hash.
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    let mut h: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let mut msg = bytes.to_vec();
    let bit_len = (bytes.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 { msg.push(0); }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in msg.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 { w[i] = u32::from_be_bytes([chunk[4 * i], chunk[4 * i + 1], chunk[4 * i + 2], chunk[4 * i + 3]]); }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
        }
        for (i, v) in [a, b, c, d, e, f, g, hh].iter().enumerate() { h[i] = h[i].wrapping_add(*v); }
    }
    let mut out = [0u8; 32];
    for (i, v) in h.iter().enumerate() { out[4 * i..4 * i + 4].copy_from_slice(&v.to_be_bytes()); }
    out
}

pub fn build(sources: &[Source]) -> anyhow::Result<Vec<FixtureItem>> {
    let mut items = Vec::new();
    for s in sources {
        let text = std::fs::read_to_string(&s.path).map_err(|e| anyhow::anyhow!("{}: {e}", s.path.display()))?;
        for (i, line) in text.lines().enumerate() {
            if line.trim().is_empty() { continue; }
            let v: serde_json::Value = serde_json::from_str(line)?;
            let prompt = v[s.field].as_str().ok_or_else(|| anyhow::anyhow!("{} line {}: no `{}`", s.path.display(), i + 1, s.field))?;
            let id = v["id"].as_str().map(|x| x.to_string()).unwrap_or_else(|| format!("{}-{}", s.name, i + 1));
            items.push(FixtureItem { text: prompt.to_string(), unsafe_label: s.label, source: s.name.to_string(), id });
        }
    }
    Ok(items)
}

pub fn write_jsonl(path: &Path, items: &[FixtureItem]) -> anyhow::Result<()> {
    let mut out = String::new();
    for it in items { out.push_str(&serde_json::to_string(it)?); out.push('\n'); }
    std::fs::write(path, out)?;
    Ok(())
}

pub fn read_jsonl(path: &Path) -> anyhow::Result<Vec<FixtureItem>> {
    let text = std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    text.lines().filter(|l| !l.trim().is_empty()).map(|l| Ok(serde_json::from_str(l)?)).collect()
}

pub fn manifest(items: &[FixtureItem]) -> Vec<ManifestItem> {
    items.iter().map(|it| ManifestItem { sha256: sha256_hex(&it.text), unsafe_label: it.unsafe_label, source: it.source.clone(), id: it.id.clone() }).collect()
}

pub fn as_labeled(items: &[FixtureItem]) -> Vec<LabeledExample> {
    items.iter().map(|it| LabeledExample { text: it.text.clone(), unsafe_label: it.unsafe_label }).collect()
}

/// Deterministic sample of `k` indices from `n` with a splitmix-seeded Fisher-Yates partial shuffle.
pub fn sample_indices(n: usize, k: usize, seed: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut s = seed;
    let mut next = move || { s = s.wrapping_add(0x9E3779B97F4A7C15); let mut z = s; z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9); z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB); z ^ (z >> 31) };
    for i in 0..k.min(n) {
        let j = i + (next() as usize) % (n - i);
        idx.swap(i, j);
    }
    idx.truncate(k.min(n));
    idx
}

/// Cohen's kappa between two binary ratings.
pub fn cohen_kappa(a: &[bool], b: &[bool]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    if n == 0.0 { return f64::NAN; }
    let po = a.iter().zip(b).filter(|(x, y)| x == y).count() as f64 / n;
    let pa = a.iter().filter(|x| **x).count() as f64 / n;
    let pb = b.iter().filter(|x| **x).count() as f64 / n;
    let pe = pa * pb + (1.0 - pa) * (1.0 - pb);
    if (1.0 - pe).abs() < 1e-12 { return 1.0; }
    (po - pe) / (1.0 - pe)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_known_vectors() {
        assert_eq!(sha256_hex(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(sha256_hex("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        let long = "a".repeat(1000);
        assert_eq!(sha256_hex(&long), "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3");
    }

    #[test]
    fn kappa_is_one_on_agreement_and_zero_on_chance() {
        let a = vec![true, false, true, false];
        assert!((cohen_kappa(&a, &a) - 1.0).abs() < 1e-12);
        let b = vec![true, true, false, false];
        assert!(cohen_kappa(&a, &b).abs() < 1e-12);
    }

    #[test]
    fn sampling_is_deterministic_and_without_replacement() {
        let s1 = sample_indices(100, 40, 42);
        let s2 = sample_indices(100, 40, 42);
        assert_eq!(s1, s2);
        let mut u = s1.clone(); u.sort(); u.dedup();
        assert_eq!(u.len(), 40);
    }
}
