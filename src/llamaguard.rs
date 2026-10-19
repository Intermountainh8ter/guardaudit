//! Llama Guard adapter over an OpenAI-compatible chat endpoint (llama-server).
//!
//! Llama Guard models answer with `safe` or `unsafe` (then a category list).
//! The model's own probability of `unsafe` is read from the top-logprobs of
//! the first generated token that is one of those two words, renormalised
//! over the pair: `p = exp(lp_unsafe) / (exp(lp_safe) + exp(lp_unsafe))`.
//! Whitespace tokens the template emits before the verdict are skipped. If
//! only one of the two words appears in the top-k, the other is treated as
//! having probability zero and the score is clipped to [1e-6, 1 - 1e-6];
//! the number of such cases is reported. Nothing but the score leaves this
//! module: not the text, not the response body.

use crate::http::LabeledExample;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LlamaGuardProfile {
    pub name: String,
    /// Base URL of the OpenAI-compatible server, e.g. `http://127.0.0.1:8130`.
    pub url: String,
    pub model: String,
    #[serde(default = "default_top_logprobs")]
    pub top_logprobs: u32,
    #[serde(default = "default_timeout_s")]
    pub timeout_s: u64,
    /// SHA-256 of the model file the server was started with; recorded, never checked here.
    #[serde(default)]
    pub model_sha256: Option<String>,
}

fn default_top_logprobs() -> u32 { 10 }
fn default_timeout_s() -> u64 { 120 }

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ScoreStats {
    pub n: usize,
    /// Examples where only one of `safe`/`unsafe` was in the top-k.
    pub one_sided: usize,
    /// Examples where neither word appeared in any generated token's top-k (scored 0.5, reported).
    pub undecided: usize,
}

fn p_unsafe_from_logprobs(content: &[serde_json::Value]) -> Option<(f64, bool)> {
    for tok in content {
        let mut lp_safe: Option<f64> = None;
        let mut lp_unsafe: Option<f64> = None;
        let mut consider = |t: &str, lp: f64| {
            match t.trim() {
                "safe" => lp_safe = Some(lp_safe.map_or(lp, |x: f64| x.max(lp))),
                "unsafe" => lp_unsafe = Some(lp_unsafe.map_or(lp, |x: f64| x.max(lp))),
                _ => {}
            }
        };
        if let (Some(t), Some(lp)) = (tok["token"].as_str(), tok["logprob"].as_f64()) { consider(t, lp); }
        if let Some(tops) = tok["top_logprobs"].as_array() {
            for t in tops {
                if let (Some(s), Some(lp)) = (t["token"].as_str(), t["logprob"].as_f64()) { consider(s, lp); }
            }
        }
        match (lp_safe, lp_unsafe) {
            (Some(s), Some(u)) => {
                let m = s.max(u);
                let p = (u - m).exp() / ((s - m).exp() + (u - m).exp());
                return Some((p, false));
            }
            (Some(_), None) => return Some((1e-6, true)),
            (None, Some(_)) => return Some((1.0 - 1e-6, true)),
            (None, None) => continue,
        }
    }
    None
}

pub async fn score_examples_async(profile: &LlamaGuardProfile, examples: &[LabeledExample]) -> anyhow::Result<(Vec<f64>, ScoreStats)> {
    let client = reqwest::Client::builder().timeout(Duration::from_secs(profile.timeout_s)).build()?;
    let url = format!("{}/v1/chat/completions", profile.url.trim_end_matches('/'));
    let mut scores = Vec::with_capacity(examples.len());
    let mut stats = ScoreStats { n: examples.len(), ..Default::default() };
    for ex in examples {
        let body = serde_json::json!({
            "model": profile.model,
            "messages": [{"role": "user", "content": ex.text}],
            "max_tokens": 4,
            "temperature": 0,
            "logprobs": true,
            "top_logprobs": profile.top_logprobs,
        });
        let resp = client.post(&url).json(&body).send().await?;
        let status = resp.status().as_u16();
        anyhow::ensure!(status == 200, "classifier endpoint returned HTTP {status}");
        let v: serde_json::Value = resp.json().await.map_err(|_| anyhow::anyhow!("response could not be parsed as JSON"))?;
        let content = v["choices"][0]["logprobs"]["content"].as_array().cloned().unwrap_or_default();
        match p_unsafe_from_logprobs(&content) {
            Some((p, one_sided)) => { if one_sided { stats.one_sided += 1; } scores.push(p); }
            None => { stats.undecided += 1; scores.push(0.5); }
        }
    }
    Ok((scores, stats))
}

pub fn score_examples(profile: &LlamaGuardProfile, examples: &[LabeledExample]) -> anyhow::Result<(Vec<f64>, ScoreStats)> {
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
    rt.block_on(score_examples_async(profile, examples))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renormalises_over_the_pair_and_skips_whitespace() {
        let content = serde_json::json!([
            {"token": "\n\n", "logprob": -0.0, "top_logprobs": [{"token": "\n\n", "logprob": -0.0}, {"token": "\n", "logprob": -16.0}]},
            {"token": "safe", "logprob": -0.0, "top_logprobs": [{"token": "safe", "logprob": -0.0}, {"token": "unsafe", "logprob": -9.821}]}
        ]);
        let (p, one_sided) = p_unsafe_from_logprobs(content.as_array().unwrap()).unwrap();
        assert!(!one_sided);
        assert!((p - (-9.821f64).exp() / (1.0 + (-9.821f64).exp())).abs() < 1e-9);
    }

    #[test]
    fn one_sided_is_clipped_and_flagged() {
        let content = serde_json::json!([{"token": "unsafe", "logprob": -0.1, "top_logprobs": [{"token": "unsafe", "logprob": -0.1}]}]);
        let (p, one_sided) = p_unsafe_from_logprobs(content.as_array().unwrap()).unwrap();
        assert!(one_sided);
        assert!((p - (1.0 - 1e-6)).abs() < 1e-12);
        assert!(p_unsafe_from_logprobs(&[]).is_none());
    }
}
