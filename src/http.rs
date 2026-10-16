//! Live HTTP classifier adapter.
//!
//! A safety classifier endpoint takes a text input and returns one scalar
//! probability of unsafe. Two shapes are common in deployed services:
//!
//! * **Score endpoint**: `POST {url}` with JSON `{<prompt_field>: text}`
//!   → `{<score_field>: 0.73}`.
//! * **Chat classifier**: `POST {url}/chat/completions` with an
//!   OpenAI-style request → extract the probability from a top-logprob.
//!   Out of scope here; the score endpoint covers Llama-Guard-style
//!   deployments, Perspective, PromptGuard, and the handful of
//!   vendor-hosted safety classifiers.
//!
//! The adapter never prints or logs the input text, the response body,
//! or any auth credential. Inputs to the classifier can themselves be
//! harmful content; the module boundary keeps them from leaking.

use crate::{audit_fixture, certify, CertificationReport, FixtureAudit};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;

/// A score-endpoint classifier profile.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ClassifierProfile {
    pub name: String,
    pub url: String,
    #[serde(default = "default_prompt_field")]
    pub prompt_field: String,
    /// JSON pointer-style path to the scalar probability in the response
    /// (e.g. `"scores/unsafe"` reads `body["scores"]["unsafe"]`).
    pub score_json_path: String,
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default = "default_timeout_s")]
    pub timeout_s: u64,
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
}

fn default_prompt_field() -> String {
    "text".into()
}

fn default_timeout_s() -> u64 {
    30
}

fn default_max_retries() -> u32 {
    2
}

#[derive(Debug, Error)]
pub enum ClassifierError {
    #[error("api_key_env `{0}` is set in profile but the environment variable is not set")]
    AuthMissing(String),
    #[error("endpoint refused authentication (HTTP {status})")]
    AuthRejected { status: u16 },
    #[error("endpoint returned HTTP {status} after {attempts} attempts")]
    HttpStatus { status: u16, attempts: u32 },
    #[error("response is missing field at JSON path `{path}`")]
    MissingField { path: String },
    #[error("response field at `{path}` is not a number")]
    NotANumber { path: String },
    #[error("response field at `{path}` is out of [0,1]: {value}")]
    OutOfRange { path: String, value: f64 },
    #[error("response could not be parsed as JSON")]
    MalformedJson,
    #[error("transport error after {attempts} attempts: {source}")]
    Transport {
        attempts: u32,
        #[source]
        source: reqwest::Error,
    },
}

fn resolve_auth(profile: &ClassifierProfile) -> Result<Option<String>, ClassifierError> {
    match &profile.api_key_env {
        None => Ok(None),
        Some(name) => std::env::var(name)
            .map(Some)
            .map_err(|_| ClassifierError::AuthMissing(name.clone())),
    }
}

fn pointer_from_path(path: &str) -> String {
    let trimmed = path.trim_matches('/');
    if trimmed.is_empty() {
        String::new()
    } else {
        let mut s = String::with_capacity(trimmed.len() + 1);
        for seg in trimmed.split('/') {
            s.push('/');
            s.push_str(seg);
        }
        s
    }
}

/// One labelled text; the label is 1 if the ground truth is "unsafe"
/// and 0 otherwise. `text` is sent to the classifier; it does not leave
/// the adapter in any other form.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LabeledExample {
    pub text: String,
    pub unsafe_label: bool,
}

/// Score a slice of labelled examples. Returns the predicted probability
/// of unsafe for each (in the same order) and never returns the texts.
pub fn score_examples(
    profile: &ClassifierProfile,
    examples: &[LabeledExample],
) -> Result<Vec<f64>, ClassifierError> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("current-thread runtime builds");
    rt.block_on(score_examples_async(profile, examples))
}

pub async fn score_examples_async(
    profile: &ClassifierProfile,
    examples: &[LabeledExample],
) -> Result<Vec<f64>, ClassifierError> {
    let auth = resolve_auth(profile)?;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(profile.timeout_s))
        .build()
        .map_err(|e| ClassifierError::Transport {
            attempts: 0,
            source: e,
        })?;
    let mut scores = Vec::with_capacity(examples.len());
    for ex in examples {
        let body = serde_json::json!({ profile.prompt_field.clone(): ex.text });
        let resp = send_with_retry(&client, &profile.url, auth.as_deref(), &body, profile.max_retries)
            .await?;
        let p = pointer_from_path(&profile.score_json_path);
        let val = resp.pointer(&p).ok_or(ClassifierError::MissingField {
            path: profile.score_json_path.clone(),
        })?;
        let score = val.as_f64().ok_or(ClassifierError::NotANumber {
            path: profile.score_json_path.clone(),
        })?;
        if !(0.0..=1.0).contains(&score) || !score.is_finite() {
            return Err(ClassifierError::OutOfRange {
                path: profile.score_json_path.clone(),
                value: score,
            });
        }
        scores.push(score);
    }
    Ok(scores)
}

/// Convenience: score a fixture, aggregate into a `FixtureAudit`.
pub fn audit_live(
    profile: &ClassifierProfile,
    fixture_label: &str,
    examples: &[LabeledExample],
    threshold: f64,
    boot_iters: usize,
    boot_seed: u64,
) -> Result<FixtureAudit, ClassifierError> {
    let scores = score_examples(profile, examples)?;
    let labels: Vec<bool> = examples.iter().map(|e| e.unsafe_label).collect();
    Ok(audit_fixture(
        fixture_label,
        &scores,
        &labels,
        threshold,
        boot_iters,
        boot_seed,
    ))
}

/// Full certify path: ID fixture + OOD fixture → CertificationReport.
#[allow(clippy::too_many_arguments)]
pub fn certify_live(
    profile: &ClassifierProfile,
    in_distribution: &[LabeledExample],
    out_of_distribution: &[LabeledExample],
    threshold: f64,
    boot_iters: usize,
    boot_seed: u64,
    ece_cutoff: f64,
    recall_floor: f64,
) -> Result<CertificationReport, ClassifierError> {
    let id = audit_live(
        profile,
        "in_distribution",
        in_distribution,
        threshold,
        boot_iters,
        boot_seed,
    )?;
    let ood = audit_live(
        profile,
        "ood",
        out_of_distribution,
        threshold,
        boot_iters,
        boot_seed,
    )?;
    Ok(certify(&profile.name, id, ood, ece_cutoff, recall_floor))
}

async fn send_with_retry(
    client: &reqwest::Client,
    url: &str,
    auth: Option<&str>,
    body: &serde_json::Value,
    max_retries: u32,
) -> Result<serde_json::Value, ClassifierError> {
    let mut attempt: u32 = 0;
    loop {
        attempt += 1;
        let mut req = client.post(url).json(body);
        if let Some(k) = auth {
            req = req.bearer_auth(k);
        }
        match req.send().await {
            Ok(resp) => {
                let status = resp.status();
                if status.is_success() {
                    return resp.json().await.map_err(|_| ClassifierError::MalformedJson);
                }
                if status.as_u16() == 401 || status.as_u16() == 403 {
                    return Err(ClassifierError::AuthRejected {
                        status: status.as_u16(),
                    });
                }
                let retryable = status.as_u16() == 429 || status.is_server_error();
                if !retryable || attempt > max_retries {
                    return Err(ClassifierError::HttpStatus {
                        status: status.as_u16(),
                        attempts: attempt,
                    });
                }
            }
            Err(e) => {
                let retryable = e.is_timeout() || e.is_connect() || e.is_request();
                if !retryable || attempt > max_retries {
                    return Err(ClassifierError::Transport {
                        attempts: attempt,
                        source: e,
                    });
                }
            }
        }
        let delay_ms = 100u64.saturating_mul(1u64 << (attempt - 1).min(10));
        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn profile(base: &str) -> ClassifierProfile {
        ClassifierProfile {
            name: "mock".into(),
            url: format!("{base}/score"),
            prompt_field: "text".into(),
            score_json_path: "scores/unsafe".into(),
            api_key_env: None,
            timeout_s: 2,
            max_retries: 2,
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn scores_a_batch_in_input_order() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/score"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "scores": {"unsafe": 0.9}
            })))
            .mount(&server)
            .await;
        let examples = vec![
            LabeledExample {
                text: "a".into(),
                unsafe_label: true,
            },
            LabeledExample {
                text: "b".into(),
                unsafe_label: false,
            },
        ];
        let scores = score_examples_async(&profile(&server.uri()), &examples)
            .await
            .expect("scores");
        assert_eq!(scores, vec![0.9, 0.9]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn out_of_range_score_is_a_fatal_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/score"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "scores": {"unsafe": 1.5}
            })))
            .mount(&server)
            .await;
        let err = score_examples_async(
            &profile(&server.uri()),
            &[LabeledExample {
                text: "x".into(),
                unsafe_label: true,
            }],
        )
        .await
        .expect_err("out of range");
        assert!(matches!(err, ClassifierError::OutOfRange { .. }));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn non_numeric_score_errors_with_the_path() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/score"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "scores": {"unsafe": "high"}
            })))
            .mount(&server)
            .await;
        let err = score_examples_async(
            &profile(&server.uri()),
            &[LabeledExample {
                text: "x".into(),
                unsafe_label: true,
            }],
        )
        .await
        .expect_err("not a number");
        match err {
            ClassifierError::NotANumber { path } => assert_eq!(path, "scores/unsafe"),
            e => panic!("wrong variant: {e:?}"),
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn retries_a_429() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/score"))
            .respond_with(ResponseTemplate::new(429))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/score"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "scores": {"unsafe": 0.1}
            })))
            .mount(&server)
            .await;
        let scores = score_examples_async(
            &profile(&server.uri()),
            &[LabeledExample {
                text: "x".into(),
                unsafe_label: false,
            }],
        )
        .await
        .expect("retry ok");
        assert_eq!(scores, vec![0.1]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn auth_rejected_is_fatal() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/score"))
            .respond_with(ResponseTemplate::new(401))
            .expect(1)
            .mount(&server)
            .await;
        let err = score_examples_async(
            &profile(&server.uri()),
            &[LabeledExample {
                text: "x".into(),
                unsafe_label: false,
            }],
        )
        .await
        .expect_err("auth refused");
        assert!(matches!(err, ClassifierError::AuthRejected { status: 401 }));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn missing_env_var_errors_before_any_request() {
        std::env::remove_var("GUARDAUDIT_TEST_MISSING_KEY_9K2X");
        let server = MockServer::start().await;
        let mut p = profile(&server.uri());
        p.api_key_env = Some("GUARDAUDIT_TEST_MISSING_KEY_9K2X".into());
        let err = score_examples_async(
            &p,
            &[LabeledExample {
                text: "x".into(),
                unsafe_label: false,
            }],
        )
        .await
        .expect_err("missing env");
        assert!(matches!(err, ClassifierError::AuthMissing(_)));
    }
}
