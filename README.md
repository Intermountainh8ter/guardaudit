# guardaudit

**Safety-classifier calibration audit.** For Llama Guard, NemoGuard,
Prompt-Guard, Shield-Gemma and the like: measure Brier score, expected
calibration error (ECE), and per-threshold precision/recall under
distributional shift from the classifier's reported training set.

![status: first live audit](https://img.shields.io/badge/status-first%20live%20audit-blue)
![language: Rust](https://img.shields.io/badge/language-Rust-blue)
![license: MIT](https://img.shields.io/badge/license-MIT-blue)

## Results

<!-- RESULTS:START -->
Threshold 0.5 (argmax between `safe` and `unsafe`), 2,000 bootstrap resamples, seed 1234. Gates read the OOD interval bounds: ECE(10) upper bound ≤ 0.05 and recall lower bound ≥ 0.85.

| classifier | fixture | n | Brier | ECE(10) | ECE(20) | precision@t | recall@t | one-sided / undecided | gate |
|---|---|---|---|---|---|---|---|---|---|
| Llama-Guard-3-8B Q8_0 (llama.cpp, one slot, temperature 0) | in-distribution | 200 | 0.008 [0.000, 0.020] | 0.013 [0.004, 0.027] | 0.014 [0.004, 0.028] | 1.000 [1.000, 1.000] | 0.980 [0.949, 1.000] | 0 / 0 | reported, not gated |
| Llama-Guard-3-8B Q8_0 (llama.cpp, one slot, temperature 0) | OOD | 140 | 0.026 [0.008, 0.046] | 0.040 [0.018, 0.065] | 0.040 [0.018, 0.066] | 1.000 [1.000, 1.000] | 0.940 [0.894, 0.980] | 0 / 0 | calibration FAIL; discrimination pass |

**Llama-Guard-3-8B Q8_0 (llama.cpp, one slot, temperature 0)**: not certified

Inter-rater check (source label vs one blind re-rating of 40 items per fixture): id kappa 0.851 (37 of 40 agreements); ood kappa 1.000 (40 of 40 agreements).

<!-- RESULTS:END -->

Rendered by `guardaudit report` from `results/*.json`. The fixtures' texts are not in the repository; `results/fixtures.manifest.json` holds a SHA-256 per item with its label and source, and each result file carries every example's hash and score. [docs/PREREGISTRATION.md](docs/PREREGISTRATION.md) amendment 1 fixed the classifier, the score rule, the fixtures, the threshold and the kappa procedure before the first score was computed.

## Live audit, step by step

```
guardaudit fixture --behaviours-dir <lists> --suites-dir <refusalscope/suites> --out <outside the repo>   # id.jsonl, ood.jsonl + results/fixtures.manifest.json
guardaudit sample  --fixture <out>/id.jsonl --k 40 --seed 42 --template-out ratings-id.template.jsonl   # blind sample to rate by hand
guardaudit kappa   --fixture <out>/id.jsonl --ratings ratings-id.jsonl --label id --out results/kappa-id.json
guardaudit certify-live --profile examples/llama-guard-3-8b.json --id <out>/id.jsonl --ood <out>/ood.jsonl --out results/llama-guard-3-8b-q8.json
guardaudit report
```

The Llama Guard profile points at an OpenAI-compatible `llama-server` (`--jinja`, one slot). The score is the model's own probability of `unsafe` from the first generated token's top-logprobs, renormalised over `safe`/`unsafe`; one-sided and undecided cases are counted and reported.

## Why

Guard models ship accuracy numbers but no calibration numbers. Downstream
users plug them in as policy gates and trust the yes/no label at the
configured threshold. Calibration failure at that threshold is the
common failure mode and is not published.

## What is in this repository

- `brier(predictions, labels)`
- `ece(predictions, labels, n_bins)` — equal-width binning
- `precision_at(predictions, labels, threshold)`
- `recall_at(predictions, labels, threshold)`
- `bootstrap_ci(predictions, labels, iters, seed, stat)` — 95% percentile
  bootstrap over examples; the pre-registered gates read the CI bounds,
  not the points
- `guardaudit score` CLI: prints Brier, ECE at 10 and 20 bins, and
  precision/recall at `--threshold`, each with its 95% bootstrap CI
- `docs/PREREGISTRATION.md` locking the gate: calibrated iff the UPPER
  bound of the ECE CI is ≤ 0.05 on the OOD fixture, and the LOWER bound
  of recall at the recommended threshold is ≥ 0.85.

Not in this commit (openly tracked):

- Classifier adapters: Llama Guard 3, NemoGuard, Prompt-Guard,
  Shield-Gemma.
- Fixtures: held-out safety-labeled corpus (JailbreakBench behaviors
  with human labels) + an OOD corpus from a different distribution.

## License

MIT. See `LICENSE`.
