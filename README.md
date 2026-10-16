# guardaudit

**Safety-classifier calibration audit.** For Llama Guard, NemoGuard,
Prompt-Guard, Shield-Gemma and the like: measure Brier score, expected
calibration error (ECE), and per-threshold precision/recall under
distributional shift from the classifier's reported training set.

![status: initial scaffold](https://img.shields.io/badge/status-initial%20scaffold-lightgrey)
![language: Rust](https://img.shields.io/badge/language-Rust-blue)
![license: MIT](https://img.shields.io/badge/license-MIT-blue)

## Why

Guard models ship accuracy numbers but no calibration numbers. Downstream
users plug them in as policy gates and trust the yes/no label at the
configured threshold. Calibration failure at that threshold is the
common failure mode and is not published.

## What is in this scaffold

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
