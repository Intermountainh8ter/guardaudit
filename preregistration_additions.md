# Pre-registration additions

## 2026-11-24T10:13:38Z constant-floor proposal:34c3f86edc99774b3f53e37534f252456df1326532b9ba3c200a06e60f6a21a7

Hypothesis: on the OOD fixture (n=140) the best constant responder (always-flag) has an F1 floor that both ShieldGemma 2B rows (all-in-one and per-policy max) fall below at threshold 0.5, while the Llama Guard 3 8B row stays above it, so exactly 2 of the 3 audited OOD rows are below the floor. F1 is computed from the committed per-example labels and scores; the floor is the maximum F1 over always-flag, never-flag and a seeded random responder (seed 584).

Metric: `/floor/ood/classifiers_below_floor`. Expected: equals 2.

Falsifier: any count other than 2, i.e. a ShieldGemma row at or above the always-flag F1 on OOD, or the Llama Guard row below it.

## 2026-11-24T12:26:51Z brier-skill proposal:4f263fb1dc8e34fcee07659968d2737e43a684996c5cb56d18d8d9a9d59f1dcc

Hypothesis: on the OOD fixture (n=140) both ShieldGemma 2B rows (all-in-one and per-policy max) have a point-estimate Brier score above the constant-prevalence predictor's Brier score p(1-p), where p is the unsafe prevalence computed from the committed labels, while the Llama Guard 3 8B row is below it, so exactly 2 of the 3 audited OOD rows are worse than a predictor that ignores the input. On the in-distribution fixture (n=200) 0 of 3 are. Bootstrap intervals (2,000 resamples, seed 586) are reported for each row's Brier score and for the paired difference to the constant predictor.

Metric: `/brier_skill/ood/classifiers_worse_than_prevalence`. Expected: equals 2.

Falsifier: any count other than 2 on OOD, i.e. a ShieldGemma row with Brier at or below the prevalence-constant Brier, or the Llama Guard row above it.
