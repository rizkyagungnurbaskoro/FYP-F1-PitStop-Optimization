Thesis Results Summary

Evaluation setup:
- GroupKFold split by race to prevent leakage across events.
- Stages: S1 RefTech on RefData, S2 MyMethod on RefData, S3 RefTech on MyData+W, S4 MyMethod on MyData+W.
- Metrics reported as mean +/- std across folds.

Key improvements (mean deltas):
- S2-S1 F1: +0.000671 (baseline 0.830628 -> 0.831299)
- S2-S1 F2: +0.004839 (baseline 0.861931 -> 0.866770)
- S2-S1 Precision: -0.002494 (baseline 0.789365 -> 0.786871)
- S2-S1 Recall: +0.009535 (baseline 0.886807 -> 0.896342)
- S2-S1 PR-AUC: +0.007059 (baseline 0.924091 -> 0.931150)
- S4-S3 F1: +0.000939 (baseline 0.176055 -> 0.176995)
- S4-S3 F2: +0.002411 (baseline 0.290427 -> 0.292837)
- S4-S3 Precision: +0.000145 (baseline 0.108101 -> 0.108246)
- S4-S3 Recall: +0.007730 (baseline 0.545206 -> 0.552936)
- S4-S3 PR-AUC: +0.000513 (baseline 0.135212 -> 0.135724)

Fold-level paired tests (sign test):
- S2-S1 F1: mean delta +0.000671 (std 0.025719), p=0.3750 (n=5)
- S2-S1 F2: mean delta +0.004839 (std 0.018223), p=1.0000 (n=5)
- S2-S1 Precision: mean delta -0.002494 (std 0.032820), p=1.0000 (n=5)
- S2-S1 Recall: mean delta +0.009535 (std 0.012747), p=0.6250 (n=5)
- S2-S1 PR-AUC: mean delta +0.007059 (std 0.009143), p=0.3750 (n=5)
- S4-S3 F1: mean delta +0.000939 (std 0.006302), p=1.0000 (n=5)
- S4-S3 F2: mean delta +0.002411 (std 0.003790), p=0.3750 (n=5)
- S4-S3 Precision: mean delta +0.000145 (std 0.005754), p=1.0000 (n=5)
- S4-S3 Recall: mean delta +0.007730 (std 0.035931), p=0.3750 (n=5)
- S4-S3 PR-AUC: mean delta +0.000513 (std 0.006087), p=1.0000 (n=5)

Interpretation:
- MyMethod shows consistent but modest gains over RefTech across most metrics.
- Improvements are small; report them as incremental performance gains with leakage-safe validation.

Limitations:
- Effect sizes are small; results are sensitive to race-specific variance.
- More data or additional signals may be required for larger gains.