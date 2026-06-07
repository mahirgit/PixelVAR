# Memorization Audit Summary

All audits use 4096 generated samples and compare 32x32 palette-token maps
against the processed `sprites` train/val/test splits.

| Model | Train exact | Val exact | Test exact | Generated duplicates | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| PixelVAR main | 157 | 13 | 11 | 4 | Low exact-match rate; disclose and keep |
| HMAR step=1 | 154 | 15 | 22 | 5 | Low exact-match rate; disclose and keep |
| Flat AR | 3162 | 365 | 333 | 206 | Memorizing; disqualify as clean winner |

The processed dataset has `0` cross-split exact duplicate hashes, so held-out
exact matches are not explained by exact train/validation/test duplicate
leakage.

## Interpretation

Flat AR should be treated as a memorizing baseline. It reproduces most generated
samples from the training set and also exact-matches hundreds of held-out
validation/test samples.

PixelVAR main and HMAR step=1 have low but nonzero exact-match counts. They are
not perfectly zero-match models, but their behavior is qualitatively different
from flat AR. They remain the strongest non-memorizing candidates measured so
far, with the caveat that exact matches should be disclosed.

Artifacts:

- `reports/memorization_audit/pixelvar_main/audit_report.md`
- `reports/memorization_audit/hmar_steps1/audit_report.md`
- `reports/memorization_audit/flat_ar/audit_report.md`
