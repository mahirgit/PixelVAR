# Known-Metrics Comparison

This comparison uses the shared image-folder evaluator on 4096 generated images
per method and the same validation reference folder.

Feature space: Inception V3.

| Method | FID lower | KID lower | Precision higher | Recall higher | Density higher | Coverage higher | MS-SSIM lower | Exact match lower |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Flat raster AR | 9.3558 | 0.002975 | 0.9172 | 0.9541 | 0.7654 | 0.8518 | 0.8281 | 0.0112 |
| HMAR step=1 | 12.6555 | 0.005209 | 0.6494 | 0.9314 | 0.4136 | 0.6165 | 0.8309 | 0.0005 |
| PixelVAR main | 13.2516 | 0.006010 | 0.6406 | 0.9292 | 0.4093 | 0.6050 | 0.8323 | 0.0007 |
| Flat MaskGIT | 67.3908 | 0.061825 | 0.1545 | 0.0317 | 0.0455 | 0.0454 | 0.9426 | 0.0000 |

## Readout

Flat raster AR is the numeric winner on the known image-generation metrics. It
beats PixelVAR main and HMAR on FID, KID, precision, recall, density, and
coverage.

That result is not clean. A direct token-level memorization audit found that
flat AR exactly reproduced `3162 / 4096` training images, `365 / 4096`
validation images, and `333 / 4096` test images. The processed dataset had no
cross-split exact duplicates, so the held-out exact matches are not explained by
train/validation duplicate leakage. Flat AR should be treated as a memorizing
baseline, not a clean generative winner.

Flat MaskGIT failed under the current setup. Its FID/KID are much worse, recall
and coverage are low, and MS-SSIM is high, which points to poor diversity.

## Current Standing

For report honesty:

- Known metrics: flat raster AR is best.
- Audit-adjusted known metrics: flat raster AR should be disqualified or
  reported with a memorization warning.
- Memorization audit: PixelVAR main has `13` validation and `11` test exact
  matches; HMAR step=1 has `15` validation and `22` test exact matches. These
  are low but nonzero and should be disclosed.
- Sprite-feature evaluator: PixelVAR main was best among the original proposal
  branches.
- Conservative claim: HMAR step=1 and PixelVAR main are the strongest
  non-memorizing models measured so far, with HMAR slightly ahead on Inception
  metrics and PixelVAR ahead on the sprite-feature evaluator.

Full output:

- `reports/external_eval/main_hmar_flat_known_metrics/evaluation_report.md`
- `reports/external_eval/main_hmar_flat_known_metrics/metrics.csv`
- `reports/memorization_audit/flat_ar/audit_report.md`
- `reports/final/memorization_audit_summary.md`
- `reports/memorization_audit/flat_ar/nearest_pairs.png`
- `reports/final/four_way_sample_sheet.png`
