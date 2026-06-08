# PixelVAR Final Narrative

## Result

The main result is `var_sprites_v0_full`, the real-only deterministic
palette-token VAR trained on the MSD Sprites replacement dataset.

Best setting:

- Checkpoint: `checkpoints/var_sprites_v0_full/best.ckpt`
- Sampling: `temperature=0.8`, `top_k=8`
- Shared real-validation feature score: `0.00147`
- Palette consistency: `1.0000`
- Opaque ratio: `0.2291` vs real validation `0.2354`
- Edge density: `0.1808` vs real validation `0.1811`

The final branch comparison and sample sheets are in:

- `reports/final/model_decision_table.md`
- `reports/final/known_metrics_comparison.md`
- `reports/final/external_baseline_comparison.md`
- `reports/final/memorization_audit_summary.md`
- `reports/final/four_way_sample_sheet.png`
- `reports/final/final_branch_comparison_sheet.png`
- `reports/final/final_main_var_sample_sheet.png`
- `reports/final/external_baseline_and_metrics_plan.md`
- `reports/final/mdigan_decision.md`

## What Was Compared

We compared the main VAR result against internal proposal branches and
ablations:

| Branch | Comparison status | Outcome |
| --- | --- | --- |
| Real-only VAR | Shared real-validation evaluator | Winner |
| HMAR masked refinement | Shared real-validation evaluator | Closest ablation, but worse than VAR |
| Real + generated mixed VAR | Shared real-validation evaluator | Worse than real-only VAR |
| OpenGameArt-mixed VAR | Shared real-validation evaluator | Worse than real-only VAR |
| Generated-keep VAR | Evaluated against generated-keep validation | Useful self-reference, not a direct winner comparison |
| Patch-VQ VAR | Separate decoded RGBA evaluator | Useful learned-token ablation, not a direct winner comparison |

We ran internal baselines and three external baseline attempts. The internal
baselines include HMAR, flat raster AR, and flat MaskGIT. The external baselines
include SD-piXL, practical diffusion / SSD-1B, and a Pokemon trainer sprite SDXL
LoRA, all normalized to the same 32x32 PNG protocol before evaluation. SSD-1B
practical diffusion and Pokemon sprite LoRA were both scaled to 256-image runs
and remain below PixelVAR under the shared evaluator. MDIGAN was reviewed and
classified as related work rather than a direct numeric baseline because its
protocol is conditional paired-pose imputation. PixDiff-PIG, a user study, and
64x64 generation remain future work.

## Metrics Used

### Training Metrics

Training metrics are token-level cross-entropy loss and token accuracy:

- VAR predicts target scale tokens conditioned on coarser scales.
- HMAR predicts masked target-scale tokens conditioned on coarser scales and
  visible same-scale tokens.
- VQ/patch-token branches report the same kind of token loss/accuracy in their
  token space.

These metrics tell us whether the models learned the token distribution. They
do not by themselves prove sample quality.

### Palette-Token Evaluation Score

The main evaluator is `scripts/evaluate_option_a.py`. It samples generated
sprites, decodes tokens back to palette-index maps, and compares generated
sprites against a validation set using a lightweight Frechet distance over
handcrafted sprite features.

The feature vector includes:

- Opaque-pixel ratio
- Transparent-pixel ratio
- Total, horizontal, and vertical token edge density
- Palette-token entropy over opaque pixels
- Silhouette bounding-box width, height, area, centroid, and top-left location
- Full token histogram, including transparent token `0` and palette tokens
  `1..16`

Lower feature score is better. This is a FID-style score, but it is not
Inception FID. It is designed for tiny 32x32 palette-token sprites where
Inception features would be a poor fit and expensive to justify.

Additional reported metrics:

- `palette_consistency`: whether rendered opaque pixels belong to the learned
  global palette. Deterministic palette-token outputs are `1.0000`.
- `opaque_ratio`: how much of each sprite is non-transparent.
- `edge_density`: how much local token change exists, a proxy for outline and
  detail density.
- `token_entropy`: color/token diversity across opaque pixels.

### Patch-VQ Decoded Evaluation

Patch-VQ uses `scripts/evaluate_decoded_patch_vq.py`, because patch-VQ outputs
learned patch codes that must be decoded to RGBA images before comparison.

Its feature score is a separate lightweight Frechet distance over decoded image
features:

- Opaque ratio
- Alpha-edge density
- RGB-edge density
- Color entropy
- RGB mean and standard deviation
- Silhouette bounding-box features
- Coarse RGB histogram

This score is useful for choosing patch-VQ sampling settings, but it is not
directly comparable to the palette-token evaluator used by the main VAR and
HMAR results.

## Final Decision

`var_sprites_v0_full` remains the main model.

HMAR was technically successful and came close with `refinement_steps=1`, but
it did not beat the main VAR score. More HMAR refinement made the samples worse
in this setup: `4` steps scored `0.00366`, and `8` steps scored `0.00822`.

Generated-data expansion and OpenGameArt mixing also did not improve the
real-validation score. Patch-VQ proved the learned-token path works, but the
decoded samples were blockier and used a separate evaluation metric.

The practical conclusion is that, for the current 32x32 low-color sprite
setting, the deterministic palette-token VAR is the strongest completed
proposal path.

## External Benchmark Upgrade

The next benchmark pass should export every method to the same 32x32 PNG folder
format and run `scripts/evaluate_image_folders.py`.

That adds standard image-generation metrics:

- Inception FID
- KID
- Precision and recall
- Density and coverage
- Mean MS-SSIM diversity

It also keeps pixel-art-specific checks:

- Palette consistency
- Unique opaque colors per sprite
- Opaque ratio
- Edge density
- Pixel nearest-neighbor distance against validation sprites

The planned external baseline order is documented in
`reports/final/external_baseline_and_metrics_plan.md`.

A 4096-sample known-metrics run comparing PixelVAR main, HMAR step=1, flat
raster AR, and flat MaskGIT is available in
`reports/external_eval/main_hmar_flat_known_metrics`. In that Inception-feature
evaluation, flat raster AR is the numeric winner and flat MaskGIT fails. The
flat AR win does not survive the duplicate/memorization audit: it exactly
reproduced `3162 / 4096` training images, `365 / 4096` validation images, and
`333 / 4096` test images, with no cross-split exact duplicates in the processed
dataset. Treat flat AR as a memorizing baseline, not as a clean winner.

The four-way visual sheet is `reports/final/four_way_sample_sheet.png`.

External baseline results are also available. SD-piXL was run as a corrected
16-image metric batch in `reports/external_eval/main_vs_sd_pixl_16`; it is a
targeted pixel-art-related baseline, but under our protocol its FID/KID and
PRDC scores are far worse than PixelVAR and its sample sheet is visibly noisy.
Practical diffusion / SSD-1B was run as a 256-image metric batch in
`reports/external_eval/main_vs_practical_diffusion_256`; it sometimes produces
recognizable sprites, but it is still much worse than PixelVAR and should be
treated as a secondary generic diffusion baseline. A more targeted Pokemon
trainer sprite SDXL LoRA was also run as a 256-image metric batch in
`reports/external_eval/main_vs_pokemon_sprite_lora_256`. It is visually the
best external diffusion-style baseline so far, but it still trails PixelVAR
strongly on FID/KID and PRDC and has visible side-fragment, frame, and
paired-character failures.

The same memorization audit was also run for PixelVAR main and HMAR step=1.
PixelVAR exact-matched `13 / 4096` validation and `11 / 4096` test samples.
HMAR exact-matched `15 / 4096` validation and `22 / 4096` test samples. These
counts are low but nonzero; disclose them. They are qualitatively different
from flat AR's `365 / 4096` validation and `333 / 4096` test exact matches.
