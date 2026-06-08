# PixelVAR Visual Results and Presentation Deck

Date: 2026-06-07

This document collects the visual outputs, result tables, generated charts, and
presentation-ready interpretation for the current PixelVAR project state. The
goal is not only to show scores, but also to explain which results are strong,
which results are expected from the design, and which results need caution.

## Slide 1 - Main Takeaway

PixelVAR's main result is a working 32x32 pixel-art sprite generator that
operates directly in a discrete palette-token space and generates sprites in a
coarse-to-fine order.

Main model:

- Model: `var_sprites_v0_full`
- Resolution: `32x32`
- Token space: transparent token + 16 palette color tokens
- Scales: `[1, 2, 4, 8, 16, 32]`
- Total tokens: `1365`
- Best sampling setting: `temperature=0.8`, `top_k=8`
- Main sprite-feature score: `0.00147`
- Palette consistency: `1.0000`

Presentation line:

> The claim is not broad SOTA. The claim is that we built a palette-safe,
> coarse-to-fine, audit-aware generative pipeline for 32x32 pixel-art sprite
> generation.

## Slide 2 - Final Branch Comparison

![Final branch comparison](final_branch_comparison_sheet.png)

This sheet shows the main branches compared near the end of the project:

- Real-only PixelVAR
- HMAR masked refinement
- Patch-VQ VAR
- Other ablation outputs

Interpretation:

Real-only PixelVAR remained the main result. HMAR was technically successful and
is a close alternative, but the proposal-internal sprite-feature evaluator still
selected PixelVAR. Patch-VQ showed that the learned-token path works, but its
outputs were more blocky and did not beat the deterministic palette-token path.

## Slide 3 - Main PixelVAR Outputs

![Main PixelVAR sample sheet](final_main_var_sample_sheet.png)

This is the selected sample sheet for the main PixelVAR model.

Interpretation:

The model generally preserves the expected sprite structure: transparent
background, compact character shape, palette-limited colors, and crisp pixel-art
edges. This sheet is useful visually, but it should not be treated as the only
evidence. The later slides add metrics and audit results.

## Slide 4 - HMAR Masked Refinement Outputs

![HMAR sample sheet](final_hmar_sample_sheet.png)

HMAR was implemented as the proposal's Option B. Its logic is:

- Keep the same coarse-to-fine hierarchy.
- Mask target-scale tokens.
- Predict masked tokens in parallel.
- Optionally refine the scale over multiple steps.

Best HMAR setting:

- `refinement_steps=1`
- `temperature=0.8`
- `top_k=8`
- Feature score: `0.00189`

Interpretation:

HMAR did not fail. It is a strong alternative, and on some Inception-based
known metrics it is slightly ahead of PixelVAR. However, the sprite-feature
evaluator still prefers the main PixelVAR model. More refinement steps also made
the score worse, so "more refinement" was not automatically better in this
setting.

## Slide 5 - Patch-VQ / Learned Token Branch

![Patch-VQ sample sheet](final_patchvq_sample_sheet.png)

Patch-VQ was tested as an alternative learned-token path after the neural VQ-VAE
reconstruction quality did not pass the pixel-art quality gate. Instead of a
neural tokenizer, Patch-VQ uses KMeans over 2x2 image patches.

Patch-VQ result:

- Token map: `(93,312, 16, 16)`
- Vocabulary size: `512`
- Used codes: `143 / 512`
- VAR validation loss: `0.10122`
- VAR validation accuracy: `0.96905`
- Best decoded score: `0.04689`

Interpretation:

Patch-VQ worked technically, but it was not promoted. The samples are coherent
but more blocky. This does not mean learned tokens are impossible; it means the
best completed path so far is deterministic palette-token PixelVAR.

## Slide 6 - Patch-VQ Reconstruction Check

![Patch-VQ reconstruction comparison](../assets/sprites_patchvq16_reconstruction_compare_grid.png)

This image shows Patch-VQ reconstruction behavior.

Interpretation:

Patch-VQ preserves the overall sprite form, but the patch-based representation
introduces block artifacts. For pixel art, some blockiness can be acceptable,
but here it is less clean and less expressive than the palette-token PixelVAR
outputs.

## Slide 7 - Model Decision Scores

![Model decision scores](visuals/chart_model_decision_scores.png)

Lower is better.

| Rank | Branch | Best score | Comparable? | Decision |
| ---: | --- | ---: | --- | --- |
| 1 | Real-only VAR | `0.00147` | yes | Main result |
| 2 | HMAR masked refinement | `0.00189` | yes | Strong ablation |
| 3 | Generated-keep VAR | `0.00157` | no | Self-reference only |
| 4 | Real + generated mixed VAR | `0.00755` | yes | Worse than main |
| 5 | OpenGameArt-mixed VAR | `0.00778` | yes | Worse than main |
| 6 | Patch-VQ VAR | `0.04689` | no | Learned-token ablation |

Important note:

Generated-keep VAR has a low score, but it is not evaluated against the same
real validation reference. Patch-VQ also uses a separate decoded RGBA evaluator,
so it is not a direct winner comparison.

Presentation line:

> Among the directly comparable real-validation results, the main PixelVAR model
> is the best. HMAR is close, but it does not beat the main model on the
> sprite-feature evaluator.

## Slide 8 - Pixel-Art Structure Metrics

![Pixel-art structure metrics](visuals/chart_structure_metrics.png)

This chart compares the reference set, PixelVAR, and HMAR on two structural
pixel-art metrics:

- Opaque ratio
- Edge density

Main PixelVAR:

- Opaque ratio: `0.2276`
- Reference opaque ratio: `0.2353`
- Edge density: `0.1802`
- Reference edge density: `0.1807`

Interpretation:

Opaque ratio being close to the reference means the model learned a realistic
sprite fill ratio. If it were too low, sprites would be too empty or incomplete.
If it were too high, outputs would be too dense or blob-like.

Edge density being very close to the reference is a good pixel-art signal. It
suggests that the model is not producing overly blurry or overly noisy images.
Still, edge density is not a full semantic quality metric; it is a structural
proxy.

## Slide 9 - Palette Consistency

In the known-metrics run, palette consistency is `1.0000` for all internal
models.

| Model | Palette consistency | Interpretation |
| --- | ---: | --- |
| PixelVAR main | `1.0000` | Expected and good |
| HMAR step=1 | `1.0000` | Expected and good |
| Flat AR | `1.0000` | Expected and good |
| Flat MaskGIT | `1.0000` | Expected and good |

This result should not be oversold.

Palette consistency of `1.0000` looks excellent, and it is important, but it is
mostly expected from our representation. The model does not generate continuous
RGB values. It selects from a fixed token vocabulary, and decoding maps those
tokens back to fixed palette colors.

Presentation line:

> Palette consistency of 1.0000 is not a quality miracle. It confirms that our
> discrete palette-token design works as intended.

## Slide 10 - 170K Generation Target

![170K quality gate](visuals/chart_170k_gate.png)

170K generation result:

| Category | Count | Rate |
| --- | ---: | ---: |
| Keep | `161,479` | `94.99%` |
| Review | `8,500` | `5.00%` |
| Reject | `21` | `0.01%` |
| Total | `170,000` | `100%` |

Interpretation:

The model did not collapse during large-scale sampling. However, the gate is an
automatic quality gate, not a human study. We should not claim that 161K samples
were manually judged to be perfect.

Better wording:

> We generated 170K samples, and the large majority passed the automatic quality
> gate as usable.

## Slide 11 - 170K Keep Samples

![170K keep random](../assets/sprites_170k_inspection_keep_random.png)

This sheet shows random examples from the keep category.

Interpretation:

The examples generally preserve sprite structure: transparent background,
compact body, and limited palette. But this is still a filtered subset selected
by automatic rules, so it should be presented together with the metrics and
audit.

## Slide 12 - 170K Review and Reject Samples

Review examples:

![170K review highest score](../assets/sprites_170k_inspection_review_highest_score.png)

Reject examples:

![170K rejects](../assets/sprites_170k_inspection_rejects.png)

Interpretation:

Review samples are outputs that automatic rules considered worth checking more
carefully. The reject count is very low, which is a good sign, but it is still
based on proxy rules rather than human preference labels.

## Slide 13 - Known Metrics: FID

![Known FID](visuals/chart_known_fid.png)

Known metrics were computed on 4096 generated samples using Inception V3 feature
space.

| Model | FID ↓ | KID ↓ | Precision ↑ | Recall ↑ | Coverage ↑ | MS-SSIM ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flat raster AR | `9.3558` | `0.002975` | `0.9172` | `0.9541` | `0.8518` | `0.8281` |
| HMAR step=1 | `12.6555` | `0.005209` | `0.6494` | `0.9314` | `0.6165` | `0.8309` |
| PixelVAR main | `13.2516` | `0.006010` | `0.6406` | `0.9292` | `0.6050` | `0.8323` |
| Flat MaskGIT | `67.3908` | `0.061825` | `0.1545` | `0.0317` | `0.0454` | `0.9426` |

At first glance, Flat raster AR is the numerical winner. This table alone is
not enough, because the memorization audit shows that Flat AR strongly
memorizes the dataset.

Presentation line:

> Flat AR wins the raw FID table, but after the audit it cannot be treated as a
> clean generative winner.

## Slide 14 - Precision and Recall

![Precision recall](visuals/chart_precision_recall.png)

Interpretation:

Flat AR has very high precision and recall. Normally this would be a strong
result. With the memorization audit, however, these values are likely inflated
by exact or near-exact reproduction of training samples.

HMAR and PixelVAR are close. HMAR is slightly ahead on some Inception metrics,
while PixelVAR is ahead on the sprite-feature evaluator. The most honest
framing is that PixelVAR and HMAR are the strongest non-memorizing candidates
measured so far.

## Slide 15 - Memorization Audit

![Memorization audit](visuals/chart_memorization_audit.png)

Exact token-map audit over 4096 generated samples:

| Model | Train exact | Val exact | Test exact | Generated duplicates | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| PixelVAR main | `157` | `13` | `11` | `4` | Low but nonzero |
| HMAR step=1 | `154` | `15` | `22` | `5` | Low but nonzero |
| Flat AR | `3162` | `365` | `333` | `206` | Memorizing |

This is one of the most important methodological results in the project.

Flat AR looked best on FID/KID, but `3162 / 4096` generated samples exactly
matched training images. That makes it a memorizing baseline, not a clean
generative winner.

PixelVAR and HMAR exact-match counts are not zero and should be disclosed.
However, their behavior is qualitatively different from Flat AR.

Presentation line:

> Flat AR is suspiciously good. The audit shows that the raw metric advantage is
> largely tied to memorization. PixelVAR and HMAR are cleaner candidates.

## Slide 16 - Why Flat AR Needs Caution

![Flat AR nearest pairs](../memorization_audit/flat_ar/nearest_pairs.png)

Interpretation:

Flat AR's FID result is not reliable on its own. Many generated samples are
exact copies or extremely close to existing dataset samples. This means the
model is not clearly learning a new generative distribution; it is heavily
reproducing data.

Therefore, Flat AR should be presented as:

- Raw metrics winner
- Audit-adjusted disqualified model
- A useful baseline that exposes memorization risk

## Slide 17 - Flat MaskGIT Result

![Four-way sample sheet](four_way_sample_sheet.png)

Flat MaskGIT was weak under the current setup.

Important metrics:

- FID: `67.3908`
- KID: `0.061825`
- Precision: `0.1545`
- Recall: `0.0317`
- Coverage: `0.0454`
- MS-SSIM: `0.9426`

Interpretation:

This does not mean MaskGIT is a bad idea generally. It means this flat setup and
training configuration did not work well for our 32x32 palette-token sprites.
Distribution coverage was poor, FID/KID were bad, and the diversity signal was
weak.

## Slide 18 - Mixed Data Experiments

Mixed training results:

| Experiment | Training data | Real-val score | Decision |
| --- | --- | ---: | --- |
| Real-only VAR | Real MSD Sprites | `0.00147` | Main result |
| Generated-keep VAR | Filtered generated data | `0.00157` | Not directly comparable |
| Real + generated mixed VAR | Real + generated | `0.00755` | Did not beat main |
| OpenGameArt-mixed VAR | Real + generated + OGA | `0.00778` | Did not beat main |

Generated-keep examples:

![Generated keep sample](../assets/sprites_generated_keep_v0_full_t08_top8.png)

Mixed model examples:

![Mixed sample](../assets/sprites_mixed_v0_full_t08_top8.png)

OpenGameArt-mixed examples:

![OpenGameArt mixed sample](../assets/sprites_mixed_oga_v0_full_t08_top8.png)

Interpretation:

These experiments show that more data is not automatically better. Generated
data and OpenGameArt data made the training set larger, but they did not improve
real-validation quality over the real-only PixelVAR model.

## Slide 19 - OpenGameArt Data Check

![OpenGameArt data check](../assets/opengameart_data_check_grid.png)

OpenGameArt curated data:

- Curated frames: `4,659`
- Groups: `105`
- Train: `3,493`
- Validation: `502`
- Test: `664`

Interpretation:

OpenGameArt was useful as an additional public data source, but it did not match
the main MSD Sprites validation distribution closely enough to improve the main
score.

## Slide 20 - Why There Is No 64x64 Result Yet

A likely question is: "Why did we not test 64x64?"

Short answer:

We prioritized completing the proposal's main 32x32 system first. Moving to
64x64 is technically possible, but it significantly increases cost.

Token count:

| Resolution | Pyramid | Token count |
| --- | --- | ---: |
| 32x32 | `1+4+16+64+256+1024` | `1365` |
| 64x64 | `1+4+16+64+256+1024+4096` | `5461` |

64x64 is about 4x longer in sequence length. For Transformer training, this
increases memory, sampling time, evaluation time, and experiment turnaround.

We also had infrastructure friction with Lightning AI and Modal, including long
job handling, spend limits, GPU limits, and local command time limits. Because
of that, the more responsible path was to finish the 32x32 pipeline, baselines,
metrics, and audit before moving to a larger-resolution experiment.

Presentation line:

> 64x64 was postponed because the correct priority was to finish the 32x32
> proposal pipeline and comparisons first. 64x64 is a next-stage experiment, not
> a completed result.

## Slide 21 - External Baseline Status

Three external baselines have now been run through the shared 32x32 image-folder
protocol:

| External baseline | Run status | Result |
| --- | --- | --- |
| SD-piXL | 16-image metric batch | Valid run, but visually poor and far behind PixelVAR |
| Practical diffusion / SSD-1B | 256-image metric run | Recognizable sprites sometimes, but much worse than PixelVAR |
| Pokemon sprite SDXL LoRA | 256-image metric run | Best external diffusion-style visual baseline, still far behind PixelVAR |

SD-piXL is the more targeted related method because it is explicitly about
pixel-art-like score-distillation generation. However, under our fixed-prompt
32x32 sprite protocol it produced noisy tiled outputs rather than centered
characters.

Correct presentation wording:

> We ran external baselines, but they should be presented carefully. SD-piXL is
> a serious targeted attempt, yet it fails under our adapted 32x32 sprite
> protocol. SSD-1B and the Pokemon sprite LoRA are practical diffusion-style
> comparisons; both are useful, but neither beats PixelVAR under the shared
> evaluator.

## Slide 22 - Final Result Interpretation

Honest quality summary:

| Model / Experiment | Status | Interpretation |
| --- | --- | --- |
| PixelVAR main | Good and defensible | Main winner, palette-safe, low exact-match rate |
| HMAR step=1 | Strong alternative | Close on Inception metrics, behind on sprite-feature |
| Flat AR | Suspiciously good | Strong raw metrics, disqualified by memorization |
| Flat MaskGIT | Weak | Failed under current setup |
| Generated/mixed data | Moderate | Larger dataset did not improve real-val quality |
| Patch-VQ | Technically works | Learned-token path works, but blocky and worse |
| SD-piXL external | Weak | Targeted baseline ran, but metrics and visuals are poor |
| Practical diffusion | Weak/moderate | Generic diffusion baseline, not directly competitive |
| Pokemon sprite LoRA | Moderate | Best external diffusion-style visual baseline, still below PixelVAR metrics |

Final claim:

> PixelVAR is a working palette-safe, coarse-to-fine generator for 32x32
> pixel-art sprites. Among the completed internal models, it is the strongest
> result on the sprite-feature evaluator with a much cleaner memorization profile
> than Flat AR. External diffusion baselines do not beat it under our protocol,
> but without broader external reproductions and 64x64 experiments, we should
> not make a broader SOTA claim.

## Slide 23 - Output and Artifact List

Useful files for presentation:

| Content | File |
| --- | --- |
| Main final sample sheet | `reports/final/final_main_var_sample_sheet.png` |
| HMAR sample sheet | `reports/final/final_hmar_sample_sheet.png` |
| Patch-VQ sample sheet | `reports/final/final_patchvq_sample_sheet.png` |
| Final branch comparison | `reports/final/final_branch_comparison_sheet.png` |
| Four-way model comparison | `reports/final/four_way_sample_sheet.png` |
| SD-piXL external sheet | `reports/final/sd_pixl_sample_sheet.png` |
| Practical diffusion sheet | `reports/final/practical_diffusion_sample_sheet.png` |
| Pokemon sprite LoRA sheet | `reports/final/pokemon_sprite_lora_sample_sheet.png` |
| Model decision table | `reports/final/model_decision_table.md` |
| Known metrics comparison | `reports/final/known_metrics_comparison.md` |
| External baseline comparison | `reports/final/external_baseline_comparison.md` |
| Memorization audit summary | `reports/final/memorization_audit_summary.md` |
| Detailed Turkish report | `reports/final/turkish_project_status_report.md` |
| Three-person Turkish script | `reports/final/turkish_three_person_presentation_script.md` |
| English visual deck | `reports/final/visual_results_deck.md` |

Generated chart files:

| Chart | File |
| --- | --- |
| Model decision scores | `reports/final/visuals/chart_model_decision_scores.png` |
| FID comparison | `reports/final/visuals/chart_known_fid.png` |
| Precision / Recall | `reports/final/visuals/chart_precision_recall.png` |
| Structure metrics | `reports/final/visuals/chart_structure_metrics.png` |
| Memorization audit | `reports/final/visuals/chart_memorization_audit.png` |
| 170K quality gate | `reports/final/visuals/chart_170k_gate.png` |

## Slide 24 - Short Closing

The project completed the core 32x32 PixelVAR pipeline from the proposal. The
model generates sprites within a fixed palette, matches key structural metrics
closely, and was compared against HMAR, Flat AR, Flat MaskGIT, and Patch-VQ
branches.

The most important caution is Flat AR: it looks strongest on raw FID/KID, but
the memorization audit disqualifies it as a clean winner. PixelVAR and HMAR are
therefore the more reliable non-memorizing candidates measured so far.

The remaining gaps are clear: more ablations, user study, MDIGAN only if a clean
conditional-to-unconditional adaptation is justified, and 64x64 experiments.
These were postponed because of compute, infrastructure, and time constraints,
and because the 32x32 proposal result and external-baseline section had to be
made complete first.
