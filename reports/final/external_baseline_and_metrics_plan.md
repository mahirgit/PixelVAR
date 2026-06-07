# External Baseline and Metrics Plan

## Current Status

The current PixelVAR result is a strong internal result, not a complete external
benchmark result.

We have compared these completed branches:

- Real-only palette-token VAR
- HMAR masked refinement
- Real + generated-data VAR
- OpenGameArt-mixed VAR
- Generated-keep VAR
- Patch-VQ VAR

The main score used so far is a lightweight Frechet distance over sprite
features. It is useful for choosing among our branches, but it is not enough on
its own for a serious paper-style comparison.

## External Baselines to Use

### Tier 1: Most Related Pixel-Art/Sprite Baselines

1. `Generating Pixel Art Character Sprites using GANs`
   - Why: direct pixel-art character sprite generation.
   - Caveat: conditional pose-to-pose generation, not exactly unconditional
     sprite sampling.
   - Paper: https://arxiv.org/abs/2208.06413

2. `A Missing Data Imputation GAN for Character Sprite Generation` / MDIGAN
   - Why: direct character sprite generation with released code.
   - Caveat: also pose-imputation rather than unconditional sampling.
   - Paper: https://arxiv.org/abs/2409.10721
   - Code: https://github.com/fegemo/mdigan-characters

3. `SD-piXL`
   - Why: recent SIGGRAPH Asia 2024 work for low-resolution, color-limited
     imagery including pixel art.
   - Caveat: prompt/image-conditioned optimization method, not trained on our
     dataset by default.
   - Paper: https://arxiv.org/abs/2410.06236
   - Code: https://github.com/AlexandreBinninger/SD-piXL

### Tier 2: Architecture-Adjacent Baselines

4. Flat raster-scan autoregressive transformer
   - Why: the standard next-token baseline that VAR claims to improve over.
   - Caveat: we should implement this locally on the same palette tokens.

5. Flat MaskGIT
   - Why: known masked image-token baseline and closest contrast to HMAR.
   - Caveat: we should implement this locally on the same palette tokens.
   - Paper: https://arxiv.org/abs/2202.04200

6. Patch-VQ/VQGAN-style token baseline
   - Why: checks whether learned visual tokens beat deterministic palette
     tokens.
   - Caveat: our first patch-VQ branch was visibly blockier and used a separate
     decoded-image evaluator.

### Tier 3: Practical External Generator Baseline

7. SDXL or Stable Diffusion pixel-art LoRA + quantization
   - Why: a practical external baseline people will expect.
   - Caveat: outputs must be normalized to 32x32 transparent PNGs before
     evaluation, and prompt choice must be fixed.

This tier is useful for a demo/report, but it should not be the only external
comparison because it is sensitive to prompts and post-processing.

## Metrics to Report

### Known Image-Generation Metrics

1. FID
   - Standard real-vs-generated distribution distance using Inception features.
   - Lower is better.
   - Caveat: Inception features are imperfect for tiny pixel art, so this should
     be reported alongside domain metrics.

2. KID
   - Kernel Inception Distance using polynomial MMD.
   - Lower is better.
   - Useful because it is common in generative-model evaluation and more stable
     to report with mean/std over subsets.

3. Precision and Recall
   - Precision measures sample fidelity; recall measures coverage/diversity.
   - Higher is better.

4. Density and Coverage
   - Related to precision/recall but designed to be more reliable and
     interpretable.
   - Higher is better.

5. Mean MS-SSIM
   - Diversity check across generated samples.
   - Lower means more diverse samples.
   - Should be compared against the real validation set's MS-SSIM, not read in
     isolation.

### Pixel-Art-Specific Metrics

1. Palette consistency
   - Share of opaque generated pixels that exactly belong to the project
     palette.
   - Higher is better.

2. Unique opaque colors per sprite
   - Checks whether outputs respect limited-color pixel-art behavior.

3. Opaque ratio and silhouette delta
   - Measures whether generated sprites occupy a realistic amount of the 32x32
     canvas.

4. Edge density
   - Proxy for outline/detail density.

5. Pixel nearest-neighbor distance
   - Memorization check against the held-out validation set.
   - Exact-match rate should be near zero.

## New Repo Tools

1. Export comparable image folders:

```bash
python scripts/export_eval_images.py \
  --config configs/train/sprites_v0_full.yaml \
  --checkpoint checkpoints/var_sprites_v0_full/best.ckpt \
  --generated-name pixelvar_main \
  --temperature 0.8 \
  --top-k 8 \
  --num-reference 4096 \
  --num-generated 4096 \
  --output-dir outputs/eval_images/pixelvar_main
```

2. Evaluate our model and any external PNG folder:

```bash
python scripts/evaluate_image_folders.py \
  --reference-dir outputs/eval_images/pixelvar_main/reference \
  --generated-dir pixelvar_main=outputs/eval_images/pixelvar_main/pixelvar_main \
  --generated-dir external_method=outputs/external_baselines/external_method \
  --palette-json data/processed/sprites/palette.json \
  --feature-space inception \
  --max-images 4096 \
  --output-dir reports/external_eval/main_vs_external
```

The same evaluator also has `--feature-space pixel` for quick smoke tests when
Inception weights are not available, but paper/report numbers should use
`--feature-space inception`.

3. Run main VAR vs HMAR on Modal:

```bash
modal run modal_train.py --action benchmark-main-hmar-known-metrics --num-samples 4096
```

This writes:

- `outputs/external_eval/main_vs_hmar_known_metrics/metrics.csv`
- `outputs/external_eval/main_vs_hmar_known_metrics/evaluation_report.md`
- `outputs/eval_images/pixelvar_main`
- `outputs/eval_images/hmar_steps1`

A 4096-sample Modal run for PixelVAR main vs HMAR completed successfully:

| Method | FID | KID mean | Precision | Recall | Density | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PixelVAR main | 13.2516 | 0.006010 | 0.6406 | 0.9292 | 0.4093 | 0.6050 |
| HMAR steps=1 | 12.6555 | 0.005209 | 0.6494 | 0.9314 | 0.4136 | 0.6165 |

A second 4096-sample Modal run added flat raster AR and flat MaskGIT:

| Method | FID | KID mean | Precision | Recall | Density | Coverage | Exact match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Flat raster AR | 9.3558 | 0.002975 | 0.9172 | 0.9541 | 0.7654 | 0.8518 | 0.0112 |
| HMAR steps=1 | 12.6555 | 0.005209 | 0.6494 | 0.9314 | 0.4136 | 0.6165 | 0.0005 |
| PixelVAR main | 13.2516 | 0.006010 | 0.6406 | 0.9292 | 0.4093 | 0.6050 | 0.0007 |
| Flat MaskGIT | 67.3908 | 0.061825 | 0.1545 | 0.0317 | 0.0455 | 0.0454 | 0.0000 |

Flat raster AR is the known-metrics winner before auditing, but the follow-up
memorization audit found exact reproduction of `3162 / 4096` train images,
`365 / 4096` validation images, and `333 / 4096` test images. The processed
dataset has no cross-split exact duplicates, so the held-out matches are not a
split-duplicate artifact. Flat AR should be reported as memorizing, not as a
clean winner. Flat MaskGIT is not competitive in the current setup.

## SD-piXL External Baseline Start

SD-piXL is now wired as the first released-code external baseline. It is the
closest current external method because it targets low-resolution, color-limited
imagery and pixel art through score distillation:

- Paper: https://arxiv.org/abs/2410.06236
- Code: https://github.com/AlexandreBinninger/SD-piXL

Important caveat: SD-piXL is prompt/image-conditioned optimization, not an
unconditional sprite sampler trained on our dataset. To keep the comparison
honest, the PixelVAR protocol uses fixed text prompts and does not feed
validation/test sprites into SD-piXL. The raw SD-piXL `final_argmax.png` outputs
are normalized into the same `32x32` RGBA PNG folder protocol used by
`scripts/evaluate_image_folders.py`.

New adapter pieces:

- `configs/external/sd_pixl_prompts.txt`
- `scripts/export_palette_hex.py`
- `scripts/prepare_sd_pixl_baseline.py`
- `scripts/normalize_external_images.py`

Modal actions:

```bash
modal run modal_train.py --action prepare-sd-pixl-baseline
modal run modal_train.py --action run-sd-pixl-smoke --sd-pixl-steps 250
modal run modal_train.py --action run-sd-pixl-batch --num-samples 4 --sd-pixl-steps 1000
modal run modal_train.py --action normalize-sd-pixl-baseline
modal run modal_train.py --action build-sd-pixl-sample-sheet
```

Expected outputs:

- `outputs/external_baselines/sd_pixl/pixelvar_palette.hex`
- `outputs/external_baselines/sd_pixl/workdir/.../final_argmax.png`
- `outputs/external_baselines/sd_pixl/png32/*.png`
- `outputs/final/sd_pixl_sample_sheet.png`

Do not treat the one-image smoke run as a metric result. Use it only to verify
the environment, downloads, prompt path, palette conversion, and output
normalization. Numeric metrics should only be reported if we generate enough
SD-piXL samples to make FID/KID/PRDC at least minimally meaningful; for this
optimization-based method, that may be too expensive and it may remain a
qualitative related-work baseline.

## Recommended Order

1. Done: run the new evaluator on PixelVAR main vs HMAR using the same exported
   PNG protocol.
2. Done: implement and run local flat raster AR and flat MaskGIT baselines.
3. Done: run duplicate/memorization audit for flat AR.
4. Done: generate four-way visual sample sheet with the flat AR memorization
   caveat.
5. Done: run memorization audits for PixelVAR main and HMAR step=1 with the
   same audit script.
6. Done: add SD-piXL as the first external released-code baseline path.
7. Next: run the SD-piXL smoke action on Modal and inspect
   `outputs/final/sd_pixl_sample_sheet.png`.
8. Next: decide whether SD-piXL stays qualitative or whether to spend on a small
   prompt batch for external-folder metrics.
9. Try MDIGAN only if we can adapt its conditional pose task cleanly to our data;
   otherwise cite it as related work rather than a direct numeric baseline.
10. Add one practical SDXL/LoRA+quantization baseline for user-facing
    comparison.
