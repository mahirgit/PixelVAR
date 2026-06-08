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
- SD-piXL external baseline
- SSD-1B practical diffusion baseline
- Pokemon trainer sprite SDXL LoRA baseline

The main score used so far is a lightweight Frechet distance over sprite
features. It is useful for choosing among our branches, but it is not enough on
its own for a serious paper-style comparison.

The consolidated external-baseline table is now in
`reports/final/external_baseline_comparison.md`. That file is the cleanest place
to read PixelVAR vs SD-piXL vs SSD-1B practical diffusion vs Pokemon sprite LoRA
without mixing the individual run notes.

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

4. `PixDiff-PIG`
   - Why: direct palette-informed diffusion for pixel-art generation.
   - Caveat: searched as a targeted next baseline, but no runnable public
     repository or weights were found. Treat as related work unless code is
     released or the authors provide it.
   - Article: https://www.researchgate.net/publication/398825200_PixDiff-PIG_Palette-Informed_Diffusion_for_Pixel_Art_Generation

5. `achsaf/vq-diffusion-pixelart-16x16`
   - Why: architecture-adjacent discrete diffusion over 16x16 pixel-art
     character tokens.
   - Caveat: the model card is relevant, but the model files returned HTTP 401
     during this run, so it could not be downloaded or evaluated. It is also
     native 16x16, so even if access becomes available it must be clearly framed
     as a lower-resolution baseline.
   - Model card: https://huggingface.co/achsaf/vq-diffusion-pixelart-16x16

### Tier 2: Architecture-Adjacent Baselines

6. Flat raster-scan autoregressive transformer
   - Why: the standard next-token baseline that VAR claims to improve over.
   - Caveat: we should implement this locally on the same palette tokens.

7. Flat MaskGIT
   - Why: known masked image-token baseline and closest contrast to HMAR.
   - Caveat: we should implement this locally on the same palette tokens.
   - Paper: https://arxiv.org/abs/2202.04200

8. Patch-VQ/VQGAN-style token baseline
   - Why: checks whether learned visual tokens beat deterministic palette
     tokens.
   - Caveat: our first patch-VQ branch was visibly blockier and used a separate
     decoded-image evaluator.

### Tier 3: Practical External Generator Baseline

9. SDXL or Stable Diffusion pixel-art LoRA + quantization
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

## SD-piXL Smoke Result

Status on 2026-06-08: smoke run completed end to end on Modal B200.

Two adapter issues were fixed before completion:

1. The SD-piXL Modal image originally pinned `torch==2.4.0` with CUDA 12.1,
   which could not run kernels on B200/Blackwell. The SD-piXL image now uses
   `torch==2.8.0`, `torchvision==0.23.0`, and `torchaudio==2.8.0` from the
   CUDA 12.8 wheel index.
2. The PixelVAR palette `.hex` export originally wrote CSS-style `#RRGGBB`
   lines. SD-piXL's palette loader expects bare `RRGGBB`, so the export adapter
   now writes bare six-character hex values.

Completed smoke command:

```bash
modal run modal_train.py --action run-sd-pixl-smoke --sd-pixl-steps 250
```

Pulled artifact:

- `reports/final/sd_pixl_sample_sheet.png`

Visual finding: the smoke output is valid but poor. It appears as noisy
palette-colored blocks rather than a recognizable centered 32x32 character
sprite. Treat this as a pipeline smoke success, not as a competitive baseline
result.

The one-image smoke run was not a metric result. It only verified the
environment, downloads, prompt path, palette conversion, and output
normalization.

## SD-piXL 16-Image Metric Batch

Status on 2026-06-08: completed.

Before the metric run, three SD-piXL adapter issues were corrected:

1. `run-sd-pixl-batch --num-samples 16` previously hit the Modal entrypoint's
   default-value guard and ran only four fresh optimizations. The batch action
   now uses the explicit `--num-samples` value.
2. Old SD-piXL workdir/png32 outputs could leak into a later normalization run.
   The SD-piXL smoke/batch actions now clear those output folders before
   generating new results.
3. SD-piXL normalization now uses the same corner-background transparency
   cleanup used for the practical diffusion baseline. This reduced opaque-ratio
   bias, although it did not make the outputs competitive.

Generation command:

```bash
modal run modal_train.py --action run-sd-pixl-batch --num-samples 16 --sd-pixl-steps 250
```

Evaluation command:

```bash
modal run modal_train.py --action cmd-gpu --cmd "python scripts/evaluate_image_folders.py --reference-dir outputs/eval_images/pixelvar_main/reference --generated-dir pixelvar_main=outputs/eval_images/pixelvar_main/pixelvar_main --generated-dir sd_pixl=outputs/external_baselines/sd_pixl/png32 --palette-json data/processed/sprites/palette.json --feature-space inception --max-images 16 --batch-size 16 --kid-subsets 20 --kid-subset-size 8 --msssim-pairs 120 --output-dir outputs/external_eval/main_vs_sd_pixl_16"
```

Pulled artifacts:

- `reports/external_eval/main_vs_sd_pixl_16/evaluation_report.md`
- `reports/external_eval/main_vs_sd_pixl_16/metrics.csv`
- `reports/final/sd_pixl_sample_sheet.png`

Metric result:

| Method | Images | FID | KID mean | Precision | Recall | Density | Coverage | MS-SSIM | Palette consistency | Opaque ratio | Edge density |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PixelVAR main | 16 | 151.0927 | 0.014713 | 1.0000 | 0.9375 | 1.3500 | 1.0000 | 0.8247 | 1.0000 | 0.2319 | 0.1888 |
| SD-piXL | 16 | 497.7065 | 0.517647 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.4926 | 1.0000 | 0.6720 | 0.4007 |

Interpretation: SD-piXL is not competitive in this adapted protocol. The
16-image sheet still looks like noisy tiled color fields rather than centered
sprite characters. The zero PRDC precision/recall/density/coverage values are
consistent with that visual result. Palette consistency is 1.0 because all
outputs are quantized through the PixelVAR palette, not because SD-piXL learned
the palette distribution.

Recommendation: include SD-piXL as a serious external baseline attempt with
metrics, but clearly mark it as prompt-conditioned score-distillation adapted to
our 32x32 protocol. Do not spend on a 64-image SD-piXL run unless we decide to
tune the SD-piXL setup or need a larger negative result for completeness.

## Practical Diffusion External Baseline

This path adds a practical user-facing external baseline: an SDXL-family
text-to-image model produces raw 512x512 images, then the existing PixelVAR
external-image protocol normalizes them to 32x32 RGBA PNGs with the project
palette. The default smoke uses `segmind/SSD-1B` because it is public,
diffusers-compatible, smaller than full SDXL, and practical for Modal B200
smoke runs.

New adapter pieces:

- `configs/external/practical_diffusion_prompts.txt`
- `scripts/run_practical_diffusion_baseline.py`
- Modal actions in `modal_train.py`:
  - `run-practical-diffusion-smoke`
  - `run-practical-diffusion-batch`
  - `normalize-practical-diffusion-baseline`
  - `build-practical-diffusion-sample-sheet`

Default smoke command:

```bash
modal run modal_train.py --action run-practical-diffusion-smoke --num-samples 4 --diffusion-steps 25
```

Pulled artifact:

- `reports/final/practical_diffusion_sample_sheet.png`

Visual finding: this baseline is much more useful than the SD-piXL smoke. It
generates recognizable pixel-art-like character sprites after palette
normalization. It is still only moderate as a direct research baseline: two of
the four smoke samples show duplicate/lineup artifacts instead of a single
centered sprite. That is a known failure mode for text-to-image pixel-art
prompts, even with negative prompts against sprite sheets and multiple
characters.

LoRA check: `nerijs/pixel-art-xl` with
`pixel-art-xl.safetensors` was also smoke-tested against `segmind/SSD-1B`:

```bash
modal run modal_train.py --action run-practical-diffusion-smoke --num-samples 4 --diffusion-steps 25 \
  --diffusion-lora-id nerijs/pixel-art-xl \
  --diffusion-lora-weight-name pixel-art-xl.safetensors \
  --diffusion-lora-scale 0.8
```

The run completed, but diffusers reported many unexpected adapter keys. This
suggests the LoRA does not map cleanly onto SSD-1B's distilled UNet. The visual
result was not clearly better than the no-LoRA run, so the default remains
`segmind/SSD-1B` without LoRA. A future LoRA baseline should use a matching
full SDXL base if the model license/access path is available.

Recommendation: keep this as the practical external visual baseline now. If we
need numeric external metrics, generate a larger practical-diffusion batch first
and evaluate it through `scripts/evaluate_image_folders.py`. Do not over-claim
from the 4-image smoke sheet.

## Practical Diffusion 64-Image Metric Smoke

Status on 2026-06-08: completed.

Generation command:

```bash
modal run modal_train.py --action run-practical-diffusion-batch --num-samples 64 --diffusion-steps 25
```

Evaluation command:

```bash
modal run modal_train.py --action cmd-gpu --cmd "python scripts/evaluate_image_folders.py --reference-dir outputs/eval_images/pixelvar_main/reference --generated-dir pixelvar_main=outputs/eval_images/pixelvar_main/pixelvar_main --generated-dir practical_diffusion=outputs/external_baselines/practical_diffusion/png32 --palette-json data/processed/sprites/palette.json --feature-space inception --max-images 64 --batch-size 64 --kid-subsets 20 --kid-subset-size 32 --msssim-pairs 512 --output-dir outputs/external_eval/main_vs_practical_diffusion_64"
```

Pulled artifacts:

- `reports/external_eval/main_vs_practical_diffusion_64/evaluation_report.md`
- `reports/external_eval/main_vs_practical_diffusion_64/metrics.csv`
- `reports/final/practical_diffusion_sample_sheet.png`

Metric smoke result:

| Method | Images | FID | KID mean | Precision | Recall | Density | Coverage | MS-SSIM | Palette consistency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PixelVAR main | 64 | 87.8128 | 0.007124 | 0.8125 | 0.9375 | 0.6375 | 0.9219 | 0.8535 | 1.0000 |
| Practical diffusion | 64 | 196.9628 | 0.092097 | 0.0312 | 0.6719 | 0.0094 | 0.0469 | 0.3935 | 1.0000 |

Interpretation: this is not competitive with PixelVAR main under the same
64-image evaluation slice. Practical diffusion has recognizable individual
sprites in some cases, but the 64-sample sheet still contains duplicated
characters, lineup/sprite-sheet outputs, and malformed centered crops. The
palette consistency score is expected to be 1.0 because the outputs are
normalized through the PixelVAR palette protocol; it should not be read as
evidence that the diffusion model itself learned the project palette.

## Practical Diffusion 256-Image Metric Run

Status on 2026-06-08: completed.

Generation command:

```bash
modal run modal_train.py --action run-practical-diffusion-batch --num-samples 256 --diffusion-steps 25 --diffusion-height 512 --diffusion-width 512
```

Evaluation command:

```bash
modal run modal_train.py --action cmd-gpu --cmd "python scripts/evaluate_image_folders.py --reference-dir outputs/eval_images/pixelvar_main/reference --generated-dir pixelvar_main=outputs/eval_images/pixelvar_main/pixelvar_main --generated-dir practical_diffusion=outputs/external_baselines/practical_diffusion/png32 --palette-json data/processed/sprites/palette.json --feature-space inception --max-images 256 --batch-size 64 --kid-subsets 20 --kid-subset-size 128 --msssim-pairs 2048 --output-dir outputs/external_eval/main_vs_practical_diffusion_256"
```

Pulled artifacts:

- `reports/external_eval/main_vs_practical_diffusion_256/evaluation_report.md`
- `reports/external_eval/main_vs_practical_diffusion_256/metrics.csv`
- `reports/final/practical_diffusion_sample_sheet.png`

Metric result:

| Method | Images | FID | KID mean | Precision | Recall | Density | Coverage | MS-SSIM | Palette consistency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PixelVAR main | 256 | 46.4794 | 0.007480 | 0.9102 | 0.8711 | 0.8852 | 0.8906 | 0.8368 | 1.0000 |
| Practical diffusion | 256 | 158.5819 | 0.102159 | 0.0547 | 0.4922 | 0.0141 | 0.0352 | 0.4107 | 1.0000 |

Interpretation after scaling: this remains a useful practical baseline, not a
competitive one. It has better recall than the Pokemon sprite LoRA, which means
it covers more varied regions of the reference feature space, but precision,
density, and coverage remain very low. The sample sheet still contains
lineups, paired characters, object-like outputs, and malformed crops.

Recommendation after the 256-image run: keep practical diffusion as the generic
practical comparison. Do not spend on a 4096-image practical diffusion run
unless we specifically need a large negative external-baseline number.

## Pokemon Trainer Sprite LoRA External Baseline

This path is the strongest practical external generator attempted so far. It
uses the public `sWizad/pokemon-trainer-sprite-pixelart` SDXL LoRA on top of
`stabilityai/stable-diffusion-xl-base-1.0`, then normalizes outputs to the same
32x32 transparent PixelVAR palette protocol. It is still not a perfect
apples-to-apples baseline because it is text-to-image and trained on Pokemon
trainer-style sprites, not our MSD Sprites train split, but it is much more
targeted than generic SSD-1B.

Implemented files/actions:

- `configs/external/pokemon_sprite_lora_prompts.txt`
- Reused `scripts/run_practical_diffusion_baseline.py` with explicit method and
  filename labels.
- Modal actions in `modal_train.py`:
  - `run-pokemon-sprite-lora-smoke`
  - `run-pokemon-sprite-lora-batch`
  - `normalize-pokemon-sprite-lora-baseline`
  - `build-pokemon-sprite-lora-sample-sheet`

Generation command:

```bash
modal run modal_train.py --action run-pokemon-sprite-lora-batch --num-samples 64 --diffusion-steps 25 --diffusion-height 512 --diffusion-width 512
```

Evaluation command:

```bash
modal run modal_train.py --action cmd-gpu --cmd "python scripts/evaluate_image_folders.py --reference-dir outputs/eval_images/pixelvar_main/reference --generated-dir pixelvar_main=outputs/eval_images/pixelvar_main/pixelvar_main --generated-dir pokemon_sprite_lora=outputs/external_baselines/pokemon_sprite_lora/png32 --palette-json data/processed/sprites/palette.json --feature-space inception --max-images 64 --batch-size 64 --kid-subsets 20 --kid-subset-size 32 --msssim-pairs 512 --output-dir outputs/external_eval/main_vs_pokemon_sprite_lora_64"
```

Pulled artifacts:

- `reports/external_eval/main_vs_pokemon_sprite_lora_64/evaluation_report.md`
- `reports/external_eval/main_vs_pokemon_sprite_lora_64/metrics.csv`
- `reports/final/pokemon_sprite_lora_sample_sheet.png`

Metric smoke result:

| Method | Images | FID | KID mean | Precision | Recall | Density | Coverage | MS-SSIM | Palette consistency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PixelVAR main | 64 | 87.8128 | 0.007124 | 0.8125 | 0.9375 | 0.6375 | 0.9219 | 0.8535 | 1.0000 |
| Pokemon sprite LoRA | 64 | 184.3101 | 0.130036 | 0.3125 | 0.2812 | 0.0844 | 0.1094 | 0.5171 | 1.0000 |

Visual finding: this is the best external diffusion-style visual baseline so
far. Many samples are recognizable centered pixel sprites after normalization.
It still has failure modes that matter: some outputs include side fragments,
decorative frames, paired characters, or sprite-sheet-like leftovers. The
metrics match that visual read: much better PRDC than the generic SSD-1B
baseline, but still far below PixelVAR main on FID/KID and coverage.

Recommendation: include this as the primary practical external generator
baseline. Keep SD-piXL as the most research-targeted external attempt and SSD-1B
as a generic practical baseline. Do not claim the LoRA is a direct architecture
competitor to PixelVAR.

Scaled 256-image result:

```bash
modal run modal_train.py --action run-pokemon-sprite-lora-batch --num-samples 256 --diffusion-steps 25 --diffusion-height 512 --diffusion-width 512
modal run modal_train.py --action cmd-gpu --cmd "python scripts/evaluate_image_folders.py --reference-dir outputs/eval_images/pixelvar_main/reference --generated-dir pixelvar_main=outputs/eval_images/pixelvar_main/pixelvar_main --generated-dir pokemon_sprite_lora=outputs/external_baselines/pokemon_sprite_lora/png32 --palette-json data/processed/sprites/palette.json --feature-space inception --max-images 256 --batch-size 64 --kid-subsets 20 --kid-subset-size 128 --msssim-pairs 2048 --output-dir outputs/external_eval/main_vs_pokemon_sprite_lora_256"
```

| Method | Images | FID | KID mean | Precision | Recall | Density | Coverage | MS-SSIM | Palette consistency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PixelVAR main | 256 | 46.4794 | 0.007480 | 0.9102 | 0.8711 | 0.8852 | 0.8906 | 0.8368 | 1.0000 |
| Pokemon sprite LoRA | 256 | 154.0150 | 0.136589 | 0.1172 | 0.1211 | 0.0312 | 0.0508 | 0.5421 | 1.0000 |

Interpretation after scaling: the LoRA remains the best-looking external
diffusion-style baseline, but the larger run makes the gap clearer rather than
smaller. FID improves versus the 64-image smoke, but precision, recall, density,
and coverage are far below PixelVAR. The refreshed sample sheet still shows
recognizable sprites mixed with repeated multi-character, frame, and side-piece
failures.

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
7. Done: run the SD-piXL smoke action on Modal and inspect
   `reports/final/sd_pixl_sample_sheet.png`.
8. Done: run a corrected 16-image SD-piXL metric batch against PixelVAR main.
9. Done: add and run the practical SSD-1B diffusion baseline smoke, then inspect
   `reports/final/practical_diffusion_sample_sheet.png`.
10. Done: run a 64-image practical-diffusion metric smoke against PixelVAR main.
11. Done: searched PixDiff-PIG. No runnable public code/weights found, so it is
    related work rather than a numeric baseline for now.
12. Done: attempted the VQ-Diffusion pixel-art model card path. The model files
    returned HTTP 401 during download checks, so it could not be evaluated.
13. Done: added and ran the public Pokemon trainer sprite SDXL LoRA as a more
    targeted practical external baseline, including a 64-image metric smoke.
14. Done: scaled the Pokemon sprite LoRA to a 256-image metric run.
15. Done: scaled the SSD-1B practical diffusion baseline to a 256-image metric
    run.
16. Try MDIGAN only if we can adapt its conditional pose task cleanly to our data;
   otherwise cite it as related work rather than a direct numeric baseline.
