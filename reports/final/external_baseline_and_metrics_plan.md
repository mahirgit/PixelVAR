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

Recommendation: do not spend on the larger SD-piXL batch without first tuning
the SD-piXL setup. The current evidence suggests SD-piXL should remain a
qualitative related external baseline path, unless we invest extra time in
prompt/control/reference tuning.

Do not treat the one-image smoke run as a metric result. Use it only to verify
the environment, downloads, prompt path, palette conversion, and output
normalization. Numeric metrics should only be reported if we generate enough
SD-piXL samples to make FID/KID/PRDC at least minimally meaningful; for this
optimization-based method, that may be too expensive and it may remain a
qualitative related-work baseline.

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

Recommendation after the metric smoke: keep practical diffusion as an external
qualitative/comparison baseline, but do not spend on a 4096-image practical
diffusion run unless we specifically need a large negative external-baseline
number. The 64-image smoke already shows a large quality gap versus PixelVAR.

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
8. Done: add and run the practical SSD-1B diffusion baseline smoke, then inspect
   `reports/final/practical_diffusion_sample_sheet.png`.
9. Done: run a 64-image practical-diffusion metric smoke against PixelVAR main.
10. Next: keep SD-piXL qualitative and practical diffusion as a weak external
    metric/visual baseline unless we decide to spend on a larger negative result.
11. Try MDIGAN only if we can adapt its conditional pose task cleanly to our data;
   otherwise cite it as related work rather than a direct numeric baseline.
