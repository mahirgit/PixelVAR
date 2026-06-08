# PixelVAR Reproducibility Commands

These commands assume Modal is installed and authenticated.

```bash
python -m pip install modal
modal setup
```

On Windows PowerShell, if Modal output hits a Unicode encoding error, run:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
```

## Main Result

Prepare the replacement Sprites dataset:

```bash
modal run modal_train.py --action prepare-sprites
```

Train the real-only VAR ladder:

```bash
modal run modal_train.py --action train-sprites-overfit32
modal run modal_train.py --action train-sprites-debug1k
modal run modal_train.py --action train-sprites-v0-full
```

Evaluate the main checkpoint:

```bash
modal run modal_train.py --action eval-sprites --num-samples 128
```

Sample the final main model:

```bash
modal run modal_train.py --action sample \
  --config configs/train/sprites_v0_full.yaml \
  --checkpoint checkpoints/var_sprites_v0_full/best.ckpt \
  --num-samples 64 \
  --temperature 0.8 \
  --top-k 8 \
  --output outputs/samples/sprites_v0_full_t08_top8.png
```

## HMAR Ablation

Train HMAR:

```bash
modal run modal_train.py --action train-sprites-hmar-overfit32
modal run modal_train.py --action train-sprites-hmar-debug1k
modal run modal_train.py --action train-sprites-hmar-v0-full
```

Run the refinement-step ablation:

```bash
modal run modal_train.py --action eval-hmar-refinement-ablation --num-samples 128
```

Best HMAR sampling found:

```bash
modal run modal_train.py --action sample-hmar \
  --num-samples 64 \
  --temperature 0.8 \
  --top-k 8 \
  --refinement-steps 1
```

## Generated-Data Branch

Generate the proposal-scale synthetic set and import the filtered keep package:

```bash
modal run modal_train.py --action generate-sprites-selected --num-samples 170000
modal run modal_train.py --action prepare-sprites-generated-keep --num-samples 170000
```

Train and evaluate generated-keep:

```bash
modal run modal_train.py --action train-sprites-generated-keep-overfit32
modal run modal_train.py --action train-sprites-generated-keep-debug1k
modal run modal_train.py --action train-sprites-generated-keep-v0-full
modal run modal_train.py --action eval-sprites \
  --config configs/train/sprites_generated_keep_v0_full.yaml \
  --checkpoint checkpoints/var_sprites_generated_keep_v0_full/best.ckpt \
  --output outputs/eval/sprites_generated_keep_v0_full \
  --num-samples 128
```

## Mixed Real + Generated Branch

```bash
modal run modal_train.py --action prepare-sprites-mixed
modal run modal_train.py --action train-sprites-mixed-overfit32
modal run modal_train.py --action train-sprites-mixed-debug1k
modal run modal_train.py --action train-sprites-mixed-v0-full
modal run modal_train.py --action eval-sprites \
  --config configs/train/sprites_v0_full.yaml \
  --checkpoint checkpoints/var_sprites_mixed_v0_full/best.ckpt \
  --output outputs/eval/sprites_mixed_v0_full_realval \
  --num-samples 128
```

If interrupted after checkpoints are saved:

```bash
modal run modal_train.py --action train-sprites-mixed-v0-full --resume
```

## OpenGameArt-Mixed Branch

```bash
modal run modal_train.py --action prepare-opengameart-public
modal run modal_train.py --action prepare-sprites-mixed-opengameart
modal run modal_train.py --action train-sprites-mixed-oga-overfit32
modal run modal_train.py --action train-sprites-mixed-oga-debug1k
modal run modal_train.py --action train-sprites-mixed-oga-v0-full
modal run modal_train.py --action eval-sprites \
  --config configs/train/sprites_v0_full.yaml \
  --checkpoint checkpoints/var_sprites_mixed_oga_v0_full/best.ckpt \
  --output outputs/eval/sprites_mixed_oga_v0_full_realval \
  --num-samples 128
```

If interrupted after checkpoints are saved:

```bash
modal run modal_train.py --action train-sprites-mixed-oga-v0-full --resume
```

## Patch-VQ Learned-Token Branch

```bash
modal run modal_train.py --action prepare-patch-vq-sprites
modal run modal_train.py --action train-sprites-patchvq16-overfit32
modal run modal_train.py --action train-sprites-patchvq16-debug1k
modal run modal_train.py --action train-sprites-patchvq16-v0-full
modal run modal_train.py --action eval-patch-vq-decoded --num-samples 128
```

Best decoded patch-VQ sampling found:

```bash
modal run modal_train.py --action sample-patch-vq-var \
  --num-samples 64 \
  --temperature 1.0 \
  --top-k 16
```

## Download Key Artifacts

```bash
modal volume get pixelvar-outputs /eval/sprites_v0_full modal_outputs/eval --force
modal volume get pixelvar-outputs /eval/hmar_sprites_refinement_ablation modal_outputs/eval --force
modal volume get pixelvar-outputs /eval/sprites_mixed_v0_full_realval modal_outputs/eval --force
modal volume get pixelvar-outputs /eval/sprites_mixed_oga_v0_full_realval modal_outputs/eval --force
modal volume get pixelvar-outputs /eval/sprites_patchvq16_decoded modal_outputs/eval --force
modal volume get pixelvar-checkpoints /var_sprites_v0_full/best.ckpt modal_checkpoints/var_sprites_v0_full_best.ckpt --force
modal volume get pixelvar-checkpoints /hmar_sprites_v0_full/best.ckpt modal_checkpoints/hmar_sprites_v0_full_best.ckpt --force
```

## Rebuild Final Local Artifacts

After metrics and grids are present under `reports/eval`, rebuild the final
decision table and sample sheets:

```bash
python scripts/build_final_artifacts.py
```

Outputs:

- `reports/final/model_decision_table.md`
- `reports/final/model_decision_table.csv`
- `reports/final/final_branch_comparison_sheet.png`
- `reports/final/final_main_var_sample_sheet.png`
- `reports/final/final_hmar_sample_sheet.png`
- `reports/final/final_patchvq_sample_sheet.png`

## External Baseline Metrics

Export a common validation folder and PixelVAR sample folder:

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

Evaluate PixelVAR against any external method that has been normalized to a
folder of PNG sprites:

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

Use `--feature-space pixel` only for quick smoke checks. Report numbers should
use `--feature-space inception`.

## SD-piXL External Baseline

SD-piXL is prompt/image-conditioned score-distillation optimization, so use it
as a released-code external baseline with a fixed prompt protocol. Do not feed
our validation/test sprites as input images.

Prepare the SD-piXL repo clone, PixelVAR palette `.hex`, and SD-piXL config:

```bash
modal run modal_train.py --action prepare-sd-pixl-baseline
```

Run one B200 smoke generation. This builds a separate SD-piXL Modal image,
downloads the required Hugging Face weights, runs one short optimization, then
normalizes `final_argmax.png` to `32x32` PNGs:

```bash
modal run modal_train.py --action run-sd-pixl-smoke --sd-pixl-steps 250
```

Download the smoke outputs:

```bash
mkdir -p outputs/external_baselines/sd_pixl reports/final
modal volume get pixelvar-outputs /external_baselines/sd_pixl/png32 outputs/external_baselines/sd_pixl/png32 --force
modal volume get pixelvar-outputs /final/sd_pixl_sample_sheet.png reports/final/sd_pixl_sample_sheet.png --force
```

Metric batch used in the report. This is expensive because SD-piXL optimizes one
image per run, so the reported direct metric batch uses 16 samples:

```bash
modal run modal_train.py --action run-sd-pixl-batch --num-samples 16 --sd-pixl-steps 250
modal run modal_train.py --action normalize-sd-pixl-baseline
modal run modal_train.py --action build-sd-pixl-sample-sheet
```

Evaluate the 16-image SD-piXL batch with the existing external-folder protocol:

```bash
modal run modal_train.py --action cmd-gpu --cmd "python scripts/evaluate_image_folders.py --reference-dir outputs/eval_images/pixelvar_main/reference --generated-dir pixelvar_main=outputs/eval_images/pixelvar_main/pixelvar_main --generated-dir sd_pixl=outputs/external_baselines/sd_pixl/png32 --palette-json data/processed/sprites/palette.json --feature-space inception --max-images 16 --batch-size 16 --kid-subsets 20 --kid-subset-size 8 --msssim-pairs 120 --output-dir outputs/external_eval/main_vs_sd_pixl_16"
```

Download the SD-piXL metric artifacts:

```bash
mkdir -p reports/external_eval/main_vs_sd_pixl_16 reports/final
modal volume get pixelvar-outputs /external_eval/main_vs_sd_pixl_16/evaluation_report.md reports/external_eval/main_vs_sd_pixl_16/evaluation_report.md --force
modal volume get pixelvar-outputs /external_eval/main_vs_sd_pixl_16/metrics.csv reports/external_eval/main_vs_sd_pixl_16/metrics.csv --force
modal volume get pixelvar-outputs /final/sd_pixl_sample_sheet.png reports/final/sd_pixl_sample_sheet.png --force
```

## Practical Diffusion External Baseline

Run the default SSD-1B practical diffusion smoke. This generates raw 512x512
text-to-image samples, normalizes them to the PixelVAR 32x32 palette protocol,
and builds a sample sheet:

```bash
modal run modal_train.py --action run-practical-diffusion-smoke --num-samples 4 --diffusion-steps 25
```

Download the sample sheet:

```bash
mkdir -p reports/final
modal volume get pixelvar-outputs /final/practical_diffusion_sample_sheet.png reports/final/practical_diffusion_sample_sheet.png --force
```

On Windows PowerShell, force UTF-8 before Modal downloads if the console hits a
checkmark encoding error:

```powershell
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'; modal volume get pixelvar-outputs /final/practical_diffusion_sample_sheet.png reports/final/practical_diffusion_sample_sheet.png --force
```

Optional LoRA smoke. This was tested, but it is not the default because the
Pixel Art XL LoRA loaded only partially against SSD-1B and did not clearly
improve the 4-image visual result:

```bash
modal run modal_train.py --action run-practical-diffusion-smoke --num-samples 4 --diffusion-steps 25 \
  --diffusion-lora-id nerijs/pixel-art-xl \
  --diffusion-lora-weight-name pixel-art-xl.safetensors \
  --diffusion-lora-scale 0.8
```

If we want numeric metrics for this external baseline, generate a larger batch
first:

```bash
modal run modal_train.py --action run-practical-diffusion-batch --num-samples 64 --diffusion-steps 25
```

Then evaluate through the same external-folder protocol:

```bash
python scripts/evaluate_image_folders.py \
  --reference-dir outputs/eval_images/pixelvar_main/reference \
  --generated-dir pixelvar_main=outputs/eval_images/pixelvar_main/pixelvar_main \
  --generated-dir practical_diffusion=outputs/external_baselines/practical_diffusion/png32 \
  --palette-json data/processed/sprites/palette.json \
  --feature-space inception \
  --max-images 64 \
  --output-dir reports/external_eval/main_vs_practical_diffusion
```

Train the flat baselines and run the four-way known-metrics comparison:

```bash
modal run modal_train.py --action train-flat-ar-ladder
modal run modal_train.py --action train-flat-maskgit-ladder
modal run modal_train.py --action benchmark-main-hmar-flat-known-metrics --num-samples 4096
```

Download the four-way comparison:

```bash
mkdir -p reports/external_eval/main_hmar_flat_known_metrics
modal volume get pixelvar-outputs /external_eval/main_hmar_flat_known_metrics/evaluation_report.md reports/external_eval/main_hmar_flat_known_metrics/evaluation_report.md --force
modal volume get pixelvar-outputs /external_eval/main_hmar_flat_known_metrics/metrics.csv reports/external_eval/main_hmar_flat_known_metrics/metrics.csv --force
```

Run and download the flat AR memorization audit:

```bash
modal run modal_train.py --action audit-flat-ar-memorization --num-samples 4096
mkdir -p reports/memorization_audit/flat_ar
modal volume get pixelvar-outputs /memorization_audit/flat_ar/audit_report.md reports/memorization_audit/flat_ar/audit_report.md --force
modal volume get pixelvar-outputs /memorization_audit/flat_ar/audit_summary.json reports/memorization_audit/flat_ar/audit_summary.json --force
modal volume get pixelvar-outputs /memorization_audit/flat_ar/exact_matches.csv reports/memorization_audit/flat_ar/exact_matches.csv --force
modal volume get pixelvar-outputs /memorization_audit/flat_ar/nearest_neighbors.csv reports/memorization_audit/flat_ar/nearest_neighbors.csv --force
modal volume get pixelvar-outputs /memorization_audit/flat_ar/nearest_pairs.png reports/memorization_audit/flat_ar/nearest_pairs.png --force
```

Run and download PixelVAR/HMAR memorization audits:

```bash
modal run modal_train.py --action audit-main-hmar-memorization --num-samples 4096
mkdir -p reports/memorization_audit/pixelvar_main reports/memorization_audit/hmar_steps1
modal volume get pixelvar-outputs /memorization_audit/pixelvar_main/audit_report.md reports/memorization_audit/pixelvar_main/audit_report.md --force
modal volume get pixelvar-outputs /memorization_audit/pixelvar_main/audit_summary.json reports/memorization_audit/pixelvar_main/audit_summary.json --force
modal volume get pixelvar-outputs /memorization_audit/pixelvar_main/exact_matches.csv reports/memorization_audit/pixelvar_main/exact_matches.csv --force
modal volume get pixelvar-outputs /memorization_audit/pixelvar_main/nearest_neighbors.csv reports/memorization_audit/pixelvar_main/nearest_neighbors.csv --force
modal volume get pixelvar-outputs /memorization_audit/pixelvar_main/nearest_pairs.png reports/memorization_audit/pixelvar_main/nearest_pairs.png --force
modal volume get pixelvar-outputs /memorization_audit/hmar_steps1/audit_report.md reports/memorization_audit/hmar_steps1/audit_report.md --force
modal volume get pixelvar-outputs /memorization_audit/hmar_steps1/audit_summary.json reports/memorization_audit/hmar_steps1/audit_summary.json --force
modal volume get pixelvar-outputs /memorization_audit/hmar_steps1/exact_matches.csv reports/memorization_audit/hmar_steps1/exact_matches.csv --force
modal volume get pixelvar-outputs /memorization_audit/hmar_steps1/nearest_neighbors.csv reports/memorization_audit/hmar_steps1/nearest_neighbors.csv --force
modal volume get pixelvar-outputs /memorization_audit/hmar_steps1/nearest_pairs.png reports/memorization_audit/hmar_steps1/nearest_pairs.png --force
```

Build and download the four-way visual sample sheet:

```bash
modal run modal_train.py --action build-four-way-sample-sheet
modal volume get pixelvar-outputs /final/four_way_sample_sheet.png reports/final/four_way_sample_sheet.png --force
```

On Modal, run the first known-metrics benchmark for main VAR vs the best HMAR
setting:

```bash
modal run modal_train.py --action benchmark-main-hmar-known-metrics --num-samples 4096
```

Download the result:

```bash
mkdir -p reports/external_eval/main_vs_hmar_known_metrics
modal volume get pixelvar-outputs /external_eval/main_vs_hmar_known_metrics/evaluation_report.md reports/external_eval/main_vs_hmar_known_metrics/evaluation_report.md --force
modal volume get pixelvar-outputs /external_eval/main_vs_hmar_known_metrics/metrics.csv reports/external_eval/main_vs_hmar_known_metrics/metrics.csv --force
modal volume get pixelvar-outputs /eval_images reports/eval_images --force
```
