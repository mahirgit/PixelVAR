# Modal B200 Run Notes

Use Modal from your local terminal. The repo code stays local, while Modal runs the selected function remotely.

## One-Time Setup

```bash
python -m pip install modal
modal setup
```

The launcher lazily creates these volumes if they do not already exist:

```bash
modal volume create pixelvar-data
modal volume create pixelvar-outputs
modal volume create pixelvar-checkpoints
```

## Quick Check

```bash
modal run modal_train.py --action cuda-check
```

This allocates one `B200` and prints CUDA/Torch/device information.

## Pokemon

```bash
modal run modal_train.py --action prepare-pokemon
modal run modal_train.py --action train-overfit32
modal run modal_train.py --action train-debug1k
```

Only run full training after the smaller stages look sane:

```bash
modal run modal_train.py --action train-v0-full
```

## Sprites

The original `brentspell/sprites-dataset` Kaggle page is no longer accessible.
`prepare-sprites` uses the live `TalBarami/msd_sprites` replacement from
Hugging Face and writes it to `data/curated/sprites`.

```bash
modal run modal_train.py --action prepare-sprites
modal run modal_train.py --action train-sprites-overfit32
modal run modal_train.py --action train-sprites-debug1k
modal run modal_train.py --action train-sprites-v0-full
modal run modal_train.py --action eval-sprites
modal run modal_train.py --action generate-sprites-selected
modal run modal_train.py --action generate-sprites-selected --num-samples 170000
modal run modal_train.py --action prepare-sprites-generated-keep --num-samples 170000
modal run modal_train.py --action train-sprites-generated-keep-overfit32
modal run modal_train.py --action train-sprites-generated-keep-debug1k
modal run modal_train.py --action train-sprites-generated-keep-v0-full
modal run modal_train.py --action prepare-sprites-mixed
modal run modal_train.py --action train-sprites-mixed-overfit32
modal run modal_train.py --action train-sprites-mixed-debug1k
modal run modal_train.py --action train-sprites-mixed-v0-full
```

## Proposal Leftovers / Stretch Ablations

These are optional proposal-cleanup runs after the main 16-color, 6-scale
PixelVAR result is in place.

8-color palette ablation:

```bash
modal run modal_train.py --action prepare-sprites-palette8
modal run modal_train.py --action train-sprites-palette8-ladder
modal run modal_train.py --action eval-sprites-palette8 --num-samples 128
```

32-color palette ablation:

```bash
modal run modal_train.py --action prepare-sprites-palette32
modal run modal_train.py --action train-sprites-palette32-ladder
modal run modal_train.py --action eval-sprites-palette32 --num-samples 128
```

4-scale hierarchy ablation:

```bash
modal run modal_train.py --action train-sprites-scale4-ladder
modal run modal_train.py --action eval-sprites-scale4 --num-samples 128
```

The detailed interpretation plan is in
`reports/final/proposal_leftovers_stretch_plan.md`.

For a manually supplied Sprites archive/folder instead, upload it into the data
volume and use the raw curation action:

```bash
modal volume put pixelvar-data /path/to/sprites.zip /raw/sprites/sprites.zip
modal run modal_train.py --action prepare-raw-sprites --transparent-color "#000000"
```

## OpenGameArt

Prepare the curated public OpenGameArt stretch set automatically:

```bash
modal run modal_train.py --action prepare-opengameart-public
```

This downloads the pinned public OpenGameArt assets, writes source/license
metadata, curates individual frames, reuses the Sprites palette, and validates
`data/processed/opengameart`.

Manual upload is still supported for extra sheets/images:

```bash
modal volume put pixelvar-data /path/to/opengameart /raw/opengameart
modal run modal_train.py --action prepare-opengameart
```

For 64x64 sheets:

```bash
modal run modal_train.py --action prepare-opengameart --sheet-tile-size 64
```

To mix OpenGameArt with the current real + generated-keep dataset:

```bash
modal run modal_train.py --action prepare-sprites-mixed-opengameart
modal run modal_train.py --action train-sprites-mixed-oga-overfit32
modal run modal_train.py --action train-sprites-mixed-oga-debug1k
modal run modal_train.py --action train-sprites-mixed-oga-v0-full
```

If the full run is interrupted after checkpoints have been saved:

```bash
modal run modal_train.py --action train-sprites-mixed-oga-v0-full --resume
```

Sample and evaluate the OpenGameArt-mixed checkpoint against the real Sprites
validation set:

```bash
modal run modal_train.py --action sample \
  --config configs/train/sprites_mixed_oga_v0_full.yaml \
  --checkpoint checkpoints/var_sprites_mixed_oga_v0_full/best.ckpt \
  --num-samples 64 \
  --temperature 0.8 \
  --top-k 8 \
  --output outputs/samples/sprites_mixed_oga_v0_full_t08_top8.png
modal run modal_train.py --action eval-sprites \
  --config configs/train/sprites_v0_full.yaml \
  --checkpoint checkpoints/var_sprites_mixed_oga_v0_full/best.ckpt \
  --output outputs/eval/sprites_mixed_oga_v0_full_realval \
  --num-samples 128
```

## Custom Commands

CPU command:

```bash
modal run modal_train.py --action cmd-cpu --cmd "python scripts/check_data.py --dataset pokemon"
```

B200 command:

```bash
modal run modal_train.py --action cmd-gpu --cmd "python scripts/train_var.py --config configs/train/debug1k.yaml"
```

## Download Results

```bash
modal volume get pixelvar-outputs /runs ./modal_outputs
mkdir -p ./modal_outputs/eval
modal volume get pixelvar-outputs eval/sprites_v0_full ./modal_outputs/eval --force
mkdir -p ./modal_outputs/generated
modal volume get pixelvar-outputs generated/sprites_v0_full_t08_top8_8192 ./modal_outputs/generated --force
modal volume get pixelvar-outputs generated/sprites_v0_full_t08_top8_8192.zip ./modal_outputs/generated/sprites_v0_full_t08_top8_8192.zip --force
modal volume get pixelvar-outputs generated/sprites_v0_full_t08_top8_170000.zip ./modal_outputs/generated/sprites_v0_full_t08_top8_170000.zip --force
mkdir -p ./data/processed/sprites
modal volume get pixelvar-data processed/sprites/palette.json ./data/processed/sprites/palette.json --force
modal volume get pixelvar-checkpoints /var_pokemon_debug1k ./modal_checkpoints/var_pokemon_debug1k
```

## Inspect Generated Set

```bash
python scripts/inspect_generated_set.py \
  --generated-dir modal_outputs/generated/sprites_v0_full_t08_top8_8192 \
  --output-dir reports/generated/sprites_v0_full_t08_top8_8192_inspection \
  --palette data/processed/sprites/palette.json \
  --reference-summary reports/eval/sprites_v0_full/reference_summary.json \
  --write-keep-package
```

For the 170k zip package, extract it first:

```bash
mkdir -p modal_outputs/generated/sprites_v0_full_t08_top8_170000
python -c "import zipfile; zipfile.ZipFile('modal_outputs/generated/sprites_v0_full_t08_top8_170000.zip').extractall('modal_outputs/generated/sprites_v0_full_t08_top8_170000')"
python scripts/inspect_generated_set.py \
  --generated-dir modal_outputs/generated/sprites_v0_full_t08_top8_170000 \
  --output-dir reports/generated/sprites_v0_full_t08_top8_170000_inspection \
  --palette data/processed/sprites/palette.json \
  --reference-summary reports/eval/sprites_v0_full/reference_summary.json \
  --write-keep-package
```

Sample and evaluate the generated-keep full checkpoint:

```bash
modal run modal_train.py --action sample \
  --config configs/train/sprites_generated_keep_v0_full.yaml \
  --checkpoint checkpoints/var_sprites_generated_keep_v0_full/best.ckpt \
  --num-samples 64 \
  --temperature 0.8 \
  --top-k 8 \
  --output outputs/samples/sprites_generated_keep_v0_full_t08_top8.png
modal run modal_train.py --action eval-sprites \
  --config configs/train/sprites_generated_keep_v0_full.yaml \
  --checkpoint checkpoints/var_sprites_generated_keep_v0_full/best.ckpt \
  --output outputs/eval/sprites_generated_keep_v0_full \
  --num-samples 128
```

## Mixed Real + Generated

```bash
modal run modal_train.py --action prepare-sprites-mixed
modal run modal_train.py --action train-sprites-mixed-overfit32
modal run modal_train.py --action train-sprites-mixed-debug1k
modal run modal_train.py --action train-sprites-mixed-v0-full
```

If the full run is interrupted after a checkpoint has been saved:

```bash
modal run modal_train.py --action train-sprites-mixed-v0-full --resume
```

Sample and evaluate the mixed checkpoint:

```bash
modal run modal_train.py --action sample \
  --config configs/train/sprites_mixed_v0_full.yaml \
  --checkpoint checkpoints/var_sprites_mixed_v0_full/best.ckpt \
  --num-samples 64 \
  --temperature 0.8 \
  --top-k 8 \
  --output outputs/samples/sprites_mixed_v0_full_t08_top8.png
modal run modal_train.py --action eval-sprites \
  --config configs/train/sprites_v0_full.yaml \
  --checkpoint checkpoints/var_sprites_mixed_v0_full/best.ckpt \
  --output outputs/eval/sprites_mixed_v0_full_realval \
  --num-samples 128
```

## Learned Tokenizer / Patch-VQ

The neural VQ-VAE path is implemented, but the first reconstruction gates were
too soft for pixel art. The current usable learned-token branch is patch-VQ:
it learns a 512-code `2x2` RGBA patch codebook, exports `16x16` code maps, then
trains VAR over `[1, 2, 4, 8, 16]` learned-token pyramids.

Prepare the patch-VQ token dataset:

```bash
modal run modal_train.py --action prepare-patch-vq-sprites
```

Train the VAR ladder on patch-VQ tokens:

```bash
modal run modal_train.py --action train-sprites-patchvq16-overfit32
modal run modal_train.py --action train-sprites-patchvq16-debug1k
modal run modal_train.py --action train-sprites-patchvq16-v0-full
```

Sample decoded patch-VQ VAR sprites:

```bash
modal run modal_train.py --action sample-patch-vq-var \
  --num-samples 64 \
  --temperature 1.0 \
  --top-k 16
```

Evaluate decoded patch-VQ samples against real validation sprites:

```bash
modal run modal_train.py --action eval-patch-vq-decoded --num-samples 128
```

Current decoded sweep finding: `temperature=1.0`, `top_k=16` is the best tested
patch-VQ setting. The earlier `temperature=0.8`, `top_k=32` grid was coherent
but scored lower in decoded image-space.

## HMAR Masked Refinement

HMAR is the proposal's Option B ablation: keep the same coarse-to-fine scale
order, but replace single-pass intra-scale prediction with iterative masked
refinement. Token `17` is the reserved mask token; generated outputs still use
tokens `0..16`.

Run the HMAR ladder:

```bash
modal run modal_train.py --action train-sprites-hmar-overfit32
modal run modal_train.py --action train-sprites-hmar-debug1k
modal run modal_train.py --action train-sprites-hmar-v0-full
```

Sample HMAR:

```bash
modal run modal_train.py --action sample-hmar \
  --num-samples 64 \
  --temperature 0.8 \
  --top-k 8 \
  --refinement-steps 1
```

Evaluate HMAR against the same real-validation sprite metric:

```bash
modal run modal_train.py --action eval-hmar-sprites --num-samples 128 --refinement-steps 1
```

Run the proposal refinement-step ablation:

```bash
modal run modal_train.py --action eval-hmar-refinement-ablation --num-samples 128
```

Current HMAR finding: the full run early-stopped at epoch 13 with
`val_loss=0.02025`, `val_acc=0.99216`. The refinement-step ablation prefers
`refinement_steps=1`, `temperature=0.8`, `top_k=8`, with feature score
`0.00189`. Extra refinement did not help here: `4` steps scored `0.00366`, and
`8` steps scored `0.00822`. The current real-only VAR baseline is still better
at `0.00147`.

## Final Artifacts

The consolidated decision table and final sample sheets are local report
artifacts:

- `reports/final/final_report_narrative.md`
- `reports/final/reproducibility_commands.md`
- `reports/final/external_baseline_and_metrics_plan.md`
- `reports/final/known_metrics_comparison.md`
- `reports/final/memorization_audit_summary.md`
- `reports/final/four_way_sample_sheet.png`
- `reports/memorization_audit/flat_ar/audit_report.md`
- `reports/final/model_decision_table.md`
- `reports/final/model_decision_table.csv`
- `reports/final/final_branch_comparison_sheet.png`
- `reports/final/final_main_var_sample_sheet.png`
- `reports/final/final_hmar_sample_sheet.png`
- `reports/final/final_patchvq_sample_sheet.png`

Known-metrics benchmark:

```bash
modal run modal_train.py --action benchmark-main-hmar-known-metrics --num-samples 4096
mkdir -p reports/external_eval/main_vs_hmar_known_metrics
modal volume get pixelvar-outputs /external_eval/main_vs_hmar_known_metrics/evaluation_report.md reports/external_eval/main_vs_hmar_known_metrics/evaluation_report.md --force
modal volume get pixelvar-outputs /external_eval/main_vs_hmar_known_metrics/metrics.csv reports/external_eval/main_vs_hmar_known_metrics/metrics.csv --force
```

Flat baseline benchmark:

```bash
modal run modal_train.py --action train-flat-ar-ladder
modal run modal_train.py --action train-flat-maskgit-ladder
modal run modal_train.py --action benchmark-main-hmar-flat-known-metrics --num-samples 4096
mkdir -p reports/external_eval/main_hmar_flat_known_metrics
modal volume get pixelvar-outputs /external_eval/main_hmar_flat_known_metrics/evaluation_report.md reports/external_eval/main_hmar_flat_known_metrics/evaluation_report.md --force
modal volume get pixelvar-outputs /external_eval/main_hmar_flat_known_metrics/metrics.csv reports/external_eval/main_hmar_flat_known_metrics/metrics.csv --force
```

Flat AR memorization audit:

```bash
modal run modal_train.py --action audit-flat-ar-memorization --num-samples 4096
mkdir -p reports/memorization_audit/flat_ar
modal volume get pixelvar-outputs /memorization_audit/flat_ar/audit_report.md reports/memorization_audit/flat_ar/audit_report.md --force
modal volume get pixelvar-outputs /memorization_audit/flat_ar/audit_summary.json reports/memorization_audit/flat_ar/audit_summary.json --force
modal volume get pixelvar-outputs /memorization_audit/flat_ar/exact_matches.csv reports/memorization_audit/flat_ar/exact_matches.csv --force
modal volume get pixelvar-outputs /memorization_audit/flat_ar/nearest_neighbors.csv reports/memorization_audit/flat_ar/nearest_neighbors.csv --force
modal volume get pixelvar-outputs /memorization_audit/flat_ar/nearest_pairs.png reports/memorization_audit/flat_ar/nearest_pairs.png --force
```

PixelVAR/HMAR memorization audit:

```bash
modal run modal_train.py --action audit-main-hmar-memorization --num-samples 4096
```

Four-way sample sheet:

```bash
modal run modal_train.py --action build-four-way-sample-sheet
modal volume get pixelvar-outputs /final/four_way_sample_sheet.png reports/final/four_way_sample_sheet.png --force
```

## SD-piXL External Baseline

SD-piXL uses a separate Modal image because it needs diffusion and
score-distillation dependencies that are not needed for PixelVAR training.

Prepare the external repo clone and PixelVAR palette:

```bash
modal run modal_train.py --action prepare-sd-pixl-baseline
```

Run the first smoke sample on B200:

```bash
modal run modal_train.py --action run-sd-pixl-smoke --sd-pixl-steps 250
```

Download the normalized sample folder and visual sheet:

```bash
mkdir -p outputs/external_baselines/sd_pixl reports/final
modal volume get pixelvar-outputs /external_baselines/sd_pixl/png32 outputs/external_baselines/sd_pixl/png32 --force
modal volume get pixelvar-outputs /final/sd_pixl_sample_sheet.png reports/final/sd_pixl_sample_sheet.png --force
```

Optional small batch:

```bash
modal run modal_train.py --action run-sd-pixl-batch --num-samples 4 --sd-pixl-steps 1000
modal run modal_train.py --action normalize-sd-pixl-baseline
modal run modal_train.py --action build-sd-pixl-sample-sheet
```

The one-image smoke run is not a metric result. Only evaluate SD-piXL after a
larger normalized folder exists under `outputs/external_baselines/sd_pixl/png32`.

The neural VQ-VAE commands remain available for further iteration:

```bash
modal run modal_train.py --action train-vqvae-sprites-overfit32
modal run modal_train.py --action train-vqvae-sprites-debug1k
modal run modal_train.py --action train-vqvae-sprites-v0-full
modal run modal_train.py --action export-vqvae-sprites
modal run modal_train.py --action train-sprites-vqvae16-v0-full
modal run modal_train.py --action sample-vq-var
```
