# Proposal Leftovers / Stretch Plan

Decision date: 2026-06-08

## Scope

This pass turns the remaining small proposal/stretched ablations into runnable
experiments. It does not claim results yet.

The core proposal result is already covered by the main PixelVAR/HMAR/external
baseline reports. The remaining useful stretch work is:

1. Codebook-size ablation: compare 8, 16, and 32 palette colors.
2. Scale-count ablation: compare the main 6-scale pyramid against a smaller
   4-scale pyramid.

These are safer and more directly proposal-aligned than starting a new large
64x64 training run.

## What Was Added

### 8-Color Palette Ablation

- Processed dataset: `data/processed/sprites_palette8`
- Train configs:
  - `configs/train/sprites_palette8_overfit32.yaml`
  - `configs/train/sprites_palette8_debug1k.yaml`
  - `configs/train/sprites_palette8_v0_full.yaml`
- Expected checkpoint:
  - `checkpoints/var_sprites_palette8_v0_full/best.ckpt`
- Expected evaluation:
  - `outputs/eval/sprites_palette8_v0_full`

Purpose: test whether a tighter palette improves crispness and fidelity or
throws away too much visual variation.

### 32-Color Palette Ablation

- Processed dataset: `data/processed/sprites_palette32`
- Train configs:
  - `configs/train/sprites_palette32_overfit32.yaml`
  - `configs/train/sprites_palette32_debug1k.yaml`
  - `configs/train/sprites_palette32_v0_full.yaml`
- Expected checkpoint:
  - `checkpoints/var_sprites_palette32_v0_full/best.ckpt`
- Expected evaluation:
  - `outputs/eval/sprites_palette32_v0_full`

Purpose: test whether a larger palette improves visual detail or makes the token
distribution harder to model.

### 4-Scale Pyramid Ablation

- Uses the existing 16-color processed dataset: `data/processed/sprites`
- Scale schedule: `[1, 4, 16, 32]`
- Train configs:
  - `configs/train/sprites_scale4_overfit32.yaml`
  - `configs/train/sprites_scale4_debug1k.yaml`
  - `configs/train/sprites_scale4_v0_full.yaml`
- Expected checkpoint:
  - `checkpoints/var_sprites_scale4_v0_full/best.ckpt`
- Expected evaluation:
  - `outputs/eval/sprites_scale4_v0_full`

Purpose: test whether the full `[1, 2, 4, 8, 16, 32]` hierarchy is actually
needed, or whether fewer intermediate scales are enough.

## Modal Commands

Run these after pulling the branch.

```powershell
modal run modal_train.py --action prepare-sprites-palette8
modal run modal_train.py --action train-sprites-palette8-ladder
modal run modal_train.py --action eval-sprites-palette8 --num-samples 128
```

```powershell
modal run modal_train.py --action prepare-sprites-palette32
modal run modal_train.py --action train-sprites-palette32-ladder
modal run modal_train.py --action eval-sprites-palette32 --num-samples 128
```

```powershell
modal run modal_train.py --action train-sprites-scale4-ladder
modal run modal_train.py --action eval-sprites-scale4 --num-samples 128
```

## Download Results

```powershell
New-Item -ItemType Directory -Force -Path reports\eval\sprites_palette8_v0_full | Out-Null
modal volume get pixelvar-outputs /eval/sprites_palette8_v0_full/evaluation_report.md reports/eval/sprites_palette8_v0_full/evaluation_report.md --force
modal volume get pixelvar-outputs /eval/sprites_palette8_v0_full/metrics.csv reports/eval/sprites_palette8_v0_full/metrics.csv --force
modal volume get pixelvar-outputs /eval/sprites_palette8_v0_full/temp_0.8_topk_8_grid.png reports/eval/sprites_palette8_v0_full/temp_0.8_topk_8_grid.png --force
```

```powershell
New-Item -ItemType Directory -Force -Path reports\eval\sprites_palette32_v0_full | Out-Null
modal volume get pixelvar-outputs /eval/sprites_palette32_v0_full/evaluation_report.md reports/eval/sprites_palette32_v0_full/evaluation_report.md --force
modal volume get pixelvar-outputs /eval/sprites_palette32_v0_full/metrics.csv reports/eval/sprites_palette32_v0_full/metrics.csv --force
modal volume get pixelvar-outputs /eval/sprites_palette32_v0_full/temp_0.8_topk_8_grid.png reports/eval/sprites_palette32_v0_full/temp_0.8_topk_8_grid.png --force
```

```powershell
New-Item -ItemType Directory -Force -Path reports\eval\sprites_scale4_v0_full | Out-Null
modal volume get pixelvar-outputs /eval/sprites_scale4_v0_full/evaluation_report.md reports/eval/sprites_scale4_v0_full/evaluation_report.md --force
modal volume get pixelvar-outputs /eval/sprites_scale4_v0_full/metrics.csv reports/eval/sprites_scale4_v0_full/metrics.csv --force
modal volume get pixelvar-outputs /eval/sprites_scale4_v0_full/temp_0.8_topk_8_grid.png reports/eval/sprites_scale4_v0_full/temp_0.8_topk_8_grid.png --force
```

## Interpretation Rules

- Do not compare the 8/32-color runs using palette consistency alone. Palette
  consistency is expected to be `1.0000` for all deterministic palette-token
  models.
- Compare feature FID, opaque ratio, edge density, token entropy, and sample
  sheets.
- If a palette ablation wins numerically but looks visibly worse, keep it as an
  ablation rather than promoting it.
- The 4-scale ablation is only fair against the same 16-color dataset and same
  evaluator. Its purpose is to test hierarchy depth, not palette quality.

## Expected Outcome

The main 16-color / 6-scale PixelVAR remains the baseline until these runs are
actually completed and inspected. If none of the stretch runs clearly beats it,
the final claim should say that the proposal leftovers were made runnable and
tested, but the original 16-color / 6-scale setup stayed strongest.
