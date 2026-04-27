# PixelVAR Option A Runbook

This milestone trains the deterministic v0 pipeline only: transparent palette tokens plus scale-wise VAR prediction. VQ-VAE, HMAR, multi-dataset training, and baselines are deferred.

## Setup

```bash
pip install -e .
python - <<'PY'
import torch, torchvision, lightning, pixelvar
print(torch.__version__, torchvision.__version__, lightning.__version__)
PY
```

If `torchvision` emits binary compatibility warnings, install a version that matches the installed Torch release before training.

## Data

```bash
python scripts/download_data.py --dataset pokemon
python scripts/preprocess_data.py --dataset pokemon --palette-size 16
python scripts/check_data.py --dataset pokemon
```

Required checks before training:

- `index_maps.npy` has shape `(N, 32, 32)` and token range `[0, 16]`.
- `alpha_masks.npy` exists and has a nonzero, non-full transparency ratio.
- `splits.json` and `manifest.json` have no Pokemon ID leakage across train/val/test.
- `outputs/data_checks/pokemon/sample_grid.png` renders transparent sprites correctly.

## Training Ladder

Run these in order and stop at the first failing stage:

```bash
python scripts/train_var.py --config configs/train/overfit32.yaml
python scripts/train_var.py --config configs/train/debug1k.yaml
python scripts/train_var.py --config configs/train/v0_full.yaml
```

`overfit32` must reach near-zero train loss or at least 98% token accuracy before `debug1k`. If it does not, inspect tokenizer offsets, label ranges, optimizer LR, and gradient finiteness before scaling up.

Outputs are written to `outputs/runs/{run_name}` and checkpoints to `checkpoints/{run_name}`. Resume a run with:

```bash
python scripts/train_var.py --config configs/train/debug1k.yaml --resume
```

## Sampling

```bash
python scripts/sample_var.py \
  --checkpoint checkpoints/var_pokemon_debug1k/best.ckpt \
  --config configs/train/debug1k.yaml \
  --num-samples 16 \
  --temperature 1.0 \
  --top-k 8 \
  --output outputs/samples/debug1k_grid.png
```

If loss decreases but samples are incoherent, first compare per-scale losses and try lower temperature or `top_k=8`; do not change architecture until overfit samples and scale metrics are understood.

## Lightning Studio

Use the same commands in Lightning Studio. Upload or clone the repo, install with `pip install -e .`, keep data under `data/raw` and `data/processed`, and use the default Trainer settings in the YAML configs. CSV logging is the default and does not require credentials; W&B can be enabled by setting `logger.name: wandb` and `WANDB_API_KEY`.
