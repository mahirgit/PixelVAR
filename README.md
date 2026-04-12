# PixelVAR

Coarse-to-fine pixel art sprite generation via next-scale prediction with palette-constrained tokenization.

Built for COMP547 Deep Unsupervised Learning, Koç University, Spring 2026.

## Overview

PixelVAR adapts the [VAR](https://arxiv.org/abs/2404.02905) (Visual AutoRegressive) framework to generate pixel art sprites. Instead of operating in continuous RGB space, it works entirely in discrete palette-index space — each pixel is one of 16 palette colors, represented as an integer token.

Generation proceeds coarse-to-fine: the model first predicts a 1×1 palette-index map, then 2×2, then 4×4, up to 32×32 — mirroring how pixel artists actually work.

## Setup

**Requirements:** Python 3.9+, pip

```bash
git clone https://github.com/mahirgit/PixelVAR.git
cd PixelVAR
pip install -e ".[dev]"
```

Verify the install:

```bash
make smoke
```

## Repo Structure

```
pixelvar/          # Main package
  data/            # Dataset, palette extraction, preprocessing
  models/          # AR Transformer, tokenizer
  tokenizers/      # Deterministic pyramid tokenizer (+ VQ-VAE later)
  utils/           # Shared utilities

scripts/           # Runnable scripts (preprocess, train, evaluate)
configs/           # YAML experiment configs
docs/              # Design decisions and workflow
tests/             # Unit and smoke tests
notebooks/         # Demo notebooks (read-only — no training logic here)
data/              # Raw data (gitignored) and processed splits
```

## Key Design Decisions

See [`docs/design_contract.md`](docs/design_contract.md) for the full spec. Quick summary:

- **Tokens** are palette color indices (0 = transparent, 1–16 = colors, 17 = \[MASK\])
- **Vocab size** is 17 for v0 (16 palette + 1 transparent)
- **Scale schedule**: 1×1 → 2×2 → 4×4 → 8×8 → 16×16 → 32×32 (1,365 tokens total)
- **Dataset** (Week 1): Pokémon sprites (~3,300 sprites), split by character ID

## Development Workflow

See [`docs/workflow.md`](docs/workflow.md) for branch naming, PR process, and commit format.

```bash
make install   # install package + dev deps
make test      # run pytest
make lint      # run ruff
make smoke     # quick end-to-end sanity check
```

## Team

- Eyüp Ahmet Başaran
- Mahir Tilkicioğlu
- Kübra Rengin Çetin

Department of Computer Engineering, Koç University.
