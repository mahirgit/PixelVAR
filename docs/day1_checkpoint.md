# Day 1 Checkpoint — April 12, 2026

## Status: IN PROGRESS

---

## Key Decisions to Lock Today

These are the 7 decisions that must be frozen before any model code is written.
Decisions marked `[PENDING]` need team agreement. Marked `[RECOMMENDED]` is ChatGPT/Claude consensus.

### 1. Token Semantics `[PENDING]`

**Question:** What does a "token" mean in our pipeline?

**Current state:** `dataset.py` already builds 1,365-token sequences from palette indices via mode-pooling. Tokens = palette color indices (0–15).

**Options:**
- **(A) Palette indices as tokens (deterministic)** — What the repo already does. No learned tokenizer needed. The AR model directly predicts palette color indices. Simple, fast, already working.
- **(B) Learned VQ-VAE codes** — The proposal's Stage 1. Train a VQ-VAE where codebook entries are *learned* latent codes (not literal colors). Codebook size matches palette size. Adds reconstruction quality but adds a training stage.

**[RECOMMENDED]:** Start with **(A)** as the v0 contract. All downstream code consumes discrete index tensors. If we later train a VQ-VAE tokenizer, it plugs in behind the same interface. This avoids blocking the AR model on tokenizer quality.

**What this means in code:**
- `token_sequence: (B, 1365)` with values in `[0, V)` where `V = palette_size + special_tokens`
- `multi_scale_maps: list of (B, h_k, w_k)` tensors per scale
- Downstream code never sees RGB, only indices

---

### 2. Palette Regime `[PENDING]`

**Question:** How many colors? Global or per-image?

**Current state:** Global 16-color palette via k-means, extracted from Pokemon dataset.

**Options:**
- Global 16 (current default)
- Global 32 (ChatGPT recommends)
- Per-image (too complex for v0)

**[RECOMMENDED]:** Keep **global 16** as the starting point. It's standard for pixel art (SNES-era), already implemented, and keeps the AR vocabulary small. Run **global 32 as an ablation** later — it's a one-flag change. Do NOT do per-image palettes in v0.

**Rationale for disagreeing with ChatGPT's 32 recommendation:** 16 is a better default because (a) smaller vocabulary = easier AR modeling, (b) pixel art genuinely uses 8-16 colors per sprite, (c) 32 can be tested later trivially.

---

### 3. Transparency `[PENDING]`

**Question:** How do we handle transparent pixels?

**Current state:** `preprocess_data.py` composites RGBA onto white, then palette.py hacks around white via luminance > 245 filter. Transparency is destroyed.

**[RECOMMENDED]:** Fix this today. The plan:

1. During preprocessing, keep the alpha channel as a separate binary mask
2. Reserve **token 0 = transparent**
3. Palette indices become **1 to K** (where K = palette_size)
4. Total vocabulary v0: `K + 1` (transparent + palette colors)
5. Later for HMAR, add a **mask token** (separate from transparent)
6. Never composite onto white — transparent pixels stay transparent

**Vocab layout:**
```
Token 0: transparent
Token 1–16: palette colors (for K=16)
Token 17: [MASK] (reserved for Option B, unused in v0)
```

---

### 4. Week 1 Dataset `[PENDING]`

**Question:** What do we actually train on this week?

**Current state:** 3,301 Pokemon sprites (front/back/shiny/shiny_back). The 170K Sprites dataset requires Kaggle. OpenGameArt is manual.

**[RECOMMENDED]:**
- **Debug/smoke set:** Pokemon (already in repo, ~3.3K sprites)
- **Main training set (Week 2+):** Sprites dataset from Kaggle (~170K frames) — but needs proper asset-group splitting
- **Deferred:** OpenGameArt (manual curation, heterogeneous, add later)

For Week 1, Pokemon is enough to build and validate the full pipeline. Don't mix sources until the pipeline is proven.

---

### 5. Split Strategy `[PENDING]`

**Question:** How do we split train/val/test?

**Current state:** No splits exist. All 3,301 Pokemon sprites are in one array.

**[RECOMMENDED]:** Split by **Pokemon ID** (character-level), never by frame/variant.

Pokemon has 4 variants per character (front, back, shiny, shiny_back). If you random-split by frame, a Pokemon's front sprite leaks into train while its back sprite goes to val — inflated metrics.

**Concrete plan:**
- Parse filenames to extract `pokemon_id` (already structured as `{id}.png`, `{id}_back.png`, etc.)
- Split on unique Pokemon IDs: **80/10/10** (train/val/test)
- ~720 characters train, ~90 val, ~90 test (× 4 variants each)
- Save split as a JSON mapping: `{pokemon_id: "train" | "val" | "test"}`

For the 170K Sprites dataset later: split by `sheet_id` or `character_id`.

---

### 6. Scope: Mainline vs. Stretch `[PENDING]`

**Question:** What is the core deliverable vs. nice-to-have?

**[RECOMMENDED]:**

| Priority | Component | Description |
|----------|-----------|-------------|
| **Mainline** | Deterministic tokenizer + VAR AR | Palette-index pyramid + next-scale Transformer |
| **Stretch 1** | Learned VQ-VAE tokenizer | Replace deterministic pyramid with learned encoder |
| **Stretch 2** | HMAR masked refinement | Iterative intra-scale refinement (Option B) |
| **Stretch 3** | Multi-dataset training | Sprites + Pokemon + OpenGameArt combined |

**Success condition for the project:** The AR Transformer generates recognizable sprites from the deterministic token pyramid. Everything else is bonus.

---

### 7. Repo Source of Truth `[PENDING]`

**Question:** Notebooks or package code?

**[RECOMMENDED]:** Package code + scripts are the source of truth. Notebooks are for demos, audits, and visualization only. No training logic, no model definitions, no preprocessing logic should live only in a notebook.

---

## Current Repo Problems (to fix today)

### Critical
- [ ] **Transparency destroyed** — `ensure_rgb()` composites alpha onto white. Must preserve alpha mask and add transparent token.
- [ ] **No train/val/test splits** — Need character-level split logic.
- [ ] **Token semantics undocumented** — No design contract exists.

### Important
- [ ] **`pokemon_dir` bug** — Line 213 of `preprocess_data.py` references `pokemon_dir` but variable is `pokemon_base`.
- [ ] **Duplicate dependency definitions** — `requirements.txt` and `setup.py` list deps separately. Should use `pyproject.toml`.
- [ ] **`sys.path` hacks** — Scripts insert repo root into sys.path. Should use editable install.
- [ ] **Generated files tracked in git** — `visualizations/pokemon/`, `data/processed/pokemon/palette.json` are committed.
- [ ] **No README** — Repo has no landing page.
- [ ] **No tests** — No test directory or test files.
- [ ] **Deterministic tokenizer not formalized** — Mode-pooling logic lives inside `dataset.py` instead of a proper tokenizer module.

### Nice-to-have (not Day 1 blockers)
- [ ] CI workflow
- [ ] Pre-commit hooks
- [ ] Makefile
- [ ] W&B integration
- [ ] Hydra configs

---

## Day 1 Task Breakdown

### Phase 1: Team Sync (60–90 min, all 3 together)

Go through decisions 1–7 above. Write final answers. This is the most important part of Day 1.

### Phase 2: Parallel Work (3 PRs)

#### PR 1: `repo-hardening` (Person A — Infra)

- [ ] Create `README.md` (project overview, setup instructions, repo structure)
- [ ] Create `pyproject.toml` (single source for deps + project metadata)
- [ ] Remove `setup.py` and `requirements.txt` (after pyproject.toml works)
- [ ] Expand `.gitignore` (add: `visualizations/`, `outputs/`, `checkpoints/`, `wandb/`, `*.pt`, `*.pth`, `*.ckpt`, `reports/generated/`)
- [ ] Remove tracked generated files (`visualizations/pokemon/*`, `data/processed/pokemon/palette*`)
- [ ] Create `tests/` directory with `test_imports.py`
- [ ] Create `Makefile` with targets: `install`, `test`, `lint`, `smoke`
- [ ] (Optional) `.pre-commit-config.yaml`, `.github/workflows/ci.yml`
- [ ] Protect `main` branch

#### PR 2: `data-transparency-splits` (Person B — Data)

- [ ] Fix `preprocess_data.py` to preserve alpha as binary mask
- [ ] Update `palette.py` to skip transparent pixels (not white-filter hack)
- [ ] Add transparent token (index 0) to the vocabulary
- [ ] Save `alpha_masks.npy` alongside `index_maps.npy`
- [ ] Fix `pokemon_dir` → `pokemon_base` bug (line 213)
- [ ] Add character-level split logic (`scripts/make_splits.py`)
- [ ] Save split JSON with pokemon_id → train/val/test mapping
- [ ] Update `dataset.py` to accept split parameter
- [ ] Create `docs/design_contract.md` with token specs and tensor shapes

#### PR 3: `deterministic-tokenizer` (Person C — Tokenizer/Pipeline)

- [ ] Create `pixelvar/tokenizers/__init__.py`
- [ ] Create `pixelvar/tokenizers/base.py` (abstract interface)
- [ ] Create `pixelvar/tokenizers/deterministic.py` (extract mode-pooling from dataset.py)
- [ ] Update `dataset.py` to use the tokenizer interface
- [ ] Create `scripts/smoke.py` (load 8 sprites → preprocess → tokenize → print shapes → save grid)
- [ ] Create `tests/test_deterministic_pyramid.py`
- [ ] (Optional) `scripts/visualize_pyramid.py` for multi-scale gallery export

---

## End-of-Day 1 Success Criteria

- [ ] All 7 decisions documented with final answers
- [ ] Fresh clone → `pip install -e .` → `make smoke` works
- [ ] Transparency preserved in preprocessing (not composited onto white)
- [ ] Train/val/test split exists for Pokemon (by character ID)
- [ ] Deterministic tokenizer is a proper module, not embedded in dataset.py
- [ ] No generated files tracked in git
- [ ] `main` branch is protected
- [ ] Token vocabulary is defined: `{0: transparent, 1-K: palette, K+1: mask_reserved}`

---

## What NOT To Do Today

- Do NOT write transformer / AR model code
- Do NOT start VQ-VAE training
- Do NOT debate HMAR or masked refinement details
- Do NOT add OpenGameArt or the 170K Sprites dataset
- Do NOT set up W&B, Hydra, or complex experiment infrastructure
- Do NOT bikeshed the repo layout — fix what's broken, don't restructure everything
