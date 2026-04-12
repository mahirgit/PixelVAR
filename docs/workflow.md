# PixelVAR — Repo Workflow & Conventions

## Branch Strategy

### Main branches
- **`main`** — Protected. Always working. Must pass smoke test. No direct pushes.
- **`dev`** — Integration branch. PRs from feature branches merge here first. Merge to `main` when stable.

### Feature branches
Naming convention: `{scope}/{short-description}`

Scopes:
- `data/` — preprocessing, palette, splits, datasets
- `tokenizer/` — deterministic pyramid, VQ-VAE, tokenizer interface
- `model/` — AR transformer, attention, sampling
- `eval/` — metrics, baselines, user study
- `infra/` — repo setup, CI, configs, tooling
- `viz/` — visualization, sample grids, reports
- `fix/` — bug fixes
- `exp/` — experimental branches (throwaway, not expected to merge)

**Examples:**
```
infra/repo-hardening
data/transparency-fix
data/pokemon-splits
tokenizer/deterministic-baseline
model/var-transformer-skeleton
eval/fid-pipeline
fix/pokemon-dir-bug
exp/vqvae-codebook-128
```

### Day 1 branches
```
infra/repo-hardening          (Person A)
data/transparency-splits      (Person B)
tokenizer/deterministic       (Person C)
```

---

## Issue Tracking

### Where: GitHub Issues

Use GitHub Issues — not a separate tool. Keep it simple.

### Labels

Create these labels on Day 1:

| Label | Color | Description |
|-------|-------|-------------|
| `P0-critical` | red | Blocks other work |
| `P1-important` | orange | Should do this week |
| `P2-later` | yellow | Backlog |
| `data` | blue | Data/preprocessing |
| `tokenizer` | purple | Tokenizer work |
| `model` | green | AR model / generation |
| `eval` | teal | Metrics / evaluation |
| `infra` | gray | Repo / tooling |
| `bug` | red | Something broken |
| `decision` | pink | Needs team discussion |

### Issue format (keep it short)

```markdown
**What:** One sentence.
**Why:** One sentence.
**Done when:** Concrete test/artifact.
**Blocked by:** #issue or "nothing"
```

### Day 1 issues to create

```
#1  [decision] Lock token semantics                          P0-critical
#2  [decision] Lock palette regime (16 vs 32)                P0-critical
#3  [decision] Lock transparency handling                    P0-critical
#4  [data] Fix preprocessing to preserve alpha               P0-critical
#5  [data] Add character-level train/val/test splits         P0-critical
#6  [infra] Add pyproject.toml, remove setup.py              P1-important
#7  [infra] Expand .gitignore, remove tracked artifacts      P1-important
#8  [infra] Add README.md                                    P1-important
#9  [tokenizer] Formalize deterministic pyramid tokenizer    P1-important
#10 [infra] Add smoke test script                            P1-important
#11 [data] Fix pokemon_dir variable bug                      bug
#12 [infra] Add tests/ directory with basic tests            P1-important
#13 [data] Build data manifest (pokemon)                     P1-important
#14 [data] Download & prepare Sprites dataset (170K)         P2-later
#15 [model] VAR transformer skeleton                         P2-later
#16 [eval] Implement FID pipeline                            P2-later
#17 [tokenizer] VQ-VAE tokenizer experiment                  P2-later
```

---

## PR Process

### Rules
1. **Every PR has one owner.** The owner writes the code.
2. **Every PR gets one review.** Any of the other two reviews it. Quick — 10 min max.
3. **PRs should be small.** One concern per PR. If it touches 10+ files, split it.
4. **Every PR must have a visible output.** A test passing, a sample grid, a shape printout — something concrete.
5. **Merge to `dev` freely.** Merge `dev` → `main` only when smoke test passes.

### PR title format
```
[scope] Short description
```
Examples:
```
[data] Fix transparency handling, add alpha mask preservation
[tokenizer] Extract deterministic pyramid into proper module
[infra] Add pyproject.toml, Makefile, basic CI
[model] VAR transformer forward pass skeleton
```

### PR description template (copy-paste, don't overthink it)
```markdown
## What
One paragraph.

## Changes
- file1: what changed
- file2: what changed

## Test
How to verify this works. Command to run or screenshot.
```

---

## Commit Messages

Format: `[scope] imperative description`

```
[data] preserve alpha channel in preprocessing
[tokenizer] extract mode-pooling into deterministic tokenizer
[infra] add pyproject.toml, remove setup.py
[model] implement block-causal attention mask
[fix] correct pokemon_dir variable reference
[eval] add palette consistency score metric
```

Keep them short. One line unless you need to explain *why*.

---

## Experiment Tracking

### Week 1: Simple
- Log configs and results in `reports/` as markdown
- Save sample grids as PNGs
- Track key numbers in a shared spreadsheet or doc

### Week 2+: W&B
- Set up Weights & Biases project `pixelvar`
- Every training run logs: config, seed, loss curves, sample grids, codebook stats
- Run naming: `{model}_{dataset}_{key-param}_{date}` (e.g., `var_pokemon_d256_0420`)

---

## Config Management

### Week 1: YAML files
```
configs/
  data/
    pokemon.yaml
    sprites.yaml
  tokenizer/
    deterministic.yaml
    vqvae.yaml        (later)
  model/
    var_small.yaml    (later)
  train/
    overfit32.yaml    (later)
    debug1k.yaml      (later)
```

### Week 2+: Consider Hydra
Only if YAML files become unwieldy. Don't add complexity before it's needed.

---

## Training Ladder

Every new model component must pass these stages in order:

| Stage | Dataset | Size | Purpose | Pass condition |
|-------|---------|------|---------|----------------|
| `overfit32` | 32 hand-picked sprites | 32 | Memorization test | Perfect reconstruction / near-zero loss |
| `debug1k` | First 1K from train split | 1,000 | Sanity check | Loss decreasing, recognizable outputs |
| `v0_full` | Full train split | ~2,500+ | Real training | FID / visual quality competitive |

**Hard rule:** If a model can't overfit 32 sprites, don't waste GPU on larger runs.

---

## Agent Context

This section helps AI agents (Claude, etc.) understand where we are.

### Project phase tracking
```
Phase 0: Repo setup, data, decisions       ← WE ARE HERE (Day 1)
Phase 1: Deterministic pipeline end-to-end  (Days 2-5)
Phase 2: AR transformer on deterministic tokens (Days 6-10)  
Phase 3: VQ-VAE tokenizer experiments       (Week 3)
Phase 4: Full VAR training + baselines      (Weeks 4-5)
Phase 5: HMAR / ablations / evaluation      (Weeks 6-8)
Phase 6: Final report                       (Week 9-10)
```

### Key files an agent should read first
```
docs/day1_checkpoint.md     — Current decisions and status
docs/workflow.md            — This file (repo conventions)
docs/design_contract.md     — Token specs, tensor shapes, interfaces (created Day 1)
configs/                    — Current experiment configs
pixelvar/                   — All model/data code
scripts/                    — Runnable scripts
```

### What an agent should NOT do without asking
- Change token vocabulary or special token assignments
- Modify the scale schedule [1, 2, 4, 8, 16, 32]
- Switch palette regime (global vs per-image)
- Add new dependencies
- Restructure the repo layout
- Modify data splits after they're frozen

### What an agent CAN do freely
- Implement model code following the design contract
- Write tests
- Create visualization scripts
- Fix bugs
- Optimize existing code for speed
- Draft reports / docs
