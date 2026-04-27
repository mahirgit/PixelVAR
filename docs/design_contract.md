# PixelVAR — Design Contract

This document is the source of truth for token semantics, tensor shapes, and interfaces.
**Do not change these decisions without team agreement.**

---

## Decision Log

### Decision 1: Token Semantics ✅ LOCKED

**Tokens are literal palette color indices.**

- v0 tokenizer = deterministic palette-index pyramid (mode-pooling)
- All downstream code (AR model, loss, sampling) consumes discrete index tensors
- Downstream code never sees RGB — only indices
- If a learned VQ-VAE tokenizer is added later, it plugs in behind the same interface (still outputs discrete indices)
- The VQ-VAE is an optional experiment, not a blocker for the AR model

### Decision 2: Palette Regime ✅ LOCKED

**Global 16-color palette for v0.**

- Extracted via k-means across the full dataset
- Per-image palettes are out of scope for v0
- Global 32 is a planned ablation (one-flag change: `--palette-size 32`)
- Palette is sorted by luminance for consistency

### Decision 3: Transparency ✅ LOCKED

**Transparency is a first-class token, not discarded.**

- Token 0 = transparent
- Tokens 1–16 = palette colors
- Token 17 = [MASK] (reserved for HMAR Option B, unused in v0)
- Total v0 vocab size: **17** (16 palette + 1 transparent)
- During preprocessing: pixels with alpha < 128 → token 0, rest → quantized to palette (1–16)
- `alpha_masks.npy` saved alongside `index_maps.npy`
- Never composite onto white — transparent pixels stay transparent

> **Implementation status:** Implemented in the preprocessing and dataset code.
> Any old processed arrays must be regenerated because legacy data used 0-indexed
> palette colors without a transparency token.

### Decision 4: Week 1 Dataset ✅ LOCKED

**Pokemon only for Week 1.**

- 3,301 sprites (front/back/shiny/shiny_back variants)
- Homogeneous style, already downloaded
- Sufficient to build and validate the full pipeline through first AR samples
- Sprites dataset (~170K frames): bring in Week 2+ with proper asset-group splitting
- OpenGameArt: deferred, may be cut

### Decision 5: Split Strategy ✅ LOCKED

**Split by Pokemon ID, never by frame/variant. 80/10/10.**

- ~905 unique Pokemon IDs
- ~724 train / ~90 val / ~90 test
- All variants (front/back/shiny/shiny_back) of one Pokemon stay in the same split
- Saved as JSON: `splits.json` → `{"1": "train", "2": "val", ...}`
- Deterministic (seed=42) for reproducibility
- For Sprites dataset later: split by sheet_id / character_id (same principle)

### Decision 6: Mainline vs. Stretch ✅ LOCKED

**Deterministic tokenizer + VAR AR is the mainline. Everything else is stretch.**

| Priority | Component | Status |
|----------|-----------|--------|
| **Mainline** | Deterministic palette-index tokenizer + VAR next-scale AR Transformer | Must ship |
| **Stretch 1** | Learned VQ-VAE tokenizer (swap behind same interface) | If time allows |
| **Stretch 2** | HMAR masked refinement (Option B) | Only after VAR generates credible samples |
| **Stretch 3** | Multi-dataset training (Sprites 170K + OpenGameArt) | Only after pipeline proven |

Success condition: AR Transformer generates recognizable sprites from the deterministic token pyramid.

### Decision 7: Repo Source of Truth ✅ LOCKED

**Package code + scripts are the source of truth.**

- All model definitions, training logic, preprocessing logic live in `pixelvar/` or `scripts/`
- Notebooks are for demos, audits, and visualization only
- No logic should exist only in a notebook

---

## Tensor Shapes & Interfaces

### Scale schedule
```
K = 6 scales: [1, 2, 4, 8, 16, 32]
Total tokens per image: 1 + 4 + 16 + 64 + 256 + 1024 = 1,365
```

### Vocabulary
```
Token 0:      transparent
Token 1–16:   palette colors (for palette_size=16)
Token 17:     [MASK] (reserved for HMAR, unused in v0)
vocab_size:   17 (v0), 18 (with mask token)
```

### Core tensors
```
index_map:        (B, 32, 32)     long    values in [0, 16]
alpha_mask:       (B, 32, 32)     bool    True = opaque, False = transparent
multi_scale_maps: list of K tensors, each (B, h_k, w_k) long
token_sequence:   (B, 1365)       long    concatenated multi_scale_maps
rgb_preview:      (B, 3, 32, 32)  float   for visualization only, never used in training
```

### Tokenizer interface
```python
class BaseTokenizer:
    def encode(self, index_map) -> list[Tensor]:
        """(B, 32, 32) -> list of (B, h_k, w_k) multi-scale maps"""
    
    def decode(self, multi_scale_maps) -> Tensor:
        """list of (B, h_k, w_k) -> (B, 32, 32) reconstructed index map"""
    
    def to_sequence(self, multi_scale_maps) -> Tensor:
        """list of (B, h_k, w_k) -> (B, 1365) flat token sequence"""
    
    def from_sequence(self, token_sequence) -> list[Tensor]:
        """(B, 1365) -> list of (B, h_k, w_k) multi-scale maps"""
```

### AR model interface
```python
class VARTransformer:
    def forward(self, token_sequence) -> Tensor:
        """(B, 1365) -> (B, 1365, vocab_size) logits"""
    
    def sample(self, temperature=1.0, top_k=None) -> Tensor:
        """() -> (B, 1365) generated token sequence"""
```
