# PixelVAR — TA Meeting Checklist (April 13, 2026)

## 📅 Context
- **Milestone 1 deadline (from proposal):** April 11, 2026 — "Dataset curation, palette extraction"
- **Next milestone:** April 25 — "Multi-scale VQ-VAE tokenizer training"
- **Official progress presentations (syllabus):** April 28–30 (Week 12)

---

## ✅ What You HAVE Done (Show These to the TA)

### 1. Project Setup & Infrastructure
- [x] Clean project structure with `setup.py`, `requirements.txt`, proper `.gitignore`
- [x] Git repo initialized with initial commit
- [x] Configuration system (`configs/default.py`) with data params (scales, palette size, etc.)
- [x] Installable Python package (`pixelvar/`)

### 2. Data Download Pipeline (`scripts/download_data.py`)
- [x] Pokemon sprites downloaded from PokeAPI (~905 sprites)
- [x] Scripts ready for Sprites dataset (170K) and OpenGameArt (manual curation)
- [x] Download script supports `--dataset` flag for all three sources

### 3. Preprocessing Pipeline (`scripts/preprocess_data.py`)
- [x] Resize to 32×32 using nearest-neighbor interpolation (preserves pixel art aesthetic)
- [x] RGBA → RGB compositing onto white background (handles transparency)
- [x] 16-color palette extraction via MiniBatchKMeans (sorted by luminance)
- [x] Quantization of every pixel to nearest palette index → `(N, 32, 32)` index maps
- [x] Saves: `index_maps.npy`, `quantized_rgb.npy`, `originals_rgb.npy`, `palette.json`
- [x] Processed Pokemon data exists and is ready

### 4. Palette System (`pixelvar/data/palette.py`)
- [x] `PaletteExtractor` class: `fit()`, `quantize()`, `save()`, `load()`, `visualize_palette()`
- [x] Handles RGBA, grayscale, and RGB input formats
- [x] Subsampling for efficient k-means on large datasets

### 5. PyTorch Dataset & DataLoader (`pixelvar/data/dataset.py`)
- [x] `PixelArtDataset`: loads processed data, creates multi-scale token maps on-the-fly
- [x] Multi-scale decomposition: 6 scales (1×1 → 2×2 → 4×4 → 8×8 → 16×16 → 32×32)
- [x] Total 1,365 tokens per image, vocabulary size = 16 (palette size)
- [x] Flattened `token_sequence` ready for Transformer input
- [x] Custom `collate_fn` for batching multi-scale maps
- [x] `get_dataloader()` and `get_combined_dataloader()` helper functions

### 6. Visualizations (`scripts/visualize.py` + `visualizations/pokemon/`)
- [x] Palette swatch plot with hex codes
- [x] Original vs. quantized comparison grids
- [x] Multi-scale decomposition visualizations (coarse → fine)
- [x] Index map heatmaps alongside RGB renderings
- [x] Dataset statistics (palette index distribution, colors-per-sprite histogram)
- [x] All saved as PNGs in `visualizations/pokemon/`

### 7. Demo Notebook (`notebooks/01_data_pipeline_demo.ipynb`)
- [x] End-to-end walkthrough: download → preprocess → palette → quantize → multi-scale → DataLoader → stats

---

## ⚠️ Issues to Be Aware Of (TA May Ask About These)

### 1. Palette Quality Problem 🔴
- **4 out of 16 palette colors are pure white (255, 255, 255)** — this wastes 25% of palette capacity!
- This happens because the white background (from RGBA compositing) dominates k-means clustering
- **Fix:** Either exclude background pixels from k-means, or use a smaller background cluster + more foreground colors
- **Talking point:** "We identified this issue and will fix it by filtering out background-dominant pixels before clustering"

### 2. Only Pokemon Dataset Processed
- Proposal mentions 3 datasets: Sprites (~170K), Pokemon (~8K), OpenGameArt
- Currently only ~905 Pokemon sprites downloaded and processed
- Sprites dataset and OpenGameArt not yet downloaded
- **Talking point:** "Pokemon is our development dataset; we'll scale to the full Sprites dataset for training"

### 3. Notebook Not Executed
- The demo notebook cells don't have saved outputs (most `execution_count` is null)
- **Action before meeting:** Run the notebook end-to-end so outputs are visible

### 4. Models Not Started Yet
- `pixelvar/models/__init__.py` is empty — no VQ-VAE or Transformer code yet
- This is **expected** per the work plan (VQ-VAE due April 25), but the TA may ask about plans

---

## 📋 TO-DO Before Tomorrow's Meeting

### Must-Do (Tonight)
- [ ] **Run the demo notebook end-to-end** so all cells have visible outputs/plots
- [ ] **Prepare a 2-3 minute verbal walkthrough:** proposal recap → what's done → what's next
- [ ] **Be ready to explain the multi-scale approach** (the core novelty): why 6 scales, why mode-downsampling, how 1,365 tokens work

### Nice-to-Have (If Time Permits)
- [ ] Fix the palette white-duplication issue (filter background pixels before k-means)
- [ ] Download the full Sprites dataset (~170K) to show data scale
- [ ] Add a short summary slide or markdown file with key numbers

---

## 🗣️ Key Talking Points for the TA

1. **Milestone 1 is complete** — data pipeline is fully functional (download → preprocess → palette → quantize → multi-scale → DataLoader)
2. **Core design decision:** We work in discrete palette-index space (vocab=16), not continuous RGB — this is what differentiates us from prior work
3. **Multi-scale decomposition is working:** 1×1 → 32×32 in 6 scales, 1,365 tokens per sprite, exactly matching the VAR framework
4. **Next 2 weeks:** Build the multi-scale VQ-VAE tokenizer (due April 25), then the VAR Transformer (due May 9)
5. **Known issue:** Palette extraction needs refinement (background pixel filtering) — we're aware and will fix

---

## 📊 Key Numbers to Know

| Metric | Value |
|--------|-------|
| Pokemon sprites downloaded | ~905 |
| Image resolution | 32×32 |
| Palette size | 16 colors |
| Number of scales | 6 (1×1 to 32×32) |
| Tokens per image | 1,365 |
| Vocabulary size | 16 (= palette size) |
| Total token breakdown | 1 + 4 + 16 + 64 + 256 + 1024 = 1,365 |

---

## 📈 Overall Assessment

**You are in good shape for tomorrow.** Milestone 1 (dataset curation + palette extraction) is essentially complete and on schedule. The data pipeline is well-engineered with proper abstractions. The main gap is the palette quality issue (duplicate whites) which is a minor fix. The TA will likely be satisfied with the progress and ask about your plans for the VQ-VAE tokenizer (Milestone 2).
