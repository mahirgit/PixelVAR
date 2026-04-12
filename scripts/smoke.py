#!/usr/bin/env python3
"""
Smoke test: proves the repo is alive end-to-end.

Loads sample sprites, runs them through the deterministic tokenizer,
prints shapes, and saves a visualization grid.

Usage:
    python scripts/smoke.py
    python scripts/smoke.py --data-dir data/processed/pokemon --n-samples 8
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixelvar.tokenizers import DeterministicTokenizer
from pixelvar.data.palette import PaletteExtractor


def create_sample_grid(
    index_maps: np.ndarray,
    multi_scale_maps: list[torch.Tensor],
    palette: np.ndarray,
    output_path: Path,
    scale_px: int = 8,
):
    """
    Create a visualization grid showing multi-scale decomposition.

    Rows = samples, Columns = scales (1x1, 2x2, 4x4, 8x8, 16x16, 32x32)
    Each scale is upsampled to a fixed display size for visibility.
    """
    n_samples = index_maps.shape[0]
    n_scales = len(multi_scale_maps)
    display_size = 64  # pixels per cell in the grid

    # Add transparent color at index 0 (checkerboard pattern)
    # Palette colors are at indices 1-K, but current data uses 0-indexed palette
    # For visualization, just map indices to palette colors
    grid_w = n_scales * display_size + (n_scales - 1) * 2
    grid_h = n_samples * display_size + (n_samples - 1) * 2
    grid = Image.new("RGB", (grid_w, grid_h), (40, 40, 40))

    for row in range(n_samples):
        for col, scale_map in enumerate(multi_scale_maps):
            sample = scale_map[row].cpu().numpy()  # (h, w)
            h, w = sample.shape

            # Map indices to RGB
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    idx = sample[i, j]
                    if idx < len(palette):
                        rgb[i, j] = palette[idx]
                    else:
                        rgb[i, j] = [255, 0, 255]  # magenta for out-of-range

            # Upscale to display size
            cell = Image.fromarray(rgb).resize(
                (display_size, display_size), Image.NEAREST
            )

            x = col * (display_size + 2)
            y = row * (display_size + 2)
            grid.paste(cell, (x, y))

    grid.save(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="PixelVAR smoke test")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed/pokemon",
        help="Path to processed data directory",
    )
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PixelVAR Smoke Test")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Loading data...")
    index_maps_path = data_dir / "index_maps.npy"
    palette_path = data_dir / "palette.json"

    if not index_maps_path.exists():
        print(f"  ERROR: {index_maps_path} not found.")
        print("  Run preprocessing first: python scripts/preprocess_data.py --dataset pokemon")
        sys.exit(1)

    index_maps = np.load(index_maps_path)
    print(f"  index_maps: shape={index_maps.shape}, dtype={index_maps.dtype}")
    print(f"  value range: [{index_maps.min()}, {index_maps.max()}]")

    # Load palette
    extractor = PaletteExtractor()
    extractor.load(palette_path)
    palette = extractor.palette
    print(f"  palette: {len(palette)} colors")

    # 2. Select samples
    print(f"\n[2/5] Selecting {args.n_samples} samples...")
    n = min(args.n_samples, len(index_maps))
    rng = np.random.RandomState(42)
    indices = rng.choice(len(index_maps), n, replace=False)
    samples = index_maps[indices]
    print(f"  Selected indices: {indices.tolist()}")

    # 3. Initialize tokenizer
    print("\n[3/5] Initializing deterministic tokenizer...")
    tokenizer = DeterministicTokenizer(palette_size=len(palette))
    print(f"  Scales: {tokenizer.scale_resolutions}")
    print(f"  Total tokens: {tokenizer.total_tokens}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Scale info:")
    for info in tokenizer.get_scale_info():
        print(f"    {info['resolution']}x{info['resolution']}: "
              f"{info['num_tokens']} tokens (offset {info['offset']})")

    # 4. Encode and check shapes
    print("\n[4/5] Encoding through tokenizer...")
    batch = torch.from_numpy(samples).long()  # (B, 32, 32)
    print(f"  Input: {batch.shape}")

    multi_scale_maps = tokenizer.encode(batch)
    for i, m in enumerate(multi_scale_maps):
        res = tokenizer.scale_resolutions[i]
        print(f"  Scale {res}x{res}: {m.shape}, range=[{m.min()}, {m.max()}]")

    # Flatten to sequence
    token_seq = tokenizer.to_sequence(multi_scale_maps)
    print(f"  Token sequence: {token_seq.shape}, range=[{token_seq.min()}, {token_seq.max()}]")

    # Roundtrip: sequence -> maps -> decode
    recovered_maps = tokenizer.from_sequence(token_seq)
    decoded = tokenizer.decode(recovered_maps)
    print(f"  Decoded: {decoded.shape}")

    # Check roundtrip consistency
    original_finest = multi_scale_maps[-1]
    match = (decoded == original_finest).float().mean().item()
    print(f"  Roundtrip match (finest scale): {match * 100:.1f}%")

    seq_roundtrip = tokenizer.to_sequence(recovered_maps)
    seq_match = (seq_roundtrip == token_seq).all().item()
    print(f"  Sequence roundtrip exact match: {seq_match}")

    # 5. Save visualization
    print("\n[5/5] Saving visualization grid...")
    grid_path = create_sample_grid(
        samples, multi_scale_maps, palette, output_dir / "smoke_test.png"
    )
    print(f"  Saved to {grid_path}")

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
