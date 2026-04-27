#!/usr/bin/env python3
"""Validate a processed PixelVAR dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pixelvar.data.palette import PaletteExtractor
from pixelvar.data.splits import assert_no_split_leakage
from pixelvar.utils import save_rgba_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Check processed PixelVAR data")
    parser.add_argument("--dataset", type=str, default="pokemon")
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/data_checks"))
    args = parser.parse_args()

    data_dir = args.processed_root / args.dataset
    required = ["index_maps.npy", "alpha_masks.npy", "palette.json", "manifest.json"]
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        print(f"[error] Missing processed files in {data_dir}: {missing}")
        print(f"Run: python scripts/preprocess_data.py --dataset {args.dataset} --palette-size 16")
        sys.exit(1)

    index_maps = np.load(data_dir / "index_maps.npy")
    alpha_masks = np.load(data_dir / "alpha_masks.npy").astype(bool)
    manifest = json.loads((data_dir / "manifest.json").read_text())

    palette = PaletteExtractor()
    palette.load(data_dir / "palette.json")

    errors = []
    if index_maps.shape != alpha_masks.shape:
        errors.append(f"index_maps shape {index_maps.shape} != alpha_masks shape {alpha_masks.shape}")
    if index_maps.ndim != 3 or index_maps.shape[1:] != (32, 32):
        errors.append(f"index_maps must have shape (N, 32, 32), got {index_maps.shape}")
    if len(palette.palette) != 16:
        errors.append(f"expected 16 palette colors, got {len(palette.palette)}")
    if index_maps.min() < 0 or index_maps.max() > 16:
        errors.append(f"token range [{index_maps.min()}, {index_maps.max()}] outside [0, 16]")
    if not np.any(index_maps == 0):
        errors.append("transparent token 0 never appears")
    if not np.any(index_maps == 16):
        print("[warn] token 16 does not appear in this processed set; this can happen on tiny subsets")
    transparency = 1.0 - float(alpha_masks.mean())
    if transparency <= 0.0 or transparency >= 1.0:
        errors.append(f"unexpected transparency ratio {transparency:.4f}")
    if manifest.get("num_samples") != len(index_maps):
        errors.append(f"manifest num_samples {manifest.get('num_samples')} != array length {len(index_maps)}")
    try:
        assert_no_split_leakage(manifest.get("samples", []))
    except ValueError as exc:
        errors.append(str(exc))

    if errors:
        for error in errors:
            print(f"[error] {error}")
        sys.exit(1)

    rendered = []
    for i in range(min(16, len(index_maps))):
        rendered.append(palette.render_index_map(index_maps[i], alpha_masks[i]))
    sample_grid = args.output_dir / args.dataset / "sample_grid.png"
    save_rgba_grid(np.stack(rendered, axis=0), sample_grid, columns=4, scale=4)

    split_counts = {}
    for sample in manifest.get("samples", []):
        split = sample.get("split", "unsplit")
        split_counts[split] = split_counts.get(split, 0) + 1

    print("Data check passed")
    print(f"  samples: {len(index_maps)}")
    print(f"  token range: [{index_maps.min()}, {index_maps.max()}]")
    print(f"  transparency ratio: {transparency:.3f}")
    print(f"  split counts: {split_counts}")
    print(f"  sample grid: {sample_grid}")


if __name__ == "__main__":
    main()
