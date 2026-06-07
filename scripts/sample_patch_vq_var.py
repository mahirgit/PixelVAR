#!/usr/bin/env python3
"""Sample a VAR trained on patch-VQ tokens and decode with the learned codebook."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixelvar.tokenizers import DeterministicPyramidTokenizer
from pixelvar.training import load_var_model_from_checkpoint
from pixelvar.utils import load_yaml, save_rgba_grid


def decode_patch_codes(code_maps: np.ndarray, codebook: np.ndarray, patch_size: int) -> np.ndarray:
    code_maps = np.asarray(code_maps, dtype=np.int64)
    patches = codebook[code_maps]
    n, grid_h, grid_w = code_maps.shape
    channels = patches.shape[-1] // (patch_size * patch_size)
    patches = patches.reshape(n, grid_h, grid_w, patch_size, patch_size, channels)
    images = patches.transpose(0, 1, 3, 2, 4, 5).reshape(n, grid_h * patch_size, grid_w * patch_size, channels)
    return np.round(np.clip(images, 0.0, 1.0) * 255.0).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample patch-VQ VAR checkpoint")
    parser.add_argument("--var-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("data/processed/sprites_patchvq16"))
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("outputs/samples/sprites_patchvq16_v0_full_t1_top16.png"))
    parser.add_argument("--save-arrays", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    manifest = json.loads((args.tokenizer_dir / "manifest.json").read_text())
    scale_resolutions = config["model"].get("scale_resolutions", manifest.get("scale_resolutions", [1, 2, 4, 8, 16]))
    tokenizer = DeterministicPyramidTokenizer(scale_resolutions)
    codebook = np.load(args.tokenizer_dir / "codebook.npy")
    patch_size = int(manifest.get("patch_size", 2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    var_model = load_var_model_from_checkpoint(args.var_checkpoint, map_location=device).to(device)
    var_model.eval()

    with torch.no_grad():
        tokens = var_model.sample(
            batch_size=args.num_samples,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
        )
        code_maps = tokenizer.from_sequence(tokens)[-1].cpu().numpy()

    images = decode_patch_codes(code_maps, codebook, patch_size=patch_size)
    save_rgba_grid(images, args.output, columns=8)
    if args.save_arrays:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output.with_suffix(".tokens.npy"), tokens.cpu().numpy().astype(np.uint16))
        np.save(args.output.with_suffix(".code_maps.npy"), code_maps.astype(np.uint16))
    print(f"Saved patch-VQ VAR sample grid to {args.output}")


if __name__ == "__main__":
    main()
