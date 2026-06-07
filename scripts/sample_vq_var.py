#!/usr/bin/env python3
"""Sample a VAR trained on VQ-VAE tokens and decode to RGBA sprites."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixelvar.tokenizers import DeterministicPyramidTokenizer
from pixelvar.training import load_var_model_from_checkpoint, load_vqvae_model_from_checkpoint
from pixelvar.utils import load_yaml, save_rgba_grid


def to_rgba_uint8(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().clamp(0.0, 1.0).permute(0, 2, 3, 1).numpy()
    return np.round(arr * 255.0).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample VQ-VAE-token VAR checkpoint")
    parser.add_argument("--var-checkpoint", type=Path, required=True)
    parser.add_argument("--vqvae-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("outputs/samples/sprites_vq_var_t08_top32.png"))
    parser.add_argument("--save-arrays", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    scale_resolutions = config["model"].get("scale_resolutions", [1, 2, 4, 8])
    tokenizer = DeterministicPyramidTokenizer(scale_resolutions)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    var_model = load_var_model_from_checkpoint(args.var_checkpoint, map_location=device).to(device)
    vqvae = load_vqvae_model_from_checkpoint(args.vqvae_checkpoint, map_location=device).to(device)
    var_model.eval()
    vqvae.eval()

    with torch.no_grad():
        tokens = var_model.sample(
            batch_size=args.num_samples,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
        )
        code_maps = tokenizer.from_sequence(tokens)[-1].to(device)
        decoded = vqvae.decode_code_indices(code_maps)

    images = to_rgba_uint8(decoded)
    save_rgba_grid(images, args.output, columns=8)
    if args.save_arrays:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output.with_suffix(".tokens.npy"), tokens.cpu().numpy().astype(np.uint16))
        np.save(args.output.with_suffix(".code_maps.npy"), code_maps.cpu().numpy().astype(np.uint16))
    print(f"Saved VQ-VAR sample grid to {args.output}")


if __name__ == "__main__":
    main()
