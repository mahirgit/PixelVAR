#!/usr/bin/env python3
"""Sample sprites from a trained PixelVAR HMAR checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixelvar.data.palette import PaletteExtractor
from pixelvar.training import load_hmar_model_from_checkpoint
from pixelvar.utils import load_yaml, save_rgba_grid, tokens_to_rgba


def parse_top_k(value: str) -> int | None:
    text = value.strip().lower()
    if text in {"none", "null", "0"}:
        return None
    return int(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from a PixelVAR HMAR checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--refinement-steps", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=parse_top_k, default=8)
    parser.add_argument("--mask-schedule", choices=["cosine", "linear"], default="cosine")
    parser.add_argument("--output", type=Path, default=Path("outputs/samples/hmar_sample_grid.png"))
    args = parser.parse_args()

    config = load_yaml(args.config)
    processed_dir = Path(config["data"]["processed_dir"])

    model = load_hmar_model_from_checkpoint(args.checkpoint)
    tokens = model.sample(
        batch_size=args.num_samples,
        refinement_steps=args.refinement_steps,
        temperature=args.temperature,
        top_k=args.top_k,
        mask_schedule=args.mask_schedule,
    )

    palette = PaletteExtractor()
    palette.load(processed_dir / "palette.json")
    images = tokens_to_rgba(tokens.cpu(), palette, scale_resolutions=config["model"].get("scale_resolutions"))
    save_rgba_grid(images, args.output, columns=8)
    print(f"Saved HMAR sample grid to {args.output}")


if __name__ == "__main__":
    main()
