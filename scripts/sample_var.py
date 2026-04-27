#!/usr/bin/env python3
"""Sample sprites from a trained PixelVAR checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixelvar.data.palette import PaletteExtractor
from pixelvar.training import LitVAR
from pixelvar.utils import load_yaml, save_rgba_grid, tokens_to_rgba


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from a PixelVAR checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/samples/sample_grid.png"))
    args = parser.parse_args()

    config = load_yaml(args.config)
    processed_dir = Path(config["data"]["processed_dir"])

    module = LitVAR.load_from_checkpoint(args.checkpoint)
    tokens = module.sample(batch_size=args.num_samples, temperature=args.temperature, top_k=args.top_k)

    palette = PaletteExtractor()
    palette.load(processed_dir / "palette.json")
    images = tokens_to_rgba(tokens.cpu(), palette, scale_resolutions=config["model"].get("scale_resolutions"))
    save_rgba_grid(images, args.output, columns=int(args.num_samples**0.5) or 1)
    print(f"Saved sample grid to {args.output}")


if __name__ == "__main__":
    main()
