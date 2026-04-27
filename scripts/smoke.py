#!/usr/bin/env python3
"""End-to-end smoke check for PixelVAR Option A."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import lightning as L
except ModuleNotFoundError as exc:  # pragma: no cover - CLI guard.
    raise SystemExit("Lightning is not installed. Run: pip install 'lightning>=2.6,<2.7'") from exc

from pixelvar.data.datamodule import PixelArtDataModule
from pixelvar.data.palette import PaletteExtractor
from pixelvar.data.splits import make_id_splits
from pixelvar.training import LitVAR
from pixelvar.utils import save_rgba_grid, tokens_to_rgba


def build_synthetic_dataset(processed_dir: Path, n_samples: int = 48) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    palette_colors = np.array(
        [
            [20, 20, 24],
            [80, 52, 40],
            [180, 64, 56],
            [236, 144, 80],
            [252, 220, 120],
            [88, 168, 96],
            [40, 104, 88],
            [72, 112, 184],
            [120, 184, 232],
            [48, 48, 96],
            [120, 72, 160],
            [216, 104, 184],
            [232, 232, 232],
            [160, 160, 160],
            [96, 96, 96],
            [40, 40, 40],
        ],
        dtype=np.uint8,
    )
    palette = PaletteExtractor(palette_size=16)
    palette.palette = palette_colors
    palette.save(processed_dir / "palette.json")

    index_maps = np.zeros((n_samples, 32, 32), dtype=np.uint8)
    for i in range(n_samples):
        token = (i % 16) + 1
        x0 = 8 + (i % 4)
        y0 = 7 + ((i // 4) % 4)
        index_maps[i, y0 : y0 + 16, x0 : x0 + 14] = token
        index_maps[i, y0 + 3 : y0 + 13, x0 + 3 : x0 + 11] = ((token + 4 - 1) % 16) + 1
        if i % 3 == 0:
            index_maps[i, y0 + 6 : y0 + 10, x0 + 6 : x0 + 8] = 0
        noise_y = rng.integers(y0, y0 + 16, size=8)
        noise_x = rng.integers(x0, x0 + 14, size=8)
        index_maps[i, noise_y, noise_x] = rng.integers(1, 17, size=8)

    alpha_masks = index_maps != 0
    quantized_rgba = np.stack([palette.render_index_map(m, a) for m, a in zip(index_maps, alpha_masks)], axis=0)

    split_map = make_id_splits([str(i + 1) for i in range(n_samples)], seed=42)
    samples = []
    for i in range(n_samples):
        pokemon_id = str(i + 1)
        samples.append(
            {
                "index": i,
                "path": f"synthetic/{pokemon_id}.png",
                "pokemon_id": pokemon_id,
                "variant": "front",
                "split": split_map[pokemon_id],
            }
        )
    manifest = {
        "dataset": "pokemon",
        "target_size": 32,
        "palette_size": 16,
        "alpha_threshold": 128,
        "transparent_token": 0,
        "palette_token_start": 1,
        "palette_token_end": 16,
        "vocab_size": 17,
        "scale_resolutions": [1, 2, 4, 8, 16, 32],
        "num_samples": n_samples,
        "samples": samples,
    }

    np.save(processed_dir / "index_maps.npy", index_maps)
    np.save(processed_dir / "alpha_masks.npy", alpha_masks)
    np.save(processed_dir / "quantized_rgba.npy", quantized_rgba)
    (processed_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (processed_dir / "splits.json").write_text(json.dumps(split_map, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a PixelVAR smoke test")
    parser.add_argument("--processed-dir", type=Path, default=Path("outputs/smoke_data/pokemon"))
    parser.add_argument("--use-existing", action="store_true", help="Use processed-dir as-is instead of synthetic data")
    args = parser.parse_args()

    if not args.use_existing:
        build_synthetic_dataset(args.processed_dir)

    L.seed_everything(42, workers=True)
    datamodule = PixelArtDataModule(
        processed_dir=args.processed_dir,
        batch_size=8,
        num_workers=0,
        max_train_samples=16,
        max_val_samples=8,
        return_rgb=False,
    )
    module = LitVAR(
        model_config={
            "vocab_size": 17,
            "d_model": 32,
            "n_layers": 1,
            "n_heads": 4,
            "mlp_dim": 64,
            "dropout": 0.0,
        },
        optimizer_config={"lr": 1e-3, "weight_decay": 0.0},
    )

    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        precision="32-true",
        max_steps=2,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    trainer.fit(module, datamodule=datamodule)

    with torch.no_grad():
        tokens = module.sample(batch_size=4, temperature=1.0, top_k=8)
    palette = PaletteExtractor()
    palette.load(args.processed_dir / "palette.json")
    images = tokens_to_rgba(tokens.cpu(), palette)
    output = Path("outputs/smoke/sample_grid.png")
    save_rgba_grid(images, output, columns=2, scale=4)

    assert tokens.shape == (4, 1365)
    assert int(tokens.min()) >= 0 and int(tokens.max()) <= 16
    print("Smoke test passed")
    print(f"  sample grid: {output}")


if __name__ == "__main__":
    main()
