#!/usr/bin/env python3
"""Export VQ-VAE code maps as a VAR-ready processed token dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixelvar.data.vqvae_dataset import get_sprite_image_dataloader
from pixelvar.training import load_vqvae_model_from_checkpoint
from pixelvar.utils import save_rgba_grid


def to_rgba_uint8(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().clamp(0.0, 1.0).permute(0, 2, 3, 1).numpy()
    return np.round(arr * 255.0).astype(np.uint8)


def load_manifest(processed_dir: Path) -> dict:
    manifest_path = processed_dir / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    image_count = len(np.load(processed_dir / "originals_rgba.npy", mmap_mode="r"))
    return {"dataset": processed_dir.name, "num_samples": image_count, "samples": [{"index": i} for i in range(image_count)]}


def pyramid_scales(final_size: int) -> list[int]:
    scales = [1]
    while scales[-1] < final_size:
        scales.append(scales[-1] * 2)
    if scales[-1] != final_size:
        raise ValueError(f"latent size {final_size} is not a power-of-two pyramid size")
    return scales


def main() -> None:
    parser = argparse.ArgumentParser(description="Export VQ-VAE token maps")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=Path("data/processed/sprites"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/sprites_vqvae16"))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-array", type=str, default="originals_rgba.npy")
    parser.add_argument("--grid-samples", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists():
        if not args.force:
            raise FileExistsError(f"{args.output_dir} exists; pass --force to overwrite it")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_vqvae_model_from_checkpoint(args.checkpoint, map_location=device).to(device)
    model.eval()

    manifest = load_manifest(args.source_dir)
    num_samples = int(manifest.get("num_samples", len(manifest.get("samples", []))))
    latent_size = int(model.latent_size)
    num_codes = int(model.num_codes)
    dtype = np.uint8 if num_codes <= 256 else np.uint16
    token_maps = np.lib.format.open_memmap(
        args.output_dir / "index_maps.npy",
        mode="w+",
        dtype=dtype,
        shape=(num_samples, latent_size, latent_size),
    )

    loader = get_sprite_image_dataloader(
        processed_dir=args.source_dir,
        split="all",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        image_array=args.image_array,
    )

    recon_images = []
    compare_images = []
    code_counts = np.zeros(num_codes, dtype=np.int64)
    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding VQ tokens"):
            images = batch["image"].to(device)
            indices = batch["index"].numpy()
            output = model(images)
            codes = output.code_indices.detach().cpu().numpy().astype(dtype)
            token_maps[indices] = codes
            code_counts += np.bincount(codes.reshape(-1), minlength=num_codes)

            if len(recon_images) < args.grid_samples:
                remaining = args.grid_samples - len(recon_images)
                take = min(remaining, images.shape[0])
                recon = to_rgba_uint8(output.recon[:take])
                target = to_rgba_uint8(images[:take])
                recon_images.extend(list(recon))
                compare_images.extend(list(np.concatenate([target, recon], axis=2)))

    token_maps.flush()

    samples = []
    for idx, sample in enumerate(manifest.get("samples", [])):
        exported = dict(sample)
        exported["index"] = int(sample.get("index", idx))
        samples.append(exported)

    out_manifest = {
        "dataset": args.output_dir.name,
        "source_dataset": manifest.get("dataset", args.source_dir.name),
        "source_processed_dir": str(args.source_dir),
        "source_checkpoint": str(args.checkpoint),
        "tokenizer": "vqvae",
        "latent_size": latent_size,
        "target_size": latent_size,
        "vocab_size": num_codes,
        "scale_resolutions": pyramid_scales(latent_size),
        "num_samples": num_samples,
        "samples": samples,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(out_manifest, indent=2))
    (args.output_dir / "codebook_usage.json").write_text(
        json.dumps(
            {
                "num_codes": num_codes,
                "used_codes": int((code_counts > 0).sum()),
                "usage_fraction": float((code_counts > 0).mean()),
                "counts": code_counts.tolist(),
            },
            indent=2,
        )
        + "\n"
    )

    if recon_images:
        save_rgba_grid(np.stack(recon_images, axis=0), args.output_dir / "reconstruction_grid.png", columns=8)
        save_rgba_grid(np.stack(compare_images, axis=0), args.output_dir / "reconstruction_compare_grid.png", columns=4)

    print("VQ token export complete")
    print(f"  output: {args.output_dir}")
    print(f"  token maps: {token_maps.shape}, dtype={token_maps.dtype}")
    print(f"  vocab size: {num_codes}")
    print(f"  used codes: {int((code_counts > 0).sum())}/{num_codes}")
    print(f"  reconstruction grid: {args.output_dir / 'reconstruction_grid.png'}")


if __name__ == "__main__":
    main()
