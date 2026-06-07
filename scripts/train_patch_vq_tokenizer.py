#!/usr/bin/env python3
"""Train a learned patch-VQ tokenizer and export VAR-ready token maps."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixelvar.utils import save_rgba_grid


def load_manifest(processed_dir: Path) -> dict:
    manifest_path = processed_dir / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    image_count = len(np.load(processed_dir / "originals_rgba.npy", mmap_mode="r"))
    return {"dataset": processed_dir.name, "num_samples": image_count, "samples": [{"index": i} for i in range(image_count)]}


def split_indices(manifest: dict, split: str) -> list[int]:
    return [
        int(sample.get("index", idx))
        for idx, sample in enumerate(manifest.get("samples", []))
        if sample.get("split") == split
    ]


def clean_rgba(images: np.ndarray) -> np.ndarray:
    arr = np.asarray(images, dtype=np.uint8).copy()
    alpha = arr[..., 3:4].astype(np.float32) / 255.0
    arr[..., :3] = np.round(arr[..., :3].astype(np.float32) * alpha).astype(np.uint8)
    return arr


def images_to_patches(images: np.ndarray, patch_size: int) -> np.ndarray:
    images = clean_rgba(images)
    n, height, width, channels = images.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(f"patch_size={patch_size} must divide image size {height}x{width}")
    grid_h = height // patch_size
    grid_w = width // patch_size
    patches = images.reshape(n, grid_h, patch_size, grid_w, patch_size, channels)
    patches = patches.transpose(0, 1, 3, 2, 4, 5)
    return patches.reshape(n, grid_h, grid_w, patch_size * patch_size * channels)


def decode_patch_codes(code_maps: np.ndarray, codebook: np.ndarray, patch_size: int) -> np.ndarray:
    code_maps = np.asarray(code_maps, dtype=np.int64)
    if code_maps.ndim == 2:
        code_maps = code_maps[None, ...]
    patches = codebook[code_maps]
    n, grid_h, grid_w = code_maps.shape
    channels = patches.shape[-1] // (patch_size * patch_size)
    patches = patches.reshape(n, grid_h, grid_w, patch_size, patch_size, channels)
    images = patches.transpose(0, 1, 3, 2, 4, 5).reshape(n, grid_h * patch_size, grid_w * patch_size, channels)
    return np.round(np.clip(images, 0.0, 1.0) * 255.0).astype(np.uint8)


def sample_training_patches(
    patches: np.ndarray,
    train_indices: list[int],
    max_patches: int,
    seed: int,
) -> np.ndarray:
    train_patches = patches[train_indices].reshape(-1, patches.shape[-1])
    if len(train_patches) <= max_patches:
        return train_patches.astype(np.float32) / 255.0
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(train_patches), size=max_patches, replace=False)
    return train_patches[selected].astype(np.float32) / 255.0


def nearest_codes(patches: np.ndarray, kmeans: MiniBatchKMeans, batch_patches: int) -> np.ndarray:
    flat = patches.reshape(-1, patches.shape[-1]).astype(np.float32) / 255.0
    codes = np.empty(len(flat), dtype=np.int64)
    for start in range(0, len(flat), batch_patches):
        end = min(start + batch_patches, len(flat))
        codes[start:end] = kmeans.predict(flat[start:end])
    return codes.reshape(patches.shape[:3])


def pyramid_scales(final_size: int) -> list[int]:
    scales = [1]
    while scales[-1] < final_size:
        scales.append(scales[-1] * 2)
    if scales[-1] != final_size:
        raise ValueError(f"final size {final_size} is not a power-of-two pyramid size")
    return scales


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a patch-VQ tokenizer")
    parser.add_argument("--source-dir", type=Path, default=Path("data/processed/sprites"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/sprites_patchvq16"))
    parser.add_argument("--image-array", type=str, default="originals_rgba.npy")
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--num-codes", type=int, default=512)
    parser.add_argument("--max-patches", type=int, default=2_000_000)
    parser.add_argument("--kmeans-batch-size", type=int, default=8192)
    parser.add_argument("--predict-batch-patches", type=int, default=262_144)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-samples", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists():
        if not args.force:
            raise FileExistsError(f"{args.output_dir} exists; pass --force to overwrite it")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    images = np.load(args.source_dir / args.image_array, mmap_mode="r")
    if images.ndim != 4 or images.shape[1:3] != (32, 32) or images.shape[-1] != 4:
        raise ValueError(f"Expected image array shape (N, 32, 32, 4), got {images.shape}")
    manifest = load_manifest(args.source_dir)
    train_indices = split_indices(manifest, "train") or list(range(len(images)))

    print("Extracting patches")
    patches = images_to_patches(np.asarray(images), patch_size=args.patch_size)
    train_patches = sample_training_patches(
        patches=patches,
        train_indices=train_indices,
        max_patches=args.max_patches,
        seed=args.seed,
    )

    print(f"Training MiniBatchKMeans: patches={len(train_patches):,}, codes={args.num_codes}")
    kmeans = MiniBatchKMeans(
        n_clusters=args.num_codes,
        random_state=args.seed,
        batch_size=args.kmeans_batch_size,
        n_init=3,
        reassignment_ratio=0.01,
        verbose=0,
    )
    kmeans.fit(train_patches)
    codebook = np.asarray(kmeans.cluster_centers_, dtype=np.float32)

    grid_size = 32 // args.patch_size
    dtype = np.uint8 if args.num_codes <= 256 else np.uint16
    token_maps = np.lib.format.open_memmap(
        args.output_dir / "index_maps.npy",
        mode="w+",
        dtype=dtype,
        shape=(len(images), grid_size, grid_size),
    )

    code_counts = np.zeros(args.num_codes, dtype=np.int64)
    for start in tqdm(range(0, len(images), 1024), desc="Encoding patch VQ tokens"):
        end = min(start + 1024, len(images))
        batch_patches = images_to_patches(np.asarray(images[start:end]), patch_size=args.patch_size)
        codes = nearest_codes(batch_patches, kmeans, batch_patches=args.predict_batch_patches).astype(dtype)
        token_maps[start:end] = codes
        code_counts += np.bincount(codes.reshape(-1), minlength=args.num_codes)
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
        "tokenizer": "patch_vq",
        "patch_size": args.patch_size,
        "latent_size": grid_size,
        "target_size": grid_size,
        "vocab_size": args.num_codes,
        "scale_resolutions": pyramid_scales(grid_size),
        "num_samples": len(images),
        "samples": samples,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(out_manifest, indent=2))
    np.save(args.output_dir / "codebook.npy", codebook)
    (args.output_dir / "codebook_usage.json").write_text(
        json.dumps(
            {
                "num_codes": args.num_codes,
                "used_codes": int((code_counts > 0).sum()),
                "usage_fraction": float((code_counts > 0).mean()),
                "counts": code_counts.tolist(),
            },
            indent=2,
        )
        + "\n"
    )

    preview_count = min(args.grid_samples, len(images))
    recon = decode_patch_codes(np.asarray(token_maps[:preview_count]), codebook, patch_size=args.patch_size)
    target = clean_rgba(np.asarray(images[:preview_count]))
    compare = np.concatenate([target, recon], axis=2)
    save_rgba_grid(recon, args.output_dir / "reconstruction_grid.png", columns=8)
    save_rgba_grid(compare, args.output_dir / "reconstruction_compare_grid.png", columns=4)

    print("Patch-VQ tokenizer export complete")
    print(f"  output: {args.output_dir}")
    print(f"  token maps: {token_maps.shape}, dtype={token_maps.dtype}")
    print(f"  vocab size: {args.num_codes}")
    print(f"  used codes: {int((code_counts > 0).sum())}/{args.num_codes}")
    print(f"  reconstruction grid: {args.output_dir / 'reconstruction_grid.png'}")


if __name__ == "__main__":
    main()
