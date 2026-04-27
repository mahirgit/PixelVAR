#!/usr/bin/env python3
"""Preprocess raw sprite datasets into PixelVAR token maps."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pixelvar.data.palette import PaletteExtractor
from pixelvar.data.splits import (
    assert_no_split_leakage,
    infer_pokemon_variant,
    make_id_splits,
    parse_pokemon_id,
)


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    image: np.ndarray


def load_images_from_dir(img_dir: Path, max_images: int | None = None) -> list[ImageRecord]:
    """Load all image files from a directory as RGBA arrays."""
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
    paths = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in extensions)
    if max_images is not None:
        paths = paths[:max_images]

    records: list[ImageRecord] = []
    for path in tqdm(paths, desc=f"Loading {img_dir.name}"):
        try:
            image = Image.open(path).convert("RGBA")
            records.append(ImageRecord(path=path, image=np.array(image)))
        except Exception as exc:
            print(f"  [warn] Skipping {path}: {exc}")
    return records


def load_npy_sprites(npy_path: Path, max_images: int | None = None) -> list[ImageRecord]:
    """Load sprites from a numpy array file."""
    data = np.load(npy_path)
    print(f"  Loaded {npy_path.name}: shape={data.shape}, dtype={data.dtype}")
    if max_images is not None:
        data = data[:max_images]

    records: list[ImageRecord] = []
    for idx, item in enumerate(data):
        image = item
        if image.ndim == 1:
            side = int(np.sqrt(len(image)))
            image = image.reshape(side, side)
        if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
            image = np.transpose(image, (1, 2, 0))
        rgba = ensure_rgba(image)
        records.append(ImageRecord(path=Path(f"{npy_path.stem}_{idx:06d}.png"), image=rgba))
    return records


def ensure_rgba(image: np.ndarray) -> np.ndarray:
    """Convert a numpy image to uint8 RGBA without compositing transparency."""
    img = np.asarray(image)
    if img.ndim == 2:
        rgb = np.stack([img] * 3, axis=-1)
        alpha = np.full((*img.shape, 1), 255, dtype=np.uint8)
        return np.concatenate([rgb, alpha], axis=-1).astype(np.uint8)
    if img.shape[-1] == 4:
        return img.astype(np.uint8)
    if img.shape[-1] == 3:
        alpha = np.full((*img.shape[:2], 1), 255, dtype=np.uint8)
        return np.concatenate([img[:, :, :3], alpha], axis=-1).astype(np.uint8)
    if img.shape[-1] == 1:
        rgb = np.repeat(img, 3, axis=-1)
        alpha = np.full((*img.shape[:2], 1), 255, dtype=np.uint8)
        return np.concatenate([rgb, alpha], axis=-1).astype(np.uint8)
    raise ValueError(f"Unsupported image shape: {img.shape}")


def resize_rgba(image: np.ndarray, target_size: int = 32) -> np.ndarray:
    """Resize to target size with nearest-neighbor interpolation."""
    rgba = ensure_rgba(image)
    pil_img = Image.fromarray(rgba, mode="RGBA")
    pil_img = pil_img.resize((target_size, target_size), Image.NEAREST)
    return np.array(pil_img)


def build_manifest(
    dataset_name: str,
    records: list[ImageRecord],
    target_size: int,
    palette_size: int,
    alpha_threshold: int,
    split_map: dict[str, str] | None,
) -> dict:
    samples = []
    for idx, record in enumerate(records):
        pokemon_id = parse_pokemon_id(record.path) if dataset_name == "pokemon" else None
        sample = {
            "index": idx,
            "path": str(record.path),
        }
        if pokemon_id is not None:
            sample["pokemon_id"] = pokemon_id
            sample["variant"] = infer_pokemon_variant(record.path)
            if split_map is not None:
                sample["split"] = split_map[pokemon_id]
        samples.append(sample)

    if split_map is not None:
        assert_no_split_leakage(samples)

    return {
        "dataset": dataset_name,
        "target_size": target_size,
        "palette_size": palette_size,
        "alpha_threshold": alpha_threshold,
        "transparent_token": 0,
        "palette_token_start": 1,
        "palette_token_end": palette_size,
        "vocab_size": palette_size + 1,
        "scale_resolutions": [1, 2, 4, 8, 16, 32],
        "num_samples": len(samples),
        "samples": samples,
    }


def preprocess_dataset(
    dataset_name: str,
    records: list[ImageRecord],
    palette_size: int = 16,
    target_size: int = 32,
    alpha_threshold: int = 128,
    split_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, PaletteExtractor]:
    """Full preprocessing pipeline for a single dataset."""
    out_dir = PROCESSED_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 50}")
    print(f"Processing: {dataset_name} ({len(records)} images)")
    print(f"  Target size: {target_size}x{target_size}")
    print(f"  Palette size: {palette_size}")
    print(f"  Alpha threshold: {alpha_threshold}")
    print(f"{'=' * 50}")

    print("\n[1/5] Resizing images...")
    resized_records = [
        ImageRecord(path=record.path, image=resize_rgba(record.image, target_size))
        for record in tqdm(records, desc="Resizing")
    ]
    resized_rgba = [record.image for record in resized_records]

    split_map = None
    if dataset_name == "pokemon":
        pokemon_ids = [parse_pokemon_id(record.path) for record in resized_records]
        split_map = make_id_splits([pid for pid in pokemon_ids if pid is not None], seed=split_seed)
        (out_dir / "splits.json").write_text(json.dumps(split_map, indent=2, sort_keys=True))
        counts = {split: list(split_map.values()).count(split) for split in ("train", "val", "test")}
        print(f"  Pokemon ID splits: {counts}")

    print("\n[2/5] Extracting palette from opaque pixels...")
    extractor = PaletteExtractor(palette_size=palette_size, alpha_threshold=alpha_threshold)
    extractor.fit(resized_rgba)
    extractor.save(out_dir / "palette.json")
    Image.fromarray(extractor.visualize_palette()).save(out_dir / "palette_swatch.png")

    print("\n[3/5] Quantizing images to token maps...")
    index_maps = []
    alpha_masks = []
    quantized_rgba = []
    for image in tqdm(resized_rgba, desc="Quantizing"):
        index_map, alpha_mask, preview = extractor.quantize_with_transparency(image)
        index_maps.append(index_map)
        alpha_masks.append(alpha_mask)
        quantized_rgba.append(preview)

    index_maps_arr = np.array(index_maps, dtype=np.uint8)
    alpha_masks_arr = np.array(alpha_masks, dtype=bool)
    quantized_rgba_arr = np.array(quantized_rgba, dtype=np.uint8)
    originals_rgba_arr = np.array(resized_rgba, dtype=np.uint8)

    print("\n[4/5] Writing arrays and metadata...")
    np.save(out_dir / "index_maps.npy", index_maps_arr)
    np.save(out_dir / "alpha_masks.npy", alpha_masks_arr)
    np.save(out_dir / "quantized_rgba.npy", quantized_rgba_arr)
    np.save(out_dir / "originals_rgba.npy", originals_rgba_arr)

    manifest = build_manifest(
        dataset_name=dataset_name,
        records=resized_records,
        target_size=target_size,
        palette_size=palette_size,
        alpha_threshold=alpha_threshold,
        split_map=split_map,
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\n[5/5] Saving sample previews...")
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    n_samples = min(20, len(originals_rgba_arr))
    for i in range(n_samples):
        Image.fromarray(originals_rgba_arr[i]).save(samples_dir / f"{i:04d}_original.png")
        Image.fromarray(quantized_rgba_arr[i]).save(samples_dir / f"{i:04d}_quantized.png")

    print(f"  index_maps.npy:    {index_maps_arr.shape}, range=[{index_maps_arr.min()}, {index_maps_arr.max()}]")
    print(f"  alpha_masks.npy:   {alpha_masks_arr.shape}, opaque={alpha_masks_arr.mean():.3f}")
    print(f"  quantized_rgba.npy:{quantized_rgba_arr.shape}")
    print(f"  manifest.json:     {len(manifest['samples'])} samples")
    print(f"  Saved {n_samples} sample comparisons to {samples_dir}")
    return index_maps_arr, alpha_masks_arr, extractor


def discover_datasets(args: argparse.Namespace) -> dict[str, list[ImageRecord]]:
    datasets: dict[str, list[ImageRecord]] = {}

    if args.dataset in ("sprites", "all"):
        sprites_dir = RAW_DIR / "sprites"
        if sprites_dir.exists():
            npy_files = [p for p in sorted(sprites_dir.glob("sprites_*.npy")) if "label" not in p.name]
            for npy_file in npy_files:
                datasets[npy_file.stem] = load_npy_sprites(npy_file, max_images=args.max_images)
            if any(sprites_dir.rglob("*.png")) or any(sprites_dir.rglob("*.jpg")):
                datasets["sprites_images"] = load_images_from_dir(sprites_dir, max_images=args.max_images)
        else:
            print(f"[warn] Sprites directory not found: {sprites_dir}")

    if args.dataset in ("pokemon", "all"):
        pokemon_base = RAW_DIR / "pokemon"
        if pokemon_base.exists():
            records = load_images_from_dir(pokemon_base, max_images=args.max_images)
            if records:
                datasets["pokemon"] = records
        else:
            print(f"[warn] Pokemon directory not found: {pokemon_base}")

    if args.dataset in ("opengameart", "all"):
        oga_dir = RAW_DIR / "opengameart"
        if oga_dir.exists() and any(oga_dir.rglob("*.png")):
            datasets["opengameart"] = load_images_from_dir(oga_dir, max_images=args.max_images)
        else:
            print(f"[info] No OpenGameArt images found in {oga_dir}")

    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess PixelVAR datasets")
    parser.add_argument("--palette-size", type=int, default=16)
    parser.add_argument("--target-size", type=int, default=32)
    parser.add_argument("--alpha-threshold", type=int, default=128)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--dataset",
        choices=["sprites", "pokemon", "opengameart", "all"],
        default="all",
    )
    args = parser.parse_args()

    datasets = discover_datasets(args)
    if not datasets:
        print("\nNo datasets found. Run 'python scripts/download_data.py --dataset pokemon' first.")
        sys.exit(1)

    for name, records in datasets.items():
        preprocess_dataset(
            dataset_name=name,
            records=records,
            palette_size=args.palette_size,
            target_size=args.target_size,
            alpha_threshold=args.alpha_threshold,
            split_seed=args.split_seed,
        )

    print("\n=== All preprocessing complete ===")
    print("Next step: python scripts/check_data.py --dataset pokemon")


if __name__ == "__main__":
    main()
