#!/usr/bin/env python3
"""
Preprocess datasets for PixelVAR.

1. Load raw images from data/raw/
2. Resize to 32x32 using nearest-neighbor interpolation
3. Extract a shared 16-color palette via k-means
4. Quantize all images to palette indices
5. Save processed data to data/processed/

Usage:
    python scripts/preprocess_data.py
    python scripts/preprocess_data.py --palette-size 32
    python scripts/preprocess_data.py --dataset pokemon
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pixelvar.data.palette import PaletteExtractor


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def load_images_from_dir(img_dir: Path, max_images: int = None) -> list[np.ndarray]:
    """Load all images from a directory."""
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
    paths = sorted([
        p for p in img_dir.rglob("*")
        if p.suffix.lower() in extensions
    ])
    if max_images:
        paths = paths[:max_images]

    images = []
    for p in tqdm(paths, desc=f"Loading {img_dir.name}"):
        try:
            img = Image.open(p).convert("RGBA")
            images.append(np.array(img))
        except Exception as e:
            print(f"  [warn] Skipping {p}: {e}")
    return images


def load_npy_sprites(npy_path: Path) -> list[np.ndarray]:
    """Load sprites from a numpy file."""
    data = np.load(npy_path)
    print(f"  Loaded {npy_path.name}: shape={data.shape}, dtype={data.dtype}")

    images = []
    if data.ndim == 3:
        # (N, H, W) grayscale or (N, H*W) flattened
        for i in range(len(data)):
            img = data[i]
            if img.ndim == 1:
                side = int(np.sqrt(len(img)))
                img = img.reshape(side, side)
            images.append(img)
    elif data.ndim == 4:
        # (N, H, W, C) or (N, C, H, W)
        for i in range(len(data)):
            img = data[i]
            if img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
                img = np.transpose(img, (1, 2, 0))
            images.append(img)
    return images


def resize_image(img: np.ndarray, target_size: int = 32) -> np.ndarray:
    """Resize image to target_size x target_size using nearest-neighbor."""
    if img.ndim == 2:
        pil_img = Image.fromarray(img, mode="L")
    elif img.shape[-1] == 4:
        pil_img = Image.fromarray(img, mode="RGBA")
    else:
        pil_img = Image.fromarray(img, mode="RGB")

    pil_img = pil_img.resize((target_size, target_size), Image.NEAREST)
    return np.array(pil_img)


def ensure_rgb(img: np.ndarray) -> np.ndarray:
    """Convert image to RGB, handling transparency by compositing onto white."""
    if img.ndim == 2:
        return np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb = img[:, :, :3].astype(np.float32)
        bg = np.full_like(rgb, 255.0)  # white background
        composited = rgb * alpha + bg * (1 - alpha)
        return composited.astype(np.uint8)
    return img[:, :, :3]


def preprocess_dataset(
    dataset_name: str,
    images: list[np.ndarray],
    palette_size: int = 16,
    target_size: int = 32,
):
    """Full preprocessing pipeline for a single dataset."""
    out_dir = PROCESSED_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Processing: {dataset_name} ({len(images)} images)")
    print(f"  Target size: {target_size}x{target_size}")
    print(f"  Palette size: {palette_size}")
    print(f"{'='*50}")

    # Step 1: Resize
    print("\n[1/4] Resizing images...")
    resized = []
    for img in tqdm(images, desc="Resizing"):
        r = resize_image(img, target_size)
        r = ensure_rgb(r)
        resized.append(r)

    # Step 2: Extract palette
    print("\n[2/4] Extracting palette...")
    extractor = PaletteExtractor(palette_size=palette_size)
    extractor.fit(resized)
    extractor.save(out_dir / "palette.json")

    # Save palette visualization
    palette_vis = extractor.visualize_palette()
    Image.fromarray(palette_vis).save(out_dir / "palette_swatch.png")

    # Step 3: Quantize
    print("\n[3/4] Quantizing images to palette...")
    index_maps = []
    quantized_imgs = []
    for img in tqdm(resized, desc="Quantizing"):
        idx_map, quant_img = extractor.quantize(img)
        index_maps.append(idx_map)
        quantized_imgs.append(quant_img)

    index_maps = np.array(index_maps, dtype=np.uint8)       # (N, 32, 32)
    quantized_imgs = np.array(quantized_imgs, dtype=np.uint8)  # (N, 32, 32, 3)
    originals = np.array(resized, dtype=np.uint8)            # (N, 32, 32, 3)

    # Step 4: Save
    print("\n[4/4] Saving processed data...")
    np.save(out_dir / "index_maps.npy", index_maps)
    np.save(out_dir / "quantized_rgb.npy", quantized_imgs)
    np.save(out_dir / "originals_rgb.npy", originals)

    print(f"  index_maps.npy:   {index_maps.shape} (palette indices)")
    print(f"  quantized_rgb.npy: {quantized_imgs.shape} (palette-colored images)")
    print(f"  originals_rgb.npy: {originals.shape} (resized originals)")
    print(f"  palette.json:      {palette_size} colors")

    # Save a few sample comparisons
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    n_samples = min(20, len(originals))
    for i in range(n_samples):
        Image.fromarray(originals[i]).save(samples_dir / f"{i:04d}_original.png")
        Image.fromarray(quantized_imgs[i]).save(samples_dir / f"{i:04d}_quantized.png")

    print(f"  Saved {n_samples} sample comparisons to {samples_dir}")
    return index_maps, quantized_imgs, extractor


def main():
    parser = argparse.ArgumentParser(description="Preprocess PixelVAR datasets")
    parser.add_argument("--palette-size", type=int, default=16)
    parser.add_argument("--target-size", type=int, default=32)
    parser.add_argument(
        "--dataset",
        choices=["sprites", "pokemon", "opengameart", "all"],
        default="all",
    )
    args = parser.parse_args()

    datasets = {}

    # Load sprites
    if args.dataset in ("sprites", "all"):
        sprites_dir = RAW_DIR / "sprites"
        if sprites_dir.exists():
            # Try numpy files first
            npy_files = list(sprites_dir.glob("sprites_*.npy"))
            if npy_files:
                for npy_f in npy_files:
                    if "label" not in npy_f.name:
                        imgs = load_npy_sprites(npy_f)
                        name = npy_f.stem
                        datasets[name] = imgs
            # Then try image files
            img_files = list(sprites_dir.rglob("*.png")) + list(sprites_dir.rglob("*.jpg"))
            if img_files:
                datasets["sprites_images"] = load_images_from_dir(sprites_dir)
        else:
            print(f"[warn] Sprites directory not found: {sprites_dir}")

    # Load Pokemon
    if args.dataset in ("pokemon", "all"):
        pokemon_dir = RAW_DIR / "pokemon" / "sprites"
        if not pokemon_dir.exists():
            pokemon_dir = RAW_DIR / "pokemon"
        if pokemon_dir.exists():
            imgs = load_images_from_dir(pokemon_dir)
            if imgs:
                datasets["pokemon"] = imgs
        else:
            print(f"[warn] Pokemon directory not found: {pokemon_dir}")

    # Load OpenGameArt
    if args.dataset in ("opengameart", "all"):
        oga_dir = RAW_DIR / "opengameart"
        if oga_dir.exists() and any(oga_dir.rglob("*.png")):
            datasets["opengameart"] = load_images_from_dir(oga_dir)
        else:
            print(f"[info] No OpenGameArt images found in {oga_dir}")

    if not datasets:
        print("\nNo datasets found! Run 'python scripts/download_data.py' first.")
        return

    # Process each dataset
    for name, images in datasets.items():
        preprocess_dataset(
            dataset_name=name,
            images=images,
            palette_size=args.palette_size,
            target_size=args.target_size,
        )

    print("\n=== All preprocessing complete ===")
    print("Next step: Use pixelvar/data/dataset.py for training")


if __name__ == "__main__":
    main()
