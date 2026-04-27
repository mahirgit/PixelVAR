#!/usr/bin/env python3
"""
Download datasets for PixelVAR.

Datasets:
  1. Sprites dataset (~170K frames) - from a public source
  2. Pokemon sprites (~8K) - from Kaggle
  3. OpenGameArt curated assets - manual curation guide

Usage:
    python scripts/download_data.py --dataset all
    python scripts/download_data.py --dataset sprites
    python scripts/download_data.py --dataset pokemon
"""

import argparse
import os
import zipfile
import tarfile
import shutil
from pathlib import Path

import requests
from tqdm import tqdm


RAW_DIR = Path("data/raw")


def download_file(url: str, dest: Path, desc: str = "Downloading"):
    """Download a file with progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] {dest} already exists")
        return

    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=desc
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def extract_archive(archive_path: Path, extract_to: Path):
    """Extract zip or tar archive."""
    extract_to.mkdir(parents=True, exist_ok=True)
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_to)
    elif archive_path.suffix in (".gz", ".tgz"):
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(extract_to)
    elif archive_path.suffix == ".tar":
        with tarfile.open(archive_path, "r") as tf:
            tf.extractall(extract_to)
    print(f"  Extracted to {extract_to}")


def download_sprites():
    """
    Download the sprites dataset.
    Uses the popular "Sprite Dataset" with ~170K character frames.
    Source: https://github.com/YingzhenLi/Sprites (or Kaggle mirrors).
    """
    print("\n=== Sprites Dataset ===")
    out_dir = RAW_DIR / "sprites"
    out_dir.mkdir(parents=True, exist_ok=True)

    # The sprites dataset from Yingzhen Li's repo (numpy arrays)
    base_url = "https://github.com/YingzhenLi/Sprites/raw/master"
    files = [
        "sprites_1788_16x16.npy",
        "sprite_labels_nc_1788_16x16.npy",
        "sprites_10000_28x28.npy",
        "sprite_labels_nc_10000_28x28.npy",
    ]

    for fname in files:
        url = f"{base_url}/{fname}"
        dest = out_dir / fname
        try:
            download_file(url, dest, desc=fname)
        except Exception as e:
            print(f"  [warn] Could not download {fname}: {e}")

    # Alternative: Kaggle sprite dataset (much larger, ~170K)
    # Requires kaggle API credentials in ~/.kaggle/kaggle.json
    print("\n  For the full 170K sprites dataset, run:")
    print("    kaggle datasets download -d brentspell/sprites-dataset -p data/raw/sprites/")
    print("  Or download manually from Kaggle and place in data/raw/sprites/")

    print("  Sprites download step complete.")


def download_pokemon():
    """
    Download Pokemon sprite dataset (~8K sprites).
    """
    print("\n=== Pokemon Sprites Dataset ===")
    out_dir = RAW_DIR / "pokemon"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pokemon sprites from a well-known Kaggle dataset
    print("  Option 1 - Kaggle (recommended, ~8K sprites):")
    print("    kaggle datasets download -d kvpratama/pokemon-images-dataset -p data/raw/pokemon/")
    print("    Then unzip in data/raw/pokemon/")

    # Alternative: PokeAPI sprites (smaller but freely accessible)
    print("\n  Option 2 - Downloading from PokeAPI (front/back/shiny variants for first 905 Pokemon)...")
    sprites_dir = out_dir / "sprites"
    sprites_dir.mkdir(parents=True, exist_ok=True)

    existing = list(sprites_dir.rglob("*.png"))
    if len(existing) >= 3000:
        print(f"  [skip] Already have {len(existing)} sprites")
        return

    base_url = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon"
    variants = {
        "front": ("", sprites_dir),
        "back": ("back", sprites_dir / "back"),
        "shiny": ("shiny", sprites_dir / "shiny"),
        "shiny_back": ("back/shiny", sprites_dir / "back" / "shiny"),
    }
    for _, dest_dir in variants.values():
        dest_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for pokemon_id in tqdm(range(1, 906), desc="Pokemon sprites"):
        for variant_path, dest_dir in variants.values():
            dest = dest_dir / f"{pokemon_id}.png"
            if dest.exists():
                downloaded += 1
                continue
            url = f"{base_url}/{variant_path}/{pokemon_id}.png" if variant_path else f"{base_url}/{pokemon_id}.png"
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    with open(dest, "wb") as f:
                        f.write(resp.content)
                    downloaded += 1
            except Exception:
                pass

    print(f"  Downloaded {downloaded} Pokemon sprites")


def download_opengameart():
    """Provide instructions for OpenGameArt curation."""
    print("\n=== OpenGameArt Assets ===")
    out_dir = RAW_DIR / "opengameart"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("  OpenGameArt requires manual curation. Recommended packs:")
    print("  - https://opengameart.org/content/16x16-dungeon-tileset")
    print("  - https://opengameart.org/content/roguelike-characters")
    print("  - https://opengameart.org/content/tiny-16-basic")
    print("  - https://opengameart.org/content/pixel-art-character-sprites")
    print(f"\n  Download and place sprite sheets/images in: {out_dir}")
    print("  The preprocessing pipeline will handle sheet splitting if needed.")


def main():
    parser = argparse.ArgumentParser(description="Download PixelVAR datasets")
    parser.add_argument(
        "--dataset",
        choices=["sprites", "pokemon", "opengameart", "all"],
        default="all",
        help="Which dataset to download",
    )
    args = parser.parse_args()

    print(f"Download directory: {RAW_DIR.resolve()}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("sprites", "all"):
        download_sprites()
    if args.dataset in ("pokemon", "all"):
        download_pokemon()
    if args.dataset in ("opengameart", "all"):
        download_opengameart()

    print("\n=== Done ===")
    print("Next step: python scripts/preprocess_data.py")


if __name__ == "__main__":
    main()
