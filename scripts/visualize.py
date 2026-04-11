#!/usr/bin/env python3
"""
Visualization utilities for PixelVAR.

Generates comparison grids, palette swatches, and multi-scale visualizations.

Usage:
    python scripts/visualize.py --dataset pokemon
    python scripts/visualize.py --dataset sprites_10000_28x28
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pixelvar.data.palette import PaletteExtractor


PROCESSED_DIR = Path("data/processed")
VIS_DIR = Path("visualizations")


def plot_palette(palette: np.ndarray, save_path: Path = None):
    """Plot palette as a color swatch with hex codes."""
    n = len(palette)
    fig, ax = plt.subplots(1, 1, figsize=(max(8, n * 0.8), 1.5))
    for i, color in enumerate(palette):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color / 255.0))
        hex_color = "#{:02x}{:02x}{:02x}".format(*color)
        luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        text_color = "white" if luminance < 128 else "black"
        ax.text(i + 0.5, 0.5, f"{i}", ha="center", va="center",
                fontsize=8, color=text_color, fontweight="bold")

    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Extracted Palette ({n} colors)", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved palette plot: {save_path}")
    plt.close()


def plot_original_vs_quantized(
    originals: np.ndarray,
    quantized: np.ndarray,
    n_samples: int = 10,
    save_path: Path = None,
):
    """Plot side-by-side comparison of original vs palette-quantized sprites."""
    n = min(n_samples, len(originals))
    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3.5))

    if n == 1:
        axes = axes.reshape(2, 1)

    for i in range(n):
        axes[0, i].imshow(originals[i], interpolation="nearest")
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("Original", fontsize=10, fontweight="bold")

        axes[1, i].imshow(quantized[i], interpolation="nearest")
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("Quantized", fontsize=10, fontweight="bold")

    fig.suptitle("Original vs Palette-Quantized Sprites", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved comparison plot: {save_path}")
    plt.close()


def plot_multiscale(
    index_map: np.ndarray,
    palette: np.ndarray,
    scales: list[int] = None,
    save_path: Path = None,
):
    """Visualize the multi-scale decomposition of a single sprite."""
    if scales is None:
        scales = [1, 2, 4, 8, 16, 32]

    h, w = index_map.shape
    fig, axes = plt.subplots(1, len(scales), figsize=(len(scales) * 2, 2.5))

    for si, res in enumerate(scales):
        if res == h:
            scale_map = index_map
        else:
            block_h = h // res
            block_w = w // res
            scale_map = np.zeros((res, res), dtype=np.uint8)
            for i in range(res):
                for j in range(res):
                    block = index_map[
                        i * block_h:(i + 1) * block_h,
                        j * block_w:(j + 1) * block_w,
                    ]
                    values, counts = np.unique(block, return_counts=True)
                    scale_map[i, j] = values[np.argmax(counts)]

        # Convert indices to RGB
        rgb = palette[scale_map]
        axes[si].imshow(rgb, interpolation="nearest")
        axes[si].set_title(f"{res}x{res}", fontsize=10)
        axes[si].axis("off")

    fig.suptitle("Multi-Scale Decomposition (Coarse → Fine)", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved multi-scale plot: {save_path}")
    plt.close()


def plot_index_map_heatmap(
    index_map: np.ndarray,
    palette: np.ndarray,
    save_path: Path = None,
):
    """Show the palette index map as a heatmap alongside the RGB rendering."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

    ax1.imshow(index_map, cmap="tab20", interpolation="nearest", vmin=0, vmax=len(palette) - 1)
    ax1.set_title("Palette Index Map", fontsize=10)
    ax1.axis("off")

    rgb = palette[index_map]
    ax2.imshow(rgb, interpolation="nearest")
    ax2.set_title("RGB Rendering", fontsize=10)
    ax2.axis("off")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved index map plot: {save_path}")
    plt.close()


def plot_dataset_stats(index_maps: np.ndarray, palette_size: int, save_path: Path = None):
    """Plot statistics about the dataset."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Palette index distribution
    flat = index_maps.flatten()
    ax1.hist(flat, bins=np.arange(palette_size + 1) - 0.5, edgecolor="black", alpha=0.7)
    ax1.set_xlabel("Palette Index")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Palette Index Distribution")
    ax1.set_xticks(range(palette_size))

    # Unique colors per image
    unique_per_img = [len(np.unique(m)) for m in index_maps]
    ax2.hist(unique_per_img, bins=range(1, palette_size + 2), edgecolor="black", alpha=0.7)
    ax2.set_xlabel("# Unique Colors")
    ax2.set_ylabel("# Images")
    ax2.set_title("Colors Used per Sprite")

    plt.suptitle("Dataset Statistics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved stats plot: {save_path}")
    plt.close()


def visualize_dataset(dataset_name: str):
    """Generate all visualizations for a processed dataset."""
    data_dir = PROCESSED_DIR / dataset_name
    vis_dir = VIS_DIR / dataset_name
    vis_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print(f"[error] Processed data not found: {data_dir}")
        print(f"  Run: python scripts/preprocess_data.py --dataset {dataset_name}")
        return

    print(f"\n=== Visualizing: {dataset_name} ===")

    # Load data
    index_maps = np.load(data_dir / "index_maps.npy")
    quantized = np.load(data_dir / "quantized_rgb.npy")
    originals = np.load(data_dir / "originals_rgb.npy")

    extractor = PaletteExtractor()
    extractor.load(data_dir / "palette.json")
    palette = extractor.palette

    print(f"  Loaded {len(index_maps)} images, palette size: {len(palette)}")

    # 1. Palette
    plot_palette(palette, vis_dir / "palette.png")

    # 2. Original vs Quantized
    plot_original_vs_quantized(originals, quantized, n_samples=10, save_path=vis_dir / "comparison.png")

    # 3. Multi-scale decomposition (for a few samples)
    for i in range(min(5, len(index_maps))):
        plot_multiscale(index_maps[i], palette, save_path=vis_dir / f"multiscale_{i:03d}.png")

    # 4. Index map visualization
    for i in range(min(3, len(index_maps))):
        plot_index_map_heatmap(index_maps[i], palette, save_path=vis_dir / f"indexmap_{i:03d}.png")

    # 5. Dataset statistics
    plot_dataset_stats(index_maps, len(palette), save_path=vis_dir / "stats.png")

    print(f"\n  All visualizations saved to: {vis_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize PixelVAR processed data")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name to visualize")
    args = parser.parse_args()

    if args.dataset:
        visualize_dataset(args.dataset)
    else:
        # Visualize all processed datasets
        if not PROCESSED_DIR.exists():
            print("No processed data found. Run preprocess_data.py first.")
            return
        for d in sorted(PROCESSED_DIR.iterdir()):
            if d.is_dir() and (d / "index_maps.npy").exists():
                visualize_dataset(d.name)


if __name__ == "__main__":
    main()
