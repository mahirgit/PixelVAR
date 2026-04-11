"""
Palette extraction and quantization for pixel art sprites.

Extracts a fixed-size palette from a dataset of images using k-means clustering,
then quantizes all pixels to their nearest palette entry.
"""

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans, MiniBatchKMeans
from pathlib import Path
from typing import Tuple, Optional
import json


class PaletteExtractor:
    """Extract and manage color palettes for pixel art."""

    def __init__(self, palette_size: int = 16, random_state: int = 42):
        self.palette_size = palette_size
        self.random_state = random_state
        self.palette: Optional[np.ndarray] = None  # (palette_size, 3) uint8

    def fit(self, images: list[np.ndarray], max_pixels: int = 500_000) -> "PaletteExtractor":
        """
        Extract a shared palette from a collection of images using k-means.

        Args:
            images: List of numpy arrays (H, W, 3) in uint8.
            max_pixels: Maximum number of pixel samples for k-means (for speed).
        """
        # Collect all pixels
        all_pixels = []
        for img in images:
            if img.ndim == 2:
                # Grayscale -> RGB
                img = np.stack([img] * 3, axis=-1)
            if img.shape[-1] == 4:
                # RGBA -> RGB (ignore transparent pixels)
                alpha = img[:, :, 3]
                rgb = img[:, :, :3]
                mask = alpha > 128
                pixels = rgb[mask]
            else:
                pixels = img.reshape(-1, 3)
            
            # Filter out near-white background pixels to avoid wasting palette slots
            # White background (from RGBA compositing) dominates k-means otherwise
            if len(pixels) > 0:
                luminance = 0.299 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 2]
                fg_mask = luminance < 245  # Keep non-white pixels
                if fg_mask.sum() > 0:
                    pixels = pixels[fg_mask]
            all_pixels.append(pixels)

        all_pixels = np.concatenate(all_pixels, axis=0)
        print(f"  Total pixels collected: {len(all_pixels):,}")

        # Subsample if too many
        if len(all_pixels) > max_pixels:
            rng = np.random.RandomState(self.random_state)
            indices = rng.choice(len(all_pixels), max_pixels, replace=False)
            all_pixels = all_pixels[indices]
            print(f"  Subsampled to {max_pixels:,} pixels for k-means")

        # Run k-means
        print(f"  Running MiniBatchKMeans with k={self.palette_size}...")
        kmeans = MiniBatchKMeans(
            n_clusters=self.palette_size,
            random_state=self.random_state,
            batch_size=1024,
            n_init=3,
        )
        kmeans.fit(all_pixels.astype(np.float32))

        self.palette = np.round(kmeans.cluster_centers_).astype(np.uint8)
        # Sort palette by luminance for consistency
        luminance = 0.299 * self.palette[:, 0] + 0.587 * self.palette[:, 1] + 0.114 * self.palette[:, 2]
        sort_idx = np.argsort(luminance)
        self.palette = self.palette[sort_idx]

        print(f"  Palette extracted: {self.palette.shape}")
        return self

    def quantize(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quantize an image to the extracted palette.

        Args:
            image: (H, W, 3) uint8 array.

        Returns:
            index_map: (H, W) array of palette indices (uint8).
            quantized: (H, W, 3) uint8 array with palette colors.
        """
        assert self.palette is not None, "Call fit() first"
        h, w = image.shape[:2]

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        pixels = image[:, :, :3].reshape(-1, 3).astype(np.float32)
        palette_f = self.palette.astype(np.float32)

        # Compute distances to each palette entry
        # (N, 1, 3) - (1, K, 3) -> (N, K)
        dists = np.sum((pixels[:, None, :] - palette_f[None, :, :]) ** 2, axis=-1)
        indices = np.argmin(dists, axis=-1).astype(np.uint8)

        index_map = indices.reshape(h, w)
        quantized = self.palette[indices].reshape(h, w, 3)
        return index_map, quantized

    def save(self, path: Path):
        """Save palette to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "palette_size": self.palette_size,
            "colors": self.palette.tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Palette saved to {path}")

    def load(self, path: Path) -> "PaletteExtractor":
        """Load palette from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        self.palette_size = data["palette_size"]
        self.palette = np.array(data["colors"], dtype=np.uint8)
        return self

    def visualize_palette(self) -> np.ndarray:
        """Create a visualization of the palette as a color swatch image."""
        assert self.palette is not None
        swatch_w = 32
        swatch_h = 32
        cols = min(8, self.palette_size)
        rows = (self.palette_size + cols - 1) // cols
        img = np.zeros((rows * swatch_h, cols * swatch_w, 3), dtype=np.uint8)
        for i, color in enumerate(self.palette):
            r, c = divmod(i, cols)
            img[r * swatch_h:(r + 1) * swatch_h, c * swatch_w:(c + 1) * swatch_w] = color
        return img
