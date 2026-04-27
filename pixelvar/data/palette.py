"""Palette extraction, quantization, and rendering for PixelVAR tokens."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans


class PaletteExtractor:
    """Extract and manage color palettes for pixel art."""

    def __init__(self, palette_size: int = 16, random_state: int = 42, alpha_threshold: int = 128):
        self.palette_size = palette_size
        self.random_state = random_state
        self.alpha_threshold = alpha_threshold
        self.palette: Optional[np.ndarray] = None  # (palette_size, 3) uint8

    def fit(self, images: list[np.ndarray], max_pixels: int = 500_000) -> "PaletteExtractor":
        """
        Extract a shared palette from a collection of images using k-means.

        Args:
            images: List of numpy arrays (H, W, 3) in uint8.
            max_pixels: Maximum number of opaque pixel samples for k-means.
        """
        all_pixels = []
        for img in images:
            img = np.asarray(img)
            if img.ndim == 2:
                pixels = np.stack([img] * 3, axis=-1).reshape(-1, 3)
            elif img.shape[-1] == 4:
                alpha = img[:, :, 3]
                rgb = img[:, :, :3]
                pixels = rgb[alpha >= self.alpha_threshold]
            else:
                pixels = img[:, :, :3].reshape(-1, 3)

            if len(pixels) > 0:
                all_pixels.append(pixels.astype(np.uint8))

        if not all_pixels:
            raise ValueError("No opaque pixels found for palette extraction")

        all_pixels = np.concatenate(all_pixels, axis=0)
        print(f"  Total pixels collected: {len(all_pixels):,}")

        # Subsample if too many
        if len(all_pixels) > max_pixels:
            rng = np.random.RandomState(self.random_state)
            indices = rng.choice(len(all_pixels), max_pixels, replace=False)
            all_pixels = all_pixels[indices]
            print(f"  Subsampled to {max_pixels:,} pixels for k-means")

        unique_pixels = np.unique(all_pixels, axis=0)
        if len(unique_pixels) <= self.palette_size:
            palette = unique_pixels
            if len(palette) < self.palette_size:
                pad = np.repeat(palette[-1:], self.palette_size - len(palette), axis=0)
                palette = np.concatenate([palette, pad], axis=0)
            self.palette = self._sort_by_luminance(palette.astype(np.uint8))
            print(f"  Palette extracted from {len(unique_pixels)} unique colors without k-means")
            return self

        print(f"  Running MiniBatchKMeans with k={self.palette_size}...")
        kmeans = MiniBatchKMeans(
            n_clusters=self.palette_size,
            random_state=self.random_state,
            batch_size=1024,
            n_init=3,
        )
        kmeans.fit(all_pixels.astype(np.float32))

        self.palette = self._sort_by_luminance(np.round(kmeans.cluster_centers_).astype(np.uint8))

        print(f"  Palette extracted: {self.palette.shape}")
        return self

    def quantize(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quantize an RGB image to 0-indexed palette IDs.

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

    def quantize_with_transparency(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Quantize an RGB/RGBA image to PixelVAR token IDs.

        Returns:
            index_map: ``(H, W)`` uint8 with 0 transparent and 1..K palette tokens.
            alpha_mask: ``(H, W)`` bool, True for opaque pixels.
            quantized_rgba: ``(H, W, 4)`` uint8 preview preserving transparency.
        """
        assert self.palette is not None, "Call fit() first"
        rgba = self._ensure_rgba(image)
        h, w = rgba.shape[:2]
        alpha_mask = rgba[:, :, 3] >= self.alpha_threshold

        index_map = np.zeros((h, w), dtype=np.uint8)
        quantized_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        if alpha_mask.any():
            rgb = rgba[:, :, :3]
            opaque_pixels = rgb[alpha_mask].astype(np.float32)
            palette_f = self.palette.astype(np.float32)
            dists = np.sum((opaque_pixels[:, None, :] - palette_f[None, :, :]) ** 2, axis=-1)
            nearest = np.argmin(dists, axis=-1).astype(np.uint8)
            index_map[alpha_mask] = nearest + 1
            quantized_rgba[alpha_mask, :3] = self.palette[nearest]
            quantized_rgba[alpha_mask, 3] = 255
        return index_map, alpha_mask, quantized_rgba

    def render_index_map(self, index_map: np.ndarray, alpha_mask: np.ndarray | None = None) -> np.ndarray:
        """Render PixelVAR token IDs to an RGBA image."""
        assert self.palette is not None, "Load or fit a palette first"
        tokens = np.asarray(index_map)
        if tokens.ndim != 2:
            raise ValueError(f"index_map must be 2D, got {tokens.shape}")
        if tokens.min() < 0 or tokens.max() > self.palette_size:
            raise ValueError(f"token range [{tokens.min()}, {tokens.max()}] outside [0, {self.palette_size}]")

        if alpha_mask is None:
            alpha_mask = tokens != 0
        alpha_mask = np.asarray(alpha_mask, dtype=bool)

        rgba = np.zeros((*tokens.shape, 4), dtype=np.uint8)
        opaque = alpha_mask & (tokens > 0)
        if opaque.any():
            rgba[opaque, :3] = self.palette[tokens[opaque] - 1]
            rgba[opaque, 3] = 255
        return rgba

    def save(self, path: Path):
        """Save palette to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "palette_size": self.palette_size,
            "transparent_token": 0,
            "palette_token_start": 1,
            "palette_token_end": self.palette_size,
            "alpha_threshold": self.alpha_threshold,
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
        self.alpha_threshold = data.get("alpha_threshold", self.alpha_threshold)
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

    @staticmethod
    def _sort_by_luminance(palette: np.ndarray) -> np.ndarray:
        luminance = 0.299 * palette[:, 0] + 0.587 * palette[:, 1] + 0.114 * palette[:, 2]
        return palette[np.argsort(luminance)]

    @staticmethod
    def _ensure_rgba(image: np.ndarray) -> np.ndarray:
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
        raise ValueError(f"Unsupported image shape: {img.shape}")
