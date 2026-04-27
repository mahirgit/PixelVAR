"""Rendering helpers for PixelVAR samples."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pixelvar.data.palette import PaletteExtractor
from pixelvar.tokenizers import DeterministicPyramidTokenizer


def tokens_to_rgba(
    token_sequence: torch.Tensor | np.ndarray,
    palette: PaletteExtractor,
    scale_resolutions: list[int] | None = None,
) -> np.ndarray:
    """Render one or more token sequences to RGBA arrays."""
    tokenizer = DeterministicPyramidTokenizer(scale_resolutions or [1, 2, 4, 8, 16, 32])
    sequence = torch.as_tensor(token_sequence, dtype=torch.long)
    squeeze = False
    if sequence.ndim == 1:
        sequence = sequence.unsqueeze(0)
        squeeze = True

    maps = tokenizer.from_sequence(sequence)
    final_maps = maps[-1].cpu().numpy()
    images = np.stack([palette.render_index_map(index_map) for index_map in final_maps], axis=0)
    return images[0] if squeeze else images


def save_rgba_grid(images: np.ndarray, path: str | Path, columns: int = 4, scale: int = 4) -> None:
    """Save a transparent RGBA image grid."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if images.ndim == 3:
        images = images[None, ...]
    n, h, w, c = images.shape
    if c != 4:
        raise ValueError(f"Expected RGBA images, got shape {images.shape}")

    columns = max(1, min(columns, n))
    rows = int(np.ceil(n / columns))
    grid = np.zeros((rows * h, columns * w, 4), dtype=np.uint8)
    for idx, image in enumerate(images):
        row, col = divmod(idx, columns)
        grid[row * h : (row + 1) * h, col * w : (col + 1) * w] = image

    pil = Image.fromarray(grid, mode="RGBA")
    if scale != 1:
        pil = pil.resize((grid.shape[1] * scale, grid.shape[0] * scale), Image.NEAREST)
    pil.save(path)
