"""
PyTorch Dataset for PixelVAR.

Loads preprocessed palette-indexed sprite data and provides
multi-scale token maps for VAR-style training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pathlib import Path
from typing import Optional
from PIL import Image

from pixelvar.data.palette import PaletteExtractor


class PixelArtDataset(Dataset):
    """
    Dataset of palette-quantized pixel art sprites.

    Each item returns:
        - index_map: (H, W) tensor of palette indices (long)
        - multi_scale_maps: list of (h_k, w_k) tensors at each scale
        - rgb: (3, H, W) float tensor of the quantized RGB image (for visualization)
    """

    def __init__(
        self,
        processed_dir: str,
        scale_resolutions: list[int] = None,
        return_rgb: bool = True,
    ):
        self.processed_dir = Path(processed_dir)
        self.return_rgb = return_rgb

        if scale_resolutions is None:
            self.scale_resolutions = [1, 2, 4, 8, 16, 32]
        else:
            self.scale_resolutions = scale_resolutions

        # Load data
        self.index_maps = np.load(self.processed_dir / "index_maps.npy")  # (N, 32, 32)

        if return_rgb:
            self.quantized_rgb = np.load(self.processed_dir / "quantized_rgb.npy")  # (N, 32, 32, 3)
        else:
            self.quantized_rgb = None

        # Load palette
        self.palette_extractor = PaletteExtractor()
        self.palette_extractor.load(self.processed_dir / "palette.json")
        self.palette = self.palette_extractor.palette  # (K, 3)
        self.palette_size = len(self.palette)

        print(f"  Loaded {len(self)} samples from {self.processed_dir}")
        print(f"  Palette size: {self.palette_size}")
        print(f"  Scales: {self.scale_resolutions}")

    def __len__(self):
        return len(self.index_maps)

    def _create_multiscale_maps(self, index_map: np.ndarray) -> list[torch.Tensor]:
        """
        Create multi-scale palette index maps by downsampling.

        For each scale resolution r, we create an (r, r) map by
        taking the most frequent palette index in each (32/r, 32/r) block.
        """
        h, w = index_map.shape
        maps = []

        for res in self.scale_resolutions:
            if res == h:
                maps.append(torch.from_numpy(index_map.copy()).long())
            else:
                block_h = h // res
                block_w = w // res
                downsampled = np.zeros((res, res), dtype=np.uint8)

                for i in range(res):
                    for j in range(res):
                        block = index_map[
                            i * block_h:(i + 1) * block_h,
                            j * block_w:(j + 1) * block_w,
                        ]
                        # Mode (most frequent index) in the block
                        values, counts = np.unique(block, return_counts=True)
                        downsampled[i, j] = values[np.argmax(counts)]

                maps.append(torch.from_numpy(downsampled).long())

        return maps

    def __getitem__(self, idx):
        index_map = self.index_maps[idx]  # (32, 32) uint8

        # Multi-scale maps
        multi_scale_maps = self._create_multiscale_maps(index_map)

        # Flatten multi-scale maps into a single token sequence
        # Scale 1x1 -> 2x2 -> 4x4 -> 8x8 -> 16x16 -> 32x32
        # Total tokens: 1 + 4 + 16 + 64 + 256 + 1024 = 1365
        token_sequence = torch.cat([m.flatten() for m in multi_scale_maps])

        result = {
            "index_map": torch.from_numpy(index_map).long(),
            "multi_scale_maps": multi_scale_maps,
            "token_sequence": token_sequence,
        }

        if self.return_rgb and self.quantized_rgb is not None:
            rgb = self.quantized_rgb[idx]  # (32, 32, 3)
            rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            result["rgb"] = rgb

        return result

    @staticmethod
    def collate_fn(batch):
        """Custom collate for variable-size multi-scale maps."""
        result = {
            "index_map": torch.stack([b["index_map"] for b in batch]),
            "token_sequence": torch.stack([b["token_sequence"] for b in batch]),
        }

        # Stack multi-scale maps per scale
        n_scales = len(batch[0]["multi_scale_maps"])
        result["multi_scale_maps"] = [
            torch.stack([b["multi_scale_maps"][s] for b in batch])
            for s in range(n_scales)
        ]

        if "rgb" in batch[0]:
            result["rgb"] = torch.stack([b["rgb"] for b in batch])

        return result


def get_dataloader(
    processed_dir: str,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    scale_resolutions: list[int] = None,
) -> DataLoader:
    """Create a DataLoader for processed pixel art data."""
    dataset = PixelArtDataset(
        processed_dir=processed_dir,
        scale_resolutions=scale_resolutions,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=PixelArtDataset.collate_fn,
        pin_memory=True,
    )


def get_combined_dataloader(
    processed_dirs: list[str],
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """Create a DataLoader combining multiple processed datasets."""
    datasets = [PixelArtDataset(d) for d in processed_dirs]
    combined = ConcatDataset(datasets)
    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=PixelArtDataset.collate_fn,
        pin_memory=True,
    )
