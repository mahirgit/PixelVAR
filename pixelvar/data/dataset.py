"""
PyTorch Dataset for PixelVAR.

Loads preprocessed palette-indexed sprite data and provides
multi-scale token maps for VAR-style training.

Uses the tokenizer interface (see pixelvar/tokenizers/) for multi-scale
encoding. See docs/design_contract.md for tensor shapes and vocabulary.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pathlib import Path
from typing import Optional

from pixelvar.data.palette import PaletteExtractor
from pixelvar.tokenizers import DeterministicTokenizer, BaseTokenizer


class PixelArtDataset(Dataset):
    """
    Dataset of palette-quantized pixel art sprites.

    Each item returns:
        - index_map: (H, W) tensor of palette indices (long)
        - multi_scale_maps: list of (h_k, w_k) tensors at each scale
        - token_sequence: (1365,) flat token sequence
        - rgb: (3, H, W) float tensor of the quantized RGB image (for visualization)
    """

    def __init__(
        self,
        processed_dir: str,
        tokenizer: BaseTokenizer = None,
        return_rgb: bool = True,
        split: str = None,
        split_file: str = None,
    ):
        self.processed_dir = Path(processed_dir)
        self.return_rgb = return_rgb

        # Load palette
        self.palette_extractor = PaletteExtractor()
        self.palette_extractor.load(self.processed_dir / "palette.json")
        self.palette = self.palette_extractor.palette  # (K, 3)
        self.palette_size = len(self.palette)

        # Initialize tokenizer (default: deterministic pyramid)
        if tokenizer is None:
            self.tokenizer = DeterministicTokenizer(palette_size=self.palette_size)
        else:
            self.tokenizer = tokenizer

        # Load data
        self.index_maps = np.load(self.processed_dir / "index_maps.npy")  # (N, 32, 32)

        if return_rgb:
            self.quantized_rgb = np.load(self.processed_dir / "quantized_rgb.npy")  # (N, 32, 32, 3)
        else:
            self.quantized_rgb = None

        # Apply split filter if provided
        if split is not None and split_file is not None:
            self._apply_split(split, split_file)

        print(f"  Loaded {len(self)} samples from {self.processed_dir}")
        print(f"  Palette size: {self.palette_size}")
        print(f"  Tokenizer: {self.tokenizer.__class__.__name__}")
        print(f"  Scales: {self.tokenizer.scale_resolutions}")

    def _apply_split(self, split: str, split_file: str):
        """Filter samples based on a split JSON file."""
        import json

        with open(split_file) as f:
            split_map = json.load(f)

        # Get indices that belong to this split
        keep = []
        for idx in range(len(self.index_maps)):
            # The split_map maps asset_id -> split_name
            # For now, we assume sequential ordering matches the manifest
            asset_id = str(idx)
            if split_map.get(asset_id, "") == split:
                keep.append(idx)

        if keep:
            keep = np.array(keep)
            self.index_maps = self.index_maps[keep]
            if self.quantized_rgb is not None:
                self.quantized_rgb = self.quantized_rgb[keep]

    def __len__(self):
        return len(self.index_maps)

    def __getitem__(self, idx):
        index_map = self.index_maps[idx]  # (32, 32) uint8
        index_tensor = torch.from_numpy(index_map.copy()).long().unsqueeze(0)  # (1, 32, 32)

        # Use tokenizer for multi-scale encoding
        multi_scale_maps = self.tokenizer.encode(index_tensor)
        # Remove batch dim for single sample: (1, h, w) -> (h, w)
        multi_scale_maps = [m.squeeze(0) for m in multi_scale_maps]

        # Flatten to token sequence
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
