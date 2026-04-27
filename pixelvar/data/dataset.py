"""PyTorch datasets for preprocessed PixelVAR sprites."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from pixelvar.data.palette import PaletteExtractor
from pixelvar.tokenizers import DeterministicPyramidTokenizer


class PixelArtDataset(Dataset):
    """
    Dataset of transparency-aware palette token maps.

    Each item returns:
        - ``index_map``: ``(32, 32)`` long, values in ``[0, palette_size]``
        - ``alpha_mask``: ``(32, 32)`` bool, True for opaque pixels
        - ``multi_scale_maps``: list of deterministic pyramid maps
        - ``token_sequence``: ``(1365,)`` long, coarse-to-fine tokens
        - ``rgb_preview``: optional ``(3, 32, 32)`` float for visualization only
    """

    def __init__(
        self,
        processed_dir: str | Path,
        split: Optional[str] = None,
        scale_resolutions: Optional[list[int]] = None,
        return_rgb: bool = True,
        max_samples: Optional[int] = None,
    ):
        self.processed_dir = Path(processed_dir)
        self.split = split
        self.return_rgb = return_rgb
        self.tokenizer = DeterministicPyramidTokenizer(scale_resolutions or [1, 2, 4, 8, 16, 32])

        index_path = self.processed_dir / "index_maps.npy"
        if not index_path.exists():
            raise FileNotFoundError(f"Missing processed index maps: {index_path}")
        self.index_maps = np.load(index_path)

        alpha_path = self.processed_dir / "alpha_masks.npy"
        if alpha_path.exists():
            self.alpha_masks = np.load(alpha_path).astype(bool)
        else:
            # Legacy fallback for old processed data. New preprocessing writes alpha_masks.npy.
            self.alpha_masks = self.index_maps != 0

        self.manifest = self._load_manifest()
        self.indices = self._select_indices(split)
        if max_samples is not None:
            self.indices = self.indices[:max_samples]

        self.preview = self._load_preview() if return_rgb else None

        self.palette_extractor = PaletteExtractor()
        self.palette_extractor.load(self.processed_dir / "palette.json")
        self.palette = self.palette_extractor.palette
        self.palette_size = len(self.palette)
        self.vocab_size = self.palette_size + 1
        self.mask_token = self.vocab_size

        self._validate_arrays()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        real_idx = self.indices[idx]
        index_map_np = self.index_maps[real_idx]
        alpha_mask_np = self.alpha_masks[real_idx]

        index_map = torch.from_numpy(index_map_np.astype(np.int64))
        alpha_mask = torch.from_numpy(alpha_mask_np.astype(bool))
        multi_scale_maps = self.tokenizer.encode(index_map)
        token_sequence = self.tokenizer.to_sequence(multi_scale_maps)

        result = {
            "index": torch.tensor(real_idx, dtype=torch.long),
            "index_map": index_map,
            "alpha_mask": alpha_mask,
            "multi_scale_maps": multi_scale_maps,
            "token_sequence": token_sequence,
        }

        if self.preview is not None:
            rgb = self._preview_to_tensor(self.preview[real_idx])
            result["rgb_preview"] = rgb
            result["rgb"] = rgb  # Backward-compatible alias for notebooks.

        return result

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        result = {
            "index": torch.stack([b["index"] for b in batch]),
            "index_map": torch.stack([b["index_map"] for b in batch]),
            "alpha_mask": torch.stack([b["alpha_mask"] for b in batch]),
            "token_sequence": torch.stack([b["token_sequence"] for b in batch]),
        }

        n_scales = len(batch[0]["multi_scale_maps"])
        result["multi_scale_maps"] = [
            torch.stack([b["multi_scale_maps"][scale_idx] for b in batch])
            for scale_idx in range(n_scales)
        ]

        if "rgb_preview" in batch[0]:
            rgb = torch.stack([b["rgb_preview"] for b in batch])
            result["rgb_preview"] = rgb
            result["rgb"] = rgb
        return result

    def _load_manifest(self) -> dict:
        manifest_path = self.processed_dir / "manifest.json"
        if manifest_path.exists():
            return json.loads(manifest_path.read_text())
        return {"samples": [{"index": i} for i in range(len(self.index_maps))]}

    def _select_indices(self, split: Optional[str]) -> list[int]:
        all_indices = list(range(len(self.index_maps)))
        if split is None or split == "all":
            return all_indices

        samples = self.manifest.get("samples", [])
        if not samples:
            raise ValueError(f"split={split!r} requested but manifest has no samples")

        selected = [
            int(sample.get("index", idx))
            for idx, sample in enumerate(samples)
            if sample.get("split") == split
        ]
        if not selected:
            raise ValueError(f"No samples found for split={split!r} in {self.processed_dir / 'manifest.json'}")
        return selected

    def _load_preview(self) -> Optional[np.ndarray]:
        for name in ("quantized_rgba.npy", "quantized_rgb.npy"):
            path = self.processed_dir / name
            if path.exists():
                return np.load(path)
        return None

    @staticmethod
    def _preview_to_tensor(preview: np.ndarray) -> torch.Tensor:
        if preview.shape[-1] == 4:
            alpha = preview[:, :, 3:4].astype(np.float32) / 255.0
            rgb = preview[:, :, :3].astype(np.float32) * alpha
        else:
            rgb = preview[:, :, :3].astype(np.float32)
        return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

    def _validate_arrays(self) -> None:
        if self.index_maps.shape != self.alpha_masks.shape:
            raise ValueError(
                f"index_maps shape {self.index_maps.shape} != alpha_masks shape {self.alpha_masks.shape}"
            )
        if self.index_maps.ndim != 3:
            raise ValueError(f"index_maps must have shape (N, H, W), got {self.index_maps.shape}")
        if self.index_maps.shape[1:] != (32, 32):
            raise ValueError(f"index_maps must be 32x32, got {self.index_maps.shape[1:]}")
        if self.index_maps.min() < 0 or self.index_maps.max() > self.vocab_size - 1:
            raise ValueError(
                f"token range [{self.index_maps.min()}, {self.index_maps.max()}] outside [0, {self.vocab_size - 1}]"
            )
        if len(self.indices) == 0:
            raise ValueError("Dataset split is empty")


def get_dataloader(
    processed_dir: str | Path,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    scale_resolutions: Optional[list[int]] = None,
    split: Optional[str] = None,
    max_samples: Optional[int] = None,
    return_rgb: bool = True,
) -> DataLoader:
    """Create a DataLoader for processed pixel art data."""
    dataset = PixelArtDataset(
        processed_dir=processed_dir,
        split=split,
        scale_resolutions=scale_resolutions,
        return_rgb=return_rgb,
        max_samples=max_samples,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=PixelArtDataset.collate_fn,
        pin_memory=False,
    )


def get_combined_dataloader(
    processed_dirs: list[str | Path],
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
        pin_memory=False,
    )
