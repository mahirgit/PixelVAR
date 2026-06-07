"""Image dataset for learned VQ-VAE tokenizer training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SpriteImageDataset(Dataset):
    """Load 32x32 RGBA sprite arrays from a processed PixelVAR dataset."""

    def __init__(
        self,
        processed_dir: str | Path,
        split: Optional[str] = None,
        image_array: str = "originals_rgba.npy",
        max_samples: Optional[int] = None,
    ):
        self.processed_dir = Path(processed_dir)
        self.split = split

        image_path = self._find_image_array(image_array)
        self.images = np.load(image_path, mmap_mode="r")
        if self.images.ndim != 4 or self.images.shape[1:3] != (32, 32) or self.images.shape[-1] not in (3, 4):
            raise ValueError(f"Expected image array shape (N, 32, 32, 3/4), got {self.images.shape}")

        self.manifest = self._load_manifest()
        self.indices = self._select_indices(split)
        if max_samples is not None:
            self.indices = self.indices[:max_samples]
        if not self.indices:
            raise ValueError(f"No samples selected from {self.processed_dir}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        real_idx = self.indices[idx]
        image = np.asarray(self.images[real_idx], dtype=np.uint8)
        if image.shape[-1] == 3:
            alpha = np.full((*image.shape[:2], 1), 255, dtype=np.uint8)
            image = np.concatenate([image, alpha], axis=-1)
        tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
        return {"index": torch.tensor(real_idx, dtype=torch.long), "image": tensor}

    def _find_image_array(self, preferred: str) -> Path:
        candidates = [preferred, "originals_rgba.npy", "quantized_rgba.npy"]
        for name in dict.fromkeys(candidates):
            path = self.processed_dir / name
            if path.exists():
                return path
        raise FileNotFoundError(
            f"No image array found in {self.processed_dir}; expected one of {', '.join(dict.fromkeys(candidates))}"
        )

    def _load_manifest(self) -> dict:
        manifest_path = self.processed_dir / "manifest.json"
        if manifest_path.exists():
            return json.loads(manifest_path.read_text())
        return {"samples": [{"index": i} for i in range(len(self.images))]}

    def _select_indices(self, split: Optional[str]) -> list[int]:
        if split is None or split == "all":
            return list(range(len(self.images)))
        selected = [
            int(sample.get("index", idx))
            for idx, sample in enumerate(self.manifest.get("samples", []))
            if sample.get("split") == split
        ]
        if not selected:
            raise ValueError(f"No samples found for split={split!r} in {self.processed_dir / 'manifest.json'}")
        return selected


def sprite_image_collate(batch: list[dict]) -> dict:
    return {
        "index": torch.stack([item["index"] for item in batch]),
        "image": torch.stack([item["image"] for item in batch]),
    }


def get_sprite_image_dataloader(
    processed_dir: str | Path,
    split: Optional[str],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    image_array: str = "originals_rgba.npy",
    max_samples: Optional[int] = None,
) -> DataLoader:
    dataset = SpriteImageDataset(
        processed_dir=processed_dir,
        split=split,
        image_array=image_array,
        max_samples=max_samples,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=sprite_image_collate,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )
