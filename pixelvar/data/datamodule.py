"""Lightning DataModule for PixelVAR."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    import lightning as L
except ModuleNotFoundError:  # pragma: no cover - exercised only before deps are installed.
    L = None

from torch.utils.data import DataLoader

from pixelvar.data.dataset import PixelArtDataset


if L is None:
    _LightningDataModule = object
else:
    _LightningDataModule = L.LightningDataModule


class PixelArtDataModule(_LightningDataModule):
    """Build train/val/test loaders from preprocessed PixelVAR arrays."""

    def __init__(
        self,
        processed_dir: str | Path,
        batch_size: int = 64,
        num_workers: int = 4,
        scale_resolutions: Optional[list[int]] = None,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
        max_test_samples: Optional[int] = None,
        return_rgb: bool = False,
    ):
        if L is None:
            raise ModuleNotFoundError("Install Lightning with `pip install lightning` to use PixelArtDataModule")
        super().__init__()
        self.processed_dir = Path(processed_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.scale_resolutions = scale_resolutions
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.max_test_samples = max_test_samples
        self.return_rgb = return_rgb

        self.train_dataset: PixelArtDataset | None = None
        self.val_dataset: PixelArtDataset | None = None
        self.test_dataset: PixelArtDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = PixelArtDataset(
                self.processed_dir,
                split="train",
                scale_resolutions=self.scale_resolutions,
                return_rgb=self.return_rgb,
                max_samples=self.max_train_samples,
            )
            self.val_dataset = PixelArtDataset(
                self.processed_dir,
                split="val",
                scale_resolutions=self.scale_resolutions,
                return_rgb=self.return_rgb,
                max_samples=self.max_val_samples,
            )
        if stage in (None, "test", "validate", "predict"):
            self.test_dataset = PixelArtDataset(
                self.processed_dir,
                split="test",
                scale_resolutions=self.scale_resolutions,
                return_rgb=self.return_rgb,
                max_samples=self.max_test_samples,
            )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, shuffle=False)

    def _loader(self, dataset: PixelArtDataset | None, shuffle: bool) -> DataLoader:
        if dataset is None:
            raise RuntimeError("DataModule setup has not been called")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=PixelArtDataset.collate_fn,
            pin_memory=False,
            persistent_workers=self.num_workers > 0,
        )
