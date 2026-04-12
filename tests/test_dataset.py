"""
Unit tests for PixelArtDataset and get_dataloader().

Uses a temporary directory with fake processed data to test
the dataset class without requiring real preprocessed sprites.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from pixelvar.data.dataset import PixelArtDataset, get_dataloader, get_combined_dataloader
from pixelvar.tokenizers import DeterministicTokenizer


@pytest.fixture
def fake_processed_dir():
    """Create a temp dir with minimal fake processed data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        n_samples = 12
        palette_size = 16

        # Fake index maps: random palette indices 0-15
        index_maps = np.random.RandomState(42).randint(
            0, palette_size, size=(n_samples, 32, 32)
        ).astype(np.uint8)
        np.save(tmpdir / "index_maps.npy", index_maps)

        # Fake quantized RGB
        quantized_rgb = np.random.RandomState(42).randint(
            0, 256, size=(n_samples, 32, 32, 3)
        ).astype(np.uint8)
        np.save(tmpdir / "quantized_rgb.npy", quantized_rgb)

        # Fake palette
        palette = np.random.RandomState(42).randint(
            0, 256, size=(palette_size, 3)
        ).tolist()
        with open(tmpdir / "palette.json", "w") as f:
            json.dump({"palette_size": palette_size, "colors": palette}, f)

        yield tmpdir


class TestPixelArtDataset:
    def test_loads_correctly(self, fake_processed_dir):
        ds = PixelArtDataset(str(fake_processed_dir))
        assert len(ds) == 12

    def test_getitem_keys(self, fake_processed_dir):
        ds = PixelArtDataset(str(fake_processed_dir))
        item = ds[0]
        assert "index_map" in item
        assert "multi_scale_maps" in item
        assert "token_sequence" in item
        assert "rgb" in item

    def test_getitem_shapes(self, fake_processed_dir):
        ds = PixelArtDataset(str(fake_processed_dir))
        item = ds[0]
        assert item["index_map"].shape == (32, 32)
        assert item["token_sequence"].shape == (1365,)
        assert item["rgb"].shape == (3, 32, 32)
        assert len(item["multi_scale_maps"]) == 6

    def test_getitem_multiscale_shapes(self, fake_processed_dir):
        ds = PixelArtDataset(str(fake_processed_dir))
        item = ds[0]
        expected_resolutions = [1, 2, 4, 8, 16, 32]
        for m, res in zip(item["multi_scale_maps"], expected_resolutions):
            assert m.shape == (res, res)

    def test_no_rgb(self, fake_processed_dir):
        ds = PixelArtDataset(str(fake_processed_dir), return_rgb=False)
        item = ds[0]
        assert "rgb" not in item

    def test_custom_tokenizer(self, fake_processed_dir):
        tok = DeterministicTokenizer(palette_size=16)
        ds = PixelArtDataset(str(fake_processed_dir), tokenizer=tok)
        item = ds[0]
        assert item["token_sequence"].shape == (1365,)

    def test_collate_fn(self, fake_processed_dir):
        ds = PixelArtDataset(str(fake_processed_dir))
        batch = [ds[i] for i in range(4)]
        collated = PixelArtDataset.collate_fn(batch)
        assert collated["index_map"].shape == (4, 32, 32)
        assert collated["token_sequence"].shape == (4, 1365)
        assert collated["rgb"].shape == (4, 3, 32, 32)
        assert len(collated["multi_scale_maps"]) == 6
        assert collated["multi_scale_maps"][0].shape == (4, 1, 1)
        assert collated["multi_scale_maps"][-1].shape == (4, 32, 32)


class TestGetDataloader:
    def test_creates_dataloader(self, fake_processed_dir):
        dl = get_dataloader(str(fake_processed_dir), batch_size=4, num_workers=0)
        batch = next(iter(dl))
        assert batch["index_map"].shape[0] == 4
        assert batch["token_sequence"].shape == (4, 1365)

    def test_with_custom_tokenizer(self, fake_processed_dir):
        tok = DeterministicTokenizer(palette_size=16)
        dl = get_dataloader(str(fake_processed_dir), batch_size=4, num_workers=0, tokenizer=tok)
        batch = next(iter(dl))
        assert batch["token_sequence"].shape == (4, 1365)


class TestGetCombinedDataloader:
    def test_combines_datasets(self, fake_processed_dir):
        # Use the same dir twice to simulate combining
        dl = get_combined_dataloader(
            [str(fake_processed_dir), str(fake_processed_dir)],
            batch_size=4,
            num_workers=0,
        )
        # Combined should have 12 + 12 = 24 samples
        total = sum(len(batch["index_map"]) for batch in dl)
        assert total == 24
