import json

import numpy as np

from pixelvar.data.dataset import PixelArtDataset
from pixelvar.data.palette import PaletteExtractor


def test_dataset_filters_split_and_returns_token_sequence(tmp_path):
    processed = tmp_path / "pokemon"
    processed.mkdir()

    index_maps = np.zeros((4, 32, 32), dtype=np.uint8)
    index_maps[:, 8:24, 8:24] = np.arange(1, 5, dtype=np.uint8)[:, None, None]
    alpha_masks = index_maps != 0
    np.save(processed / "index_maps.npy", index_maps)
    np.save(processed / "alpha_masks.npy", alpha_masks)

    palette = PaletteExtractor(palette_size=16)
    palette.palette = np.tile(np.array([[64, 128, 192]], dtype=np.uint8), (16, 1))
    palette.save(processed / "palette.json")
    np.save(processed / "quantized_rgba.npy", np.stack([palette.render_index_map(m) for m in index_maps]))

    manifest = {
        "num_samples": 4,
        "samples": [
            {"index": 0, "pokemon_id": "1", "split": "train"},
            {"index": 1, "pokemon_id": "2", "split": "train"},
            {"index": 2, "pokemon_id": "3", "split": "val"},
            {"index": 3, "pokemon_id": "4", "split": "test"},
        ],
    }
    (processed / "manifest.json").write_text(json.dumps(manifest))

    dataset = PixelArtDataset(processed, split="train")
    sample = dataset[0]

    assert len(dataset) == 2
    assert sample["index_map"].shape == (32, 32)
    assert sample["alpha_mask"].shape == (32, 32)
    assert sample["token_sequence"].shape == (1365,)
    assert sample["rgb_preview"].shape == (3, 32, 32)
