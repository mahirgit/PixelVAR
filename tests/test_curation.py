from pathlib import Path

import numpy as np

from scripts.curate_data import iter_npy_frames, iter_sheet_tiles
from scripts.preprocess_data import ImageRecord, build_manifest


def test_flat_sprite_array_groups_consecutive_frames(tmp_path):
    npy_path = tmp_path / "sprites_full.npy"
    data = np.zeros((6, 4, 4, 4), dtype=np.uint8)
    data[:, :, :, 3] = 255
    np.save(npy_path, data)

    frames = list(iter_npy_frames(npy_path, frames_per_group=3, transparent_color=None))

    assert len(frames) == 6
    assert [frame.group_id for frame in frames] == [
        "sprites_full_000000",
        "sprites_full_000000",
        "sprites_full_000000",
        "sprites_full_000001",
        "sprites_full_000001",
        "sprites_full_000001",
    ]
    assert [frame.frame_id for frame in frames] == ["0000", "0001", "0002", "0000", "0001", "0002"]


def test_sheet_split_skips_empty_tiles():
    sheet = np.zeros((4, 8, 4), dtype=np.uint8)
    sheet[:, :4, :3] = [120, 40, 80]
    sheet[:, :4, 3] = 255

    frames = list(
        iter_sheet_tiles(
            sheet,
            source_path=Path("sheet.png"),
            group_id="sheet",
            tile_size=(4, 4),
            min_opaque_pixels=1,
        )
    )

    assert len(frames) == 1
    assert frames[0].frame_id == "r000_c000"
    assert frames[0].image.shape == (4, 4, 4)


def test_manifest_assigns_generic_group_splits():
    records = [
        ImageRecord(path=Path("a.png"), image=np.zeros((32, 32, 4), dtype=np.uint8), group_id="asset_a"),
        ImageRecord(path=Path("b.png"), image=np.zeros((32, 32, 4), dtype=np.uint8), group_id="asset_b"),
    ]
    manifest = build_manifest(
        dataset_name="sprites",
        records=records,
        target_size=32,
        palette_size=16,
        alpha_threshold=128,
        split_map={"asset_a": "train", "asset_b": "val"},
    )

    assert manifest["samples"][0]["group_id"] == "asset_a"
    assert manifest["samples"][0]["split"] == "train"
    assert manifest["samples"][1]["group_id"] == "asset_b"
    assert manifest["samples"][1]["split"] == "val"
