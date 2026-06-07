#!/usr/bin/env python3
"""Import a filtered generated keep package as a processed PixelVAR dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixelvar.data.palette import PaletteExtractor
from pixelvar.data.splits import assert_no_group_split_leakage, make_group_splits
from pixelvar.utils import save_rgba_grid


def resolve_package_dir(keep_package: Path) -> tuple[tempfile.TemporaryDirectory | None, Path]:
    if keep_package.is_dir():
        return None, keep_package
    if not keep_package.exists():
        raise FileNotFoundError(keep_package)
    tempdir = tempfile.TemporaryDirectory()
    with zipfile.ZipFile(keep_package) as archive:
        archive.extractall(tempdir.name)
    return tempdir, Path(tempdir.name)


def build_manifest(dataset_name: str, split_map: dict[str, str], source_manifest: dict | None) -> dict:
    samples = []
    for idx in range(len(split_map)):
        group_id = f"generated_{idx:06d}"
        samples.append(
            {
                "index": idx,
                "path": f"generated_keep/{idx:06d}",
                "group_id": group_id,
                "variant": "sample",
                "source_kind": "generated_keep",
                "split": split_map[group_id],
            }
        )

    assert_no_group_split_leakage(samples)
    return {
        "dataset": dataset_name,
        "source_kind": "generated_keep",
        "source_manifest": source_manifest or {},
        "target_size": 32,
        "palette_size": 16,
        "alpha_threshold": 128,
        "transparent_token": 0,
        "palette_token_start": 1,
        "palette_token_end": 16,
        "vocab_size": 17,
        "scale_resolutions": [1, 2, 4, 8, 16, 32],
        "num_samples": len(samples),
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a filtered generated keep package")
    parser.add_argument("--keep-package", type=Path, required=True)
    parser.add_argument("--palette", type=Path, required=True)
    parser.add_argument("--dataset-name", type=str, default="sprites_generated_keep_170k")
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--grid-samples", type=int, default=64)
    args = parser.parse_args()

    tempdir, package_dir = resolve_package_dir(args.keep_package)
    try:
        index_maps_path = package_dir / "keep_index_maps.npy"
        if not index_maps_path.exists():
            raise FileNotFoundError(index_maps_path)

        index_maps = np.load(index_maps_path, mmap_mode="r")
        if index_maps.ndim != 3 or index_maps.shape[1:] != (32, 32):
            raise ValueError(f"Expected keep_index_maps.npy shape (N, 32, 32), got {index_maps.shape}")
        if index_maps.min() < 0 or index_maps.max() > 16:
            raise ValueError(f"Token range [{index_maps.min()}, {index_maps.max()}] outside [0, 16]")

        source_manifest_path = package_dir / "keep_manifest.json"
        source_manifest = json.loads(source_manifest_path.read_text()) if source_manifest_path.exists() else {}

        out_dir = args.processed_root / args.dataset_name
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

        print(f"Importing {len(index_maps):,} generated keep samples into {out_dir}")
        np.save(out_dir / "index_maps.npy", np.asarray(index_maps, dtype=np.uint8))
        np.save(out_dir / "alpha_masks.npy", np.asarray(index_maps != 0, dtype=bool))
        shutil.copy2(args.palette, out_dir / "palette.json")

        group_ids = [f"generated_{idx:06d}" for idx in range(len(index_maps))]
        split_map = make_group_splits(group_ids, seed=args.split_seed)
        (out_dir / "splits.json").write_text(json.dumps(split_map, indent=2, sort_keys=True))
        manifest = build_manifest(args.dataset_name, split_map, source_manifest)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        palette = PaletteExtractor()
        palette.load(out_dir / "palette.json")
        grid_count = min(args.grid_samples, len(index_maps))
        grid_images = np.stack([palette.render_index_map(index_maps[idx]) for idx in range(grid_count)], axis=0)
        save_rgba_grid(grid_images, out_dir / "sample_grid.png", columns=8)

        counts = {split: list(split_map.values()).count(split) for split in ("train", "val", "test")}
        print("Import complete")
        print(f"  samples: {len(index_maps):,}")
        print(f"  splits: {counts}")
        print(f"  token range: [{int(index_maps.min())}, {int(index_maps.max())}]")
        print(f"  opaque ratio: {float((index_maps != 0).mean()):.4f}")
    finally:
        if tempdir is not None:
            tempdir.cleanup()


if __name__ == "__main__":
    main()
