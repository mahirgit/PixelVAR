#!/usr/bin/env python3
"""Mix multiple processed PixelVAR datasets into one processed dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixelvar.data.palette import PaletteExtractor
from pixelvar.data.splits import assert_no_group_split_leakage
from pixelvar.utils import save_rgba_grid


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_palette(path: Path) -> dict:
    data = load_json(path)
    return {
        "palette_size": data["palette_size"],
        "colors": data["colors"],
        "alpha_threshold": data.get("alpha_threshold", 128),
    }


def validate_source(source_dir: Path, reference_palette: dict | None) -> dict:
    required = ["index_maps.npy", "alpha_masks.npy", "palette.json", "manifest.json"]
    missing = [name for name in required if not (source_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {missing} in {source_dir}")

    index_maps = np.load(source_dir / "index_maps.npy", mmap_mode="r")
    alpha_masks = np.load(source_dir / "alpha_masks.npy", mmap_mode="r")
    if index_maps.shape != alpha_masks.shape:
        raise ValueError(f"{source_dir}: index_maps shape {index_maps.shape} != alpha_masks shape {alpha_masks.shape}")
    if index_maps.ndim != 3 or index_maps.shape[1:] != (32, 32):
        raise ValueError(f"{source_dir}: expected index_maps shape (N, 32, 32), got {index_maps.shape}")
    if int(index_maps.min()) < 0 or int(index_maps.max()) > 16:
        raise ValueError(f"{source_dir}: token range [{index_maps.min()}, {index_maps.max()}] outside [0, 16]")

    palette = load_palette(source_dir / "palette.json")
    if reference_palette is not None and palette != reference_palette:
        raise ValueError(f"{source_dir}: palette does not match first source palette")

    manifest = load_json(source_dir / "manifest.json")
    if int(manifest.get("num_samples", -1)) != len(index_maps):
        raise ValueError(f"{source_dir}: manifest num_samples does not match index_maps length")

    return {
        "index_maps": index_maps,
        "alpha_masks": alpha_masks.astype(bool),
        "manifest": manifest,
        "palette": palette,
    }


def prefixed_sample(source_name: str, source_idx: int, new_idx: int, sample: dict) -> dict:
    group_id = sample.get("group_id", sample.get("pokemon_id", source_idx))
    mixed = {
        "index": new_idx,
        "source_dataset": source_name,
        "source_index": int(sample.get("index", source_idx)),
        "path": f"{source_name}:{sample.get('path', source_idx)}",
        "group_id": f"{source_name}:{group_id}",
        "source_kind": sample.get("source_kind", source_name),
        "split": sample.get("split", "train"),
    }
    if sample.get("variant") is not None:
        mixed["variant"] = sample["variant"]
    return mixed


def write_manifest(dataset_name: str, source_manifests: dict[str, dict], samples: list[dict], palette_size: int) -> dict:
    assert_no_group_split_leakage(samples)
    return {
        "dataset": dataset_name,
        "source_kind": "mixed_processed",
        "source_datasets": {
            name: {
                "dataset": manifest.get("dataset", name),
                "num_samples": manifest.get("num_samples"),
            }
            for name, manifest in source_manifests.items()
        },
        "target_size": 32,
        "palette_size": palette_size,
        "alpha_threshold": 128,
        "transparent_token": 0,
        "palette_token_start": 1,
        "palette_token_end": palette_size,
        "vocab_size": palette_size + 1,
        "scale_resolutions": [1, 2, 4, 8, 16, 32],
        "num_samples": len(samples),
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Mix processed PixelVAR datasets")
    parser.add_argument("--dataset-name", type=str, default="sprites_mixed_real_generated")
    parser.add_argument("--sources", nargs="+", default=["sprites", "sprites_generated_keep_170k"])
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--grid-samples-per-source", type=int, default=32)
    args = parser.parse_args()

    out_dir = args.processed_root / args.dataset_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    source_payloads = {}
    reference_palette = None
    total_samples = 0
    for source in args.sources:
        source_dir = args.processed_root / source
        payload = validate_source(source_dir, reference_palette)
        reference_palette = reference_palette or payload["palette"]
        source_payloads[source] = payload
        total_samples += len(payload["index_maps"])

    index_out = np.lib.format.open_memmap(
        out_dir / "index_maps.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(total_samples, 32, 32),
    )
    alpha_out = np.lib.format.open_memmap(
        out_dir / "alpha_masks.npy",
        mode="w+",
        dtype=bool,
        shape=(total_samples, 32, 32),
    )

    samples = []
    source_manifests = {}
    cursor = 0
    split_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for source, payload in source_payloads.items():
        index_maps = payload["index_maps"]
        alpha_masks = payload["alpha_masks"]
        count = len(index_maps)
        index_out[cursor : cursor + count] = index_maps
        alpha_out[cursor : cursor + count] = alpha_masks

        manifest = payload["manifest"]
        source_manifests[source] = manifest
        manifest_samples = manifest.get("samples", [])
        if len(manifest_samples) != count:
            raise ValueError(f"{source}: manifest sample count {len(manifest_samples)} != array count {count}")

        for source_idx, sample in enumerate(manifest_samples):
            mixed = prefixed_sample(source, source_idx, cursor + source_idx, sample)
            samples.append(mixed)
            split_counts[mixed["split"]] = split_counts.get(mixed["split"], 0) + 1
        source_counts[source] = count
        cursor += count

    index_out.flush()
    alpha_out.flush()
    shutil.copy2(args.processed_root / args.sources[0] / "palette.json", out_dir / "palette.json")
    first_palette = PaletteExtractor()
    first_palette.load(out_dir / "palette.json")
    shutil.copy2(args.processed_root / args.sources[0] / "palette_swatch.png", out_dir / "palette_swatch.png") if (args.processed_root / args.sources[0] / "palette_swatch.png").exists() else None

    manifest = write_manifest(
        args.dataset_name,
        source_manifests=source_manifests,
        samples=samples,
        palette_size=reference_palette["palette_size"],
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    grid_images = []
    for source, payload in source_payloads.items():
        n = min(args.grid_samples_per_source, len(payload["index_maps"]))
        grid_images.extend(first_palette.render_index_map(payload["index_maps"][idx]) for idx in range(n))
    if grid_images:
        save_rgba_grid(np.stack(grid_images, axis=0), out_dir / "sample_grid.png", columns=8)

    summary = {
        "dataset": args.dataset_name,
        "sources": source_counts,
        "splits": split_counts,
        "num_samples": total_samples,
        "opaque_ratio": float(alpha_out.mean()),
    }
    (out_dir / "mix_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("Mixed dataset written")
    print(f"  output: {out_dir}")
    print(f"  sources: {source_counts}")
    print(f"  splits: {split_counts}")
    print(f"  samples: {total_samples}")
    print(f"  opaque ratio: {summary['opaque_ratio']:.4f}")


if __name__ == "__main__":
    main()
