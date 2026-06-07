#!/usr/bin/env python3
"""Validate an exported VQ-token processed dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixelvar.data.splits import assert_no_group_split_leakage, assert_no_split_leakage


def main() -> None:
    parser = argparse.ArgumentParser(description="Check exported VQ token data")
    parser.add_argument("--dataset", type=str, default="sprites_vqvae16")
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    data_dir = args.processed_root / args.dataset
    required = ["index_maps.npy", "manifest.json", "codebook_usage.json"]
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        print(f"[error] Missing files in {data_dir}: {missing}")
        sys.exit(1)

    index_maps = np.load(data_dir / "index_maps.npy", mmap_mode="r")
    manifest = json.loads((data_dir / "manifest.json").read_text())
    usage = json.loads((data_dir / "codebook_usage.json").read_text())

    errors = []
    if index_maps.ndim != 3:
        errors.append(f"index_maps must have shape (N, H, W), got {index_maps.shape}")
    elif index_maps.shape[1] != index_maps.shape[2]:
        errors.append(f"index maps must be square, got {index_maps.shape[1:]}")
    vocab_size = int(manifest.get("vocab_size", usage.get("num_codes", int(index_maps.max()) + 1)))
    if int(index_maps.min()) < 0 or int(index_maps.max()) >= vocab_size:
        errors.append(f"token range [{int(index_maps.min())}, {int(index_maps.max())}] outside [0, {vocab_size - 1}]")
    if int(manifest.get("num_samples", -1)) != len(index_maps):
        errors.append(f"manifest num_samples {manifest.get('num_samples')} != array length {len(index_maps)}")
    try:
        assert_no_split_leakage(manifest.get("samples", []))
        assert_no_group_split_leakage(manifest.get("samples", []))
    except ValueError as exc:
        errors.append(str(exc))

    if errors:
        for error in errors:
            print(f"[error] {error}")
        sys.exit(1)

    split_counts = {}
    for sample in manifest.get("samples", []):
        split = sample.get("split", "unsplit")
        split_counts[split] = split_counts.get(split, 0) + 1

    print("VQ token data check passed")
    print(f"  samples: {len(index_maps)}")
    print(f"  shape: {index_maps.shape}")
    print(f"  token range: [{int(index_maps.min())}, {int(index_maps.max())}]")
    print(f"  vocab size: {vocab_size}")
    print(f"  used codes: {usage.get('used_codes')}/{usage.get('num_codes')}")
    print(f"  split counts: {split_counts}")


if __name__ == "__main__":
    main()
