#!/usr/bin/env python3
"""Curate the Hugging Face MSD Sprites dataset for PixelVAR.

The original Kaggle ``brentspell/sprites-dataset`` is no longer publicly
available. This script uses ``TalBarami/msd_sprites``, a modified variant of
the original YingzhenLi Sprites dataset, and writes it to the existing curated
Sprites layout:

    data/curated/sprites/
      manifest.json
      images/*.png

Frames are grouped by static character attributes so later train/val/test
splits do not leak the same character across splits.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from tqdm import tqdm


CURATED_DIR = Path("data/curated")
HF_DATASET = "TalBarami/msd_sprites"
STATIC_FIELDS = ("body", "bottom", "top", "hair")


def safe_name(value: str) -> str:
    keep = []
    for char in value:
        keep.append(char if char.isalnum() or char in "._-" else "_")
    cleaned = "".join(keep).strip("._")
    return cleaned or "sample"


def to_image(value: Any) -> Image.Image:
    """Convert a decoded Hugging Face Image cell to RGBA PIL."""
    if isinstance(value, Image.Image):
        return value.convert("RGBA")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(BytesIO(value["bytes"])).convert("RGBA")
        if value.get("path") is not None:
            return Image.open(value["path"]).convert("RGBA")
    array = np.asarray(value)
    if array.ndim in (2, 3):
        return Image.fromarray(array).convert("RGBA")
    raise TypeError(f"Unsupported frame type: {type(value)!r}")


def apply_corner_flood_transparency(image: Image.Image, tolerance: int) -> Image.Image:
    """Make the connected corner background transparent while preserving interior dark pixels."""
    rgba = np.array(image.convert("RGBA"))
    height, width = rgba.shape[:2]
    background = rgba[0, 0, :3].astype(np.int16)
    rgb = rgba[:, :, :3].astype(np.int16)
    close = np.abs(rgb - background).max(axis=-1) <= int(tolerance)

    visited = np.zeros((height, width), dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for y, x in ((0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)):
        if close[y, x]:
            visited[y, x] = True
            queue.append((y, x))

    while queue:
        y, x = queue.popleft()
        for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= next_y < height and 0 <= next_x < width and close[next_y, next_x] and not visited[next_y, next_x]:
                visited[next_y, next_x] = True
                queue.append((next_y, next_x))

    rgba[visited, 3] = 0
    return Image.fromarray(rgba, mode="RGBA")


def iter_rows(splits: Iterable[str]):
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency. Run: pip install datasets pyarrow") from exc

    for split in splits:
        dataset = load_dataset(HF_DATASET, split=split)
        for row_idx, row in enumerate(tqdm(dataset, desc=f"Reading {split}")):
            yield split, row_idx, row


def write_dataset(args: argparse.Namespace) -> None:
    out_dir = args.output_root / args.dataset_name
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if args.clean:
        for path in sorted(images_dir.glob("*.png")):
            path.unlink()

    samples = []
    total_rows = 0
    for split, row_idx, row in iter_rows(args.splits):
        if args.max_rows is not None and total_rows >= args.max_rows:
            break
        total_rows += 1

        static_values = {field: int(row[field]) for field in STATIC_FIELDS}
        group_id = "msd_" + "_".join(f"{field}{value}" for field, value in static_values.items())
        movement = int(row["movement"])
        frames = list(row["x"])

        for frame_idx, frame in enumerate(frames):
            image = to_image(frame)
            if args.transparent_mode == "corner-flood":
                image = apply_corner_flood_transparency(image, tolerance=args.transparent_tolerance)
            filename = f"{safe_name(group_id)}__m{movement:02d}__r{row_idx:05d}__f{frame_idx:02d}.png"
            image_path = images_dir / filename
            image.save(image_path)
            samples.append(
                {
                    "index": len(samples),
                    "image_path": str(image_path.relative_to(out_dir)),
                    "source_path": f"{HF_DATASET}:{split}:{row_idx}:{frame_idx}",
                    "group_id": group_id,
                    "frame_id": f"{split}_{row_idx:05d}_{frame_idx:02d}",
                    "variant": f"movement_{movement:02d}_frame_{frame_idx:02d}",
                    "source_kind": "msd_sprites",
                    "hf_split": split,
                    "movement": movement,
                    **static_values,
                }
            )

    manifest = {
        "dataset": args.dataset_name,
        "source_dataset": HF_DATASET,
        "source_note": "MSD Sprites replacement for unavailable brentspell/sprites-dataset.",
        "images_dir": "images",
        "num_samples": len(samples),
        "group_count": len({sample["group_id"] for sample in samples}),
        "grouping": "static character attributes: body,bottom,top,hair",
        "samples": samples,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(samples)} frames from {manifest['group_count']} groups to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate TalBarami/msd_sprites into PixelVAR frames")
    parser.add_argument("--dataset-name", default="sprites")
    parser.add_argument("--output-root", type=Path, default=CURATED_DIR)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--clean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--transparent-mode", choices=["corner-flood", "none"], default="corner-flood")
    parser.add_argument("--transparent-tolerance", type=int, default=8)
    args = parser.parse_args()
    write_dataset(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
