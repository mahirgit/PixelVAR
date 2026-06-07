#!/usr/bin/env python3
"""Curate raw sprite assets into flat RGBA frame folders.

This script prepares the proposal datasets before palette preprocessing:

- Sprites: accepts Kaggle/manual archives, numpy arrays, or image folders.
- OpenGameArt: accepts manually downloaded images/sprite sheets.

Output layout:

    data/curated/{dataset}/
      manifest.json
      images/*.png

The preprocessing script can then consume these curated folders and assign
train/val/test splits by ``group_id`` rather than by individual frame.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image
from tqdm import tqdm


RAW_DIR = Path("data/raw")
CURATED_DIR = Path("data/curated")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
ARCHIVE_EXTENSIONS = {".zip", ".tar", ".tgz", ".gz"}


@dataclass(frozen=True)
class CuratedFrame:
    image: np.ndarray
    source_path: Path
    group_id: str
    frame_id: str
    variant: str
    source_kind: str


def safe_name(value: str) -> str:
    """Return a filesystem-safe stable name."""
    value = value.replace("\\", "/").strip("/")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("._") or "sample"


def parse_hex_color(value: str | None) -> tuple[int, int, int] | None:
    if value is None:
        return None
    text = value.strip().lstrip("#")
    if len(text) != 6:
        raise ValueError(f"transparent color must be RRGGBB, got {value!r}")
    return tuple(int(text[i : i + 2], 16) for i in (0, 2, 4))


def extract_archives(raw_dir: Path) -> None:
    """Extract archives under raw_dir into raw_dir/extracted/{archive_stem}."""
    for archive in sorted(raw_dir.rglob("*")):
        if not archive.is_file() or archive.suffix.lower() not in ARCHIVE_EXTENSIONS:
            continue
        if archive.suffix.lower() == ".gz" and not archive.name.endswith(".tar.gz"):
            continue

        extract_to = raw_dir / "extracted" / safe_name(archive.stem.replace(".tar", ""))
        marker = extract_to / ".extracted"
        if marker.exists():
            continue
        extract_to.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {archive} -> {extract_to}")
        try:
            if archive.suffix.lower() == ".zip":
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(extract_to)
            else:
                with tarfile.open(archive) as tf:
                    tf.extractall(extract_to)
        except Exception as exc:
            print(f"  [warn] Could not extract {archive}: {exc}")
            continue
        marker.write_text("ok\n")


def ensure_rgba(image: np.ndarray, transparent_color: tuple[int, int, int] | None = None) -> np.ndarray:
    """Convert a numpy image to uint8 RGBA without compositing."""
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.size and arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        rgb = np.stack([arr] * 3, axis=-1)
        alpha = np.full((*arr.shape, 1), 255, dtype=np.uint8)
        arr = np.concatenate([rgb, alpha], axis=-1)
    elif arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim != 3:
        raise ValueError(f"Unsupported image shape: {arr.shape}")
    if arr.shape[-1] == 1:
        rgb = np.repeat(arr, 3, axis=-1)
        alpha = np.full((*arr.shape[:2], 1), 255, dtype=np.uint8)
        arr = np.concatenate([rgb, alpha], axis=-1)
    elif arr.shape[-1] == 3:
        alpha = np.full((*arr.shape[:2], 1), 255, dtype=np.uint8)
        arr = np.concatenate([arr[:, :, :3], alpha], axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[:, :, :4]
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")

    if transparent_color is not None:
        rgb = np.array(transparent_color, dtype=np.uint8)
        mask = np.all(arr[:, :, :3] == rgb, axis=-1)
        arr = arr.copy()
        arr[mask, 3] = 0
    return arr.astype(np.uint8)


def reshape_flat_image(flat: np.ndarray) -> np.ndarray:
    """Infer H/W/C for a flattened image vector."""
    length = int(flat.shape[0])
    for channels in (4, 3, 1):
        if length % channels != 0:
            continue
        side = int(math.sqrt(length // channels))
        if side * side * channels == length:
            if channels == 1:
                return flat.reshape(side, side)
            return flat.reshape(side, side, channels)
    raise ValueError(f"Cannot infer image shape from flattened length {length}")


def iter_npy_frames(
    npy_path: Path,
    frames_per_group: int | None,
    transparent_color: tuple[int, int, int] | None,
) -> Iterator[CuratedFrame]:
    """Yield frames from common sprite numpy array layouts."""
    data = np.load(npy_path, allow_pickle=False)
    stem = safe_name(npy_path.stem)

    def make_frame(item: np.ndarray, group_idx: int, frame_idx: int) -> CuratedFrame:
        if item.ndim == 1:
            item = reshape_flat_image(item)
        return CuratedFrame(
            image=ensure_rgba(item, transparent_color=transparent_color),
            source_path=npy_path,
            group_id=f"{stem}_{group_idx:06d}",
            frame_id=f"{frame_idx:04d}",
            variant="frame",
            source_kind="npy",
        )

    if data.ndim >= 5:
        # Typical sequence layouts: (N, T, H, W, C) or (N, T, C, H, W).
        for group_idx in range(data.shape[0]):
            for frame_idx in range(data.shape[1]):
                yield make_frame(data[group_idx, frame_idx], group_idx, frame_idx)
        return

    if data.ndim in (3, 4):
        # Flat frame layouts: (N, H, W), (N, H, W, C), or (N, C, H, W).
        for idx in range(data.shape[0]):
            group_idx = idx if frames_per_group is None else idx // frames_per_group
            frame_idx = 0 if frames_per_group is None else idx % frames_per_group
            yield make_frame(data[idx], group_idx, frame_idx)
        return

    if data.ndim == 2:
        # Either one grayscale image or rows of flattened images. Prefer rows if
        # every row can be interpreted as a square image.
        row_as_images = data.shape[0] > 1
        if row_as_images:
            try:
                reshape_flat_image(data[0])
            except ValueError:
                row_as_images = False
        if row_as_images:
            for idx in range(data.shape[0]):
                group_idx = idx if frames_per_group is None else idx // frames_per_group
                frame_idx = 0 if frames_per_group is None else idx % frames_per_group
                yield make_frame(data[idx], group_idx, frame_idx)
        else:
            yield make_frame(data, 0, 0)
        return

    raise ValueError(f"Unsupported npy shape for {npy_path}: {data.shape}")


def iter_sheet_tiles(
    image: np.ndarray,
    source_path: Path,
    group_id: str,
    tile_size: tuple[int, int],
    min_opaque_pixels: int,
) -> Iterator[CuratedFrame]:
    tile_w, tile_h = tile_size
    height, width = image.shape[:2]
    rows = height // tile_h
    cols = width // tile_w
    for row in range(rows):
        for col in range(cols):
            tile = image[row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w]
            opaque = int((tile[:, :, 3] > 0).sum())
            if opaque < min_opaque_pixels:
                continue
            yield CuratedFrame(
                image=tile,
                source_path=source_path,
                group_id=group_id,
                frame_id=f"r{row:03d}_c{col:03d}",
                variant="tile",
                source_kind="sheet",
            )


def iter_image_frames(
    image_path: Path,
    root: Path,
    sheet_tile_size: tuple[int, int] | None,
    min_opaque_pixels: int,
    transparent_color: tuple[int, int, int] | None,
) -> Iterator[CuratedFrame]:
    image = ensure_rgba(np.array(Image.open(image_path).convert("RGBA")), transparent_color=transparent_color)
    rel = image_path.relative_to(root)
    group_id = safe_name(str(rel.with_suffix("")))

    if sheet_tile_size is not None:
        tile_w, tile_h = sheet_tile_size
        height, width = image.shape[:2]
        if width >= tile_w and height >= tile_h and width % tile_w == 0 and height % tile_h == 0:
            yield from iter_sheet_tiles(image, image_path, group_id, sheet_tile_size, min_opaque_pixels)
            return

    opaque = int((image[:, :, 3] > 0).sum())
    if opaque >= min_opaque_pixels:
        yield CuratedFrame(
            image=image,
            source_path=image_path,
            group_id=group_id,
            frame_id="0000",
            variant="image",
            source_kind="image",
        )


def discover_frames(args: argparse.Namespace, dataset: str) -> Iterator[CuratedFrame]:
    raw_dir = args.raw_root / dataset
    if not raw_dir.exists():
        print(f"[warn] Missing raw dataset directory: {raw_dir}")
        return

    if args.extract_archives:
        extract_archives(raw_dir)

    transparent_color = parse_hex_color(args.transparent_color)
    tile_size = None
    if args.sheet_tile_size is not None:
        tile_size = (args.sheet_tile_size, args.sheet_tile_size)

    npy_files = sorted(p for p in raw_dir.rglob("*.npy") if "label" not in p.name.lower())
    for npy_path in npy_files:
        try:
            yield from iter_npy_frames(
                npy_path,
                frames_per_group=args.sprites_frames_per_group if dataset == "sprites" else None,
                transparent_color=transparent_color,
            )
        except Exception as exc:
            print(f"  [warn] Skipping {npy_path}: {exc}")

    image_files = sorted(
        p
        for p in raw_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and ".extracted" not in p.parts
    )
    for image_path in image_files:
        try:
            yield from iter_image_frames(
                image_path,
                root=raw_dir,
                sheet_tile_size=tile_size,
                min_opaque_pixels=args.min_opaque_pixels,
                transparent_color=transparent_color,
            )
        except Exception as exc:
            print(f"  [warn] Skipping {image_path}: {exc}")


def write_curated_dataset(args: argparse.Namespace, dataset: str) -> None:
    out_dir = args.output_root / dataset
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    frames = discover_frames(args, dataset)
    if frames is None:
        return

    for idx, frame in enumerate(tqdm(frames, desc=f"Curating {dataset}")):
        if args.max_images is not None and idx >= args.max_images:
            break
        filename = f"{safe_name(frame.group_id)}__{safe_name(frame.frame_id)}.png"
        image_path = images_dir / filename
        Image.fromarray(frame.image, mode="RGBA").save(image_path)
        samples.append(
            {
                "index": idx,
                "image_path": str(image_path.relative_to(out_dir)),
                "source_path": str(frame.source_path),
                "group_id": frame.group_id,
                "frame_id": frame.frame_id,
                "variant": frame.variant,
                "source_kind": frame.source_kind,
            }
        )

    manifest = {
        "dataset": dataset,
        "source_root": str((args.raw_root / dataset).resolve()),
        "images_dir": "images",
        "num_samples": len(samples),
        "group_count": len({sample["group_id"] for sample in samples}),
        "samples": samples,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(samples)} frames from {manifest['group_count']} groups to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate raw PixelVAR sprite datasets")
    parser.add_argument("--dataset", choices=["sprites", "opengameart", "all"], default="all")
    parser.add_argument("--raw-root", type=Path, default=RAW_DIR)
    parser.add_argument("--output-root", type=Path, default=CURATED_DIR)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--extract-archives", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--sprites-frames-per-group",
        type=int,
        default=178,
        help="For flat Sprites arrays, keep consecutive frames from the same character in one split group.",
    )
    parser.add_argument(
        "--sheet-tile-size",
        type=int,
        default=None,
        help="Split square sprite sheets into this tile size, e.g. 32 or 64.",
    )
    parser.add_argument("--min-opaque-pixels", type=int, default=16)
    parser.add_argument(
        "--transparent-color",
        type=str,
        default=None,
        help="Optional RGB key to make transparent, e.g. '#000000' or '#ff00ff'.",
    )
    args = parser.parse_args()

    datasets = ["sprites", "opengameart"] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        write_curated_dataset(args, dataset)

    next_dataset = "all" if len(datasets) > 1 else datasets[0]
    print(f"Curation complete. Next: python scripts/preprocess_data.py --dataset {next_dataset}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
