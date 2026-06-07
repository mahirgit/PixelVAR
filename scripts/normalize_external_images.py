#!/usr/bin/env python3
"""Normalize external baseline images into PixelVAR's 32x32 PNG protocol."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def image_paths(input_dir: Path, pattern: str | None) -> list[Path]:
    if pattern:
        paths = [path for path in input_dir.rglob(pattern) if path.is_file()]
    else:
        paths = [path for path in input_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS]
    paths = [path for path in paths if path.suffix.lower() in IMAGE_EXTENSIONS]
    if not paths:
        suffix = f" matching {pattern!r}" if pattern else ""
        raise ValueError(f"No image files{suffix} found in {input_dir}")
    return sorted(paths)


def choose_paths(paths: list[Path], max_images: int, seed: int) -> list[Path]:
    if max_images <= 0 or len(paths) <= max_images:
        return paths
    rng = random.Random(seed)
    selected = list(paths)
    rng.shuffle(selected)
    return sorted(selected[:max_images])


def load_palette(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    data = json.loads(path.read_text())
    colors = np.asarray(data["colors"], dtype=np.uint8)
    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError(f"Palette colors must have shape (K, 3), got {colors.shape}")
    return colors


def center_crop_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def transparent_corner_rgb(rgba: np.ndarray) -> np.ndarray:
    h, w = rgba.shape[:2]
    corner = max(1, min(h, w) // 16)
    patches = [
        rgba[:corner, :corner, :3],
        rgba[:corner, w - corner :, :3],
        rgba[h - corner :, :corner, :3],
        rgba[h - corner :, w - corner :, :3],
    ]
    return np.concatenate([patch.reshape(-1, 3) for patch in patches], axis=0).mean(axis=0)


def apply_corner_transparency(rgba: np.ndarray, tolerance: float) -> np.ndarray:
    output = rgba.copy()
    background = transparent_corner_rgb(output).astype(np.float32)
    rgb = output[:, :, :3].astype(np.float32)
    distance = np.sqrt(((rgb - background[None, None, :]) ** 2).sum(axis=2))
    output[distance <= tolerance, 3] = 0
    output[output[:, :, 3] == 0, :3] = 0
    return output


def quantize_to_palette(rgba: np.ndarray, palette: np.ndarray | None, alpha_threshold: int) -> np.ndarray:
    output = np.zeros_like(rgba, dtype=np.uint8)
    opaque = rgba[:, :, 3] >= alpha_threshold
    if not opaque.any():
        return output

    if palette is None:
        output[opaque, :3] = rgba[opaque, :3]
        output[opaque, 3] = 255
        return output

    pixels = rgba[opaque, :3].astype(np.float32)
    palette_f = palette.astype(np.float32)
    distances = ((pixels[:, None, :] - palette_f[None, :, :]) ** 2).sum(axis=2)
    nearest = distances.argmin(axis=1)
    output[opaque, :3] = palette[nearest]
    output[opaque, 3] = 255
    return output


def normalize_image(
    path: Path,
    image_size: int,
    palette: np.ndarray | None,
    alpha_threshold: int,
    crop_square: bool,
    transparent_from_corners: bool,
    transparent_tolerance: float,
) -> np.ndarray:
    image = Image.open(path).convert("RGBA")
    if crop_square:
        image = center_crop_square(image)
    image = image.resize((image_size, image_size), Image.Resampling.NEAREST)
    rgba = np.asarray(image, dtype=np.uint8)
    if transparent_from_corners:
        rgba = apply_corner_transparency(rgba, transparent_tolerance)
    return quantize_to_palette(rgba, palette, alpha_threshold)


def clear_existing_images(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize external images to 32x32 RGBA PNGs")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="", help="Optional recursive filename glob, e.g. final_argmax.png")
    parser.add_argument("--palette-json", type=Path)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", default="external")
    parser.add_argument("--alpha-threshold", type=int, default=128)
    parser.add_argument("--no-center-crop", action="store_true")
    parser.add_argument("--transparent-from-corners", action="store_true")
    parser.add_argument("--transparent-tolerance", type=float, default=12.0)
    args = parser.parse_args()

    palette = load_palette(args.palette_json)
    paths = choose_paths(image_paths(args.input_dir, args.pattern or None), args.max_images, args.seed)
    clear_existing_images(args.output_dir)

    rows = []
    for idx, path in enumerate(paths):
        image = normalize_image(
            path=path,
            image_size=args.image_size,
            palette=palette,
            alpha_threshold=args.alpha_threshold,
            crop_square=not args.no_center_crop,
            transparent_from_corners=args.transparent_from_corners,
            transparent_tolerance=args.transparent_tolerance,
        )
        out_path = args.output_dir / f"{args.prefix}_{idx:06d}.png"
        Image.fromarray(image, mode="RGBA").save(out_path)
        rows.append({"index": idx, "source": path.as_posix(), "path": out_path.as_posix()})

    manifest = {
        "input_dir": args.input_dir.as_posix(),
        "output_dir": args.output_dir.as_posix(),
        "pattern": args.pattern,
        "palette_json": args.palette_json.as_posix() if args.palette_json else None,
        "image_size": args.image_size,
        "num_images": len(rows),
        "images": rows,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(rows)} normalized images to {args.output_dir}")


if __name__ == "__main__":
    main()
