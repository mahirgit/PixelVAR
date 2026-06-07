#!/usr/bin/env python3
"""Build labeled sample sheets from generated image folders."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" in value:
        name, path = value.split("=", 1)
        return name.strip(), Path(path)
    path = Path(value)
    return path.name, path


def image_paths(directory: Path) -> list[Path]:
    paths = [path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS]
    if not paths:
        raise ValueError(f"No images found in {directory}")
    return sorted(paths)


def select_paths(paths: list[Path], count: int, seed: int) -> list[Path]:
    rng = random.Random(seed)
    shuffled = list(paths)
    rng.shuffle(shuffled)
    return shuffled[: min(count, len(shuffled))]


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def checkerboard(size: int, block: int = 8) -> Image.Image:
    image = Image.new("RGBA", (size, size), (238, 238, 238, 255))
    draw = ImageDraw.Draw(image)
    for y in range(0, size, block):
        for x in range(0, size, block):
            if (x // block + y // block) % 2:
                draw.rectangle((x, y, x + block - 1, y + block - 1), fill=(216, 216, 216, 255))
    return image


def fit_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def build_sheet(
    folders: list[tuple[str, Path]],
    output: Path,
    samples_per_method: int,
    columns: int,
    seed: int,
    scale: int,
    title: str,
    note: str,
) -> None:
    font = load_font(18)
    small_font = load_font(14)
    title_font = load_font(24)
    tile = 32 * scale
    gap = 10
    label_width = 260
    top = 92
    note_height = 62 if note else 20
    row_height = tile + gap
    rows = len(folders)
    width = label_width + columns * tile + max(0, columns - 1) * gap + 24
    height = top + rows * row_height + note_height
    sheet = Image.new("RGBA", (width, height), (248, 248, 248, 255))
    draw = ImageDraw.Draw(sheet)

    draw.text((16, 14), title, fill=(24, 24, 24, 255), font=title_font)
    if note:
        y = 48
        for line in fit_text(draw, note, small_font, width - 32):
            draw.text((16, y), line, fill=(70, 70, 70, 255), font=small_font)
            y += 17

    for row_idx, (label, folder) in enumerate(folders):
        paths = select_paths(image_paths(folder), samples_per_method, seed + row_idx)
        y0 = top + row_idx * row_height
        label_lines = fit_text(draw, label, font, label_width - 26)
        for line_idx, line in enumerate(label_lines[:3]):
            draw.text((16, y0 + 8 + line_idx * 22), line, fill=(20, 20, 20, 255), font=font)
        for col_idx, path in enumerate(paths[:columns]):
            x0 = label_width + col_idx * (tile + gap)
            cell = checkerboard(tile)
            image = Image.open(path).convert("RGBA")
            if image.size != (32, 32):
                image = image.resize((32, 32), Image.Resampling.NEAREST)
            image = image.resize((tile, tile), Image.Resampling.NEAREST)
            cell.alpha_composite(image)
            sheet.alpha_composite(cell, (x0, y0))
            draw.rectangle((x0, y0, x0 + tile - 1, y0 + tile - 1), outline=(190, 190, 190, 255), width=1)

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    print(f"Wrote sample sheet to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a labeled sample sheet from image folders")
    parser.add_argument("--folder", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-method", type=int, default=16)
    parser.add_argument("--columns", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--title", default="Four-way generated sample comparison")
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    folders = [parse_named_path(value) for value in args.folder]
    build_sheet(
        folders=folders,
        output=args.output,
        samples_per_method=args.samples_per_method,
        columns=args.columns,
        seed=args.seed,
        scale=args.scale,
        title=args.title,
        note=args.note,
    )


if __name__ == "__main__":
    main()
