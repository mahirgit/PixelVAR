#!/usr/bin/env python3
"""Export a PixelVAR palette JSON file as a plain .hex palette."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_colors(path: Path) -> list[tuple[int, int, int]]:
    data = json.loads(path.read_text())
    colors = data.get("colors")
    if not isinstance(colors, list) or not colors:
        raise ValueError(f"{path} does not contain a non-empty 'colors' list")

    parsed = []
    for idx, color in enumerate(colors):
        if not isinstance(color, list | tuple) or len(color) != 3:
            raise ValueError(f"Palette color {idx} must be an RGB triplet, got {color!r}")
        rgb = tuple(int(channel) for channel in color)
        if any(channel < 0 or channel > 255 for channel in rgb):
            raise ValueError(f"Palette color {idx} has a channel outside [0, 255]: {rgb}")
        parsed.append(rgb)
    return parsed


def write_hex(colors: list[tuple[int, int, int]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    # SD-piXL's loader expects bare RRGGBB values, not CSS-style #RRGGBB.
    lines = [f"{r:02X}{g:02X}{b:02X}" for r, g, b in colors]
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PixelVAR palette.json to .hex")
    parser.add_argument("--palette-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    colors = load_colors(args.palette_json)
    write_hex(colors, args.output)
    print(f"Wrote {len(colors)} colors to {args.output}")


if __name__ == "__main__":
    main()
