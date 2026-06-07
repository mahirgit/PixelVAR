#!/usr/bin/env python3
"""Download and curate a small public OpenGameArt sprite set.

The output is a curated frame dataset, not loose raw sheets:

    data/curated/opengameart/
      manifest.json
      source_manifest.json
      images/*.png

The source list intentionally favors CC0 character/sprite packs with either
32x32 frames or simple grid layouts.  This keeps the stretch dataset
reproducible and avoids training on resized contact sheets.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm


CURATED_DIR = Path("data/curated/opengameart")
DOWNLOAD_DIR = Path("data/raw/opengameart_public")


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    title: str
    author: str
    license: str
    page_url: str
    file_url: str
    file_name: str
    mode: str
    tile_width: int = 32
    tile_height: int = 32
    background_color: str | None = None
    include_pattern: str | None = None
    exclude_pattern: str | None = None
    group_mode: str = "file"
    max_component_width: int = 48
    max_component_height: int = 64


SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        source_id="eldiran_rpg_characters",
        title="32x32 RPG Character Sprites",
        author="Eldiran",
        license="CC0",
        page_url="https://opengameart.org/content/32x32-rpg-character-sprites",
        file_url="https://opengameart.org/sites/default/files/RPGCharacterSprites32x32.png",
        file_name="RPGCharacterSprites32x32.png",
        mode="grid",
        tile_width=32,
        tile_height=32,
        background_color="#ff00ff",
        group_mode="row",
    ),
    SourceSpec(
        source_id="eldiran_rpg_soldier",
        title="32x32 RPG Character Sprites - soldier sheet",
        author="Eldiran",
        license="CC0",
        page_url="https://opengameart.org/content/32x32-rpg-character-sprites",
        file_url="https://opengameart.org/sites/default/files/RPGSoldier32x32.png",
        file_name="RPGSoldier32x32.png",
        mode="grid",
        tile_width=32,
        tile_height=35,
        background_color="#ff00ff",
        group_mode="row",
    ),
    SourceSpec(
        source_id="cyanowl_human_sprites",
        title="32-pixel Human Sprites",
        author="cyanowl",
        license="CC0",
        page_url="https://opengameart.org/content/32-pixel-human-sprites",
        file_url="https://opengameart.org/sites/default/files/human_sprites_0.png",
        file_name="human_sprites_0.png",
        mode="components",
        group_mode="component_row",
        max_component_width=32,
        max_component_height=40,
    ),
    SourceSpec(
        source_id="arikel_rpg_walk",
        title="2D RPG character walk spritesheet",
        author="arikel",
        license="CC0 / CC-BY 4.0",
        page_url="https://opengameart.org/content/2d-rpg-character-walk-spritesheet",
        file_url="https://opengameart.org/sites/default/files/rpg_sprite_walk.png",
        file_name="rpg_sprite_walk.png",
        mode="grid",
        tile_width=32,
        tile_height=32,
        group_mode="file",
    ),
    SourceSpec(
        source_id="shade_puny_characters",
        title="Puny Characters",
        author="Shade",
        license="CC0",
        page_url="https://opengameart.org/content/puny-characters",
        file_url="https://opengameart.org/sites/default/files/puny-charactersorcs_included.zip",
        file_name="puny-charactersorcs_included.zip",
        mode="zip_grid",
        tile_width=32,
        tile_height=32,
        include_pattern=r"^Puny-Characters/.+\.png$",
        exclude_pattern=r"^Puny-Characters/Environment/",
        group_mode="archive_file",
    ),
    SourceSpec(
        source_id="doomsphere_charset",
        title="1-Bit Doomsphere Charset",
        author="B77345-100",
        license="CC0",
        page_url="https://opengameart.org/content/1-bit-doomsphere-charset",
        file_url="https://opengameart.org/sites/default/files/doomspherecharset.zip",
        file_name="doomspherecharset.zip",
        mode="zip_single",
        include_pattern=r"^Single PNGs/No-Outline/.+\.png$",
        group_mode="strip_trailing_digits",
    ),
    SourceSpec(
        source_id="kenney_roguelike_characters",
        title="Roguelike Character pack",
        author="Kenney",
        license="CC0",
        page_url="https://opengameart.org/content/roguelike-character-pack",
        file_url="https://opengameart.org/sites/default/files/Roguelike%20Characters%20pack.zip",
        file_name="Roguelike Characters pack.zip",
        mode="zip_grid",
        tile_width=17,
        tile_height=17,
        include_pattern=r"^Spritesheet/roguelikeChar_transparent\.png$",
        group_mode="row",
    ),
)


def safe_name(value: str) -> str:
    value = value.replace("\\", "/").strip("/")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("._") or "sample"


def parse_hex_color(value: str | None) -> tuple[int, int, int] | None:
    if value is None:
        return None
    text = value.strip().lstrip("#")
    if len(text) != 6:
        raise ValueError(f"Expected RRGGBB color, got {value!r}")
    return tuple(int(text[i : i + 2], 16) for i in (0, 2, 4))


def load_rgba(data: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(data)).convert("RGBA")
    return np.array(image, dtype=np.uint8)


def apply_background_key(image: np.ndarray, color: str | None) -> np.ndarray:
    rgb = parse_hex_color(color)
    if rgb is None:
        return image
    arr = image.copy()
    mask = np.all(arr[:, :, :3] == np.array(rgb, dtype=np.uint8), axis=-1)
    arr[mask, 3] = 0
    return arr


def opaque_bbox(image: np.ndarray) -> tuple[int, int, int, int] | None:
    mask = image[:, :, 3] > 0
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def should_keep(image: np.ndarray, min_opaque_pixels: int) -> bool:
    opaque = int((image[:, :, 3] > 0).sum())
    return opaque >= min_opaque_pixels


def download_file(spec: SourceSpec, download_dir: Path, timeout: int) -> tuple[Path, str, int]:
    download_dir.mkdir(parents=True, exist_ok=True)
    out_path = download_dir / safe_name(spec.file_name)
    if not out_path.exists():
        response = requests.get(spec.file_url, timeout=timeout)
        response.raise_for_status()
        out_path.write_bytes(response.content)
    data = out_path.read_bytes()
    return out_path, hashlib.sha256(data).hexdigest(), len(data)


def iter_grid_frames(
    image: np.ndarray,
    spec: SourceSpec,
    source_path: str,
    min_opaque_pixels: int,
) -> Iterable[dict]:
    keyed = apply_background_key(image, spec.background_color)
    rows = keyed.shape[0] // spec.tile_height
    cols = keyed.shape[1] // spec.tile_width
    for row in range(rows):
        for col in range(cols):
            y0 = row * spec.tile_height
            x0 = col * spec.tile_width
            tile = keyed[y0 : y0 + spec.tile_height, x0 : x0 + spec.tile_width]
            if not should_keep(tile, min_opaque_pixels):
                continue
            frame_id = f"r{row:03d}_c{col:03d}"
            group_id = group_id_for(spec, source_path, frame_id, row=row)
            yield {
                "image": tile,
                "source_path": source_path,
                "group_id": group_id,
                "frame_id": frame_id,
                "variant": "tile",
            }


def iter_component_frames(
    image: np.ndarray,
    spec: SourceSpec,
    source_path: str,
    min_opaque_pixels: int,
) -> Iterable[dict]:
    keyed = apply_background_key(image, spec.background_color)
    mask = keyed[:, :, 3] > 0
    seen = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    components: list[tuple[int, int, int, int, int]] = []

    for y in range(height):
        for x in range(width):
            if seen[y, x] or not mask[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            xs: list[int] = []
            ys: list[int] = []
            while stack:
                cy, cx = stack.pop()
                xs.append(cx)
                ys.append(cy)
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))

            count = len(xs)
            x0, x1 = min(xs), max(xs) + 1
            y0, y1 = min(ys), max(ys) + 1
            comp_w = x1 - x0
            comp_h = y1 - y0
            if count < min_opaque_pixels:
                continue
            if comp_w > spec.max_component_width or comp_h > spec.max_component_height:
                continue
            components.append((x0, y0, x1, y1, count))

    components.sort(key=lambda item: (item[1], item[0]))
    for idx, (x0, y0, x1, y1, _count) in enumerate(components):
        frame_id = f"component_{idx:04d}"
        group_id = group_id_for(spec, source_path, frame_id, row=y0 // max(1, spec.max_component_height))
        yield {
            "image": keyed[y0:y1, x0:x1],
            "source_path": source_path,
            "group_id": group_id,
            "frame_id": frame_id,
            "variant": "component",
        }


def iter_zip_frames(
    zip_path: Path,
    spec: SourceSpec,
    min_opaque_pixels: int,
) -> Iterable[dict]:
    include_re = re.compile(spec.include_pattern or r".*")
    exclude_re = re.compile(spec.exclude_pattern) if spec.exclude_pattern else None
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(name for name in zf.namelist() if not name.endswith("/"))
        for name in names:
            if not include_re.search(name):
                continue
            if exclude_re is not None and exclude_re.search(name):
                continue
            if not name.lower().endswith((".png", ".gif", ".bmp", ".jpg", ".jpeg", ".webp")):
                continue
            image = load_rgba(zf.read(name))
            source_path = f"{zip_path.name}:{name}"
            if spec.mode == "zip_grid":
                yield from iter_grid_frames(image, spec, source_path, min_opaque_pixels)
            elif spec.mode == "zip_single":
                image = apply_background_key(image, spec.background_color)
                if should_keep(image, min_opaque_pixels):
                    frame_id = safe_name(Path(name).stem)
                    yield {
                        "image": image,
                        "source_path": source_path,
                        "group_id": group_id_for(spec, source_path, frame_id),
                        "frame_id": frame_id,
                        "variant": "image",
                    }
            else:
                raise ValueError(f"Unsupported zip mode: {spec.mode}")


def group_id_for(spec: SourceSpec, source_path: str, frame_id: str, row: int | None = None) -> str:
    if spec.group_mode == "row" and row is not None:
        return f"{spec.source_id}:{safe_name(source_path)}:row_{row:03d}"
    if spec.group_mode == "component_row" and row is not None:
        return f"{spec.source_id}:component_row_{row:03d}"
    if spec.group_mode == "archive_file":
        archive_member = source_path.split(":", 1)[-1]
        return f"{spec.source_id}:{safe_name(str(Path(archive_member).with_suffix('')))}"
    if spec.group_mode == "strip_trailing_digits":
        archive_member = source_path.split(":", 1)[-1]
        stem = Path(archive_member).stem
        stem = re.sub(r"\d+$", "", stem)
        return f"{spec.source_id}:{safe_name(stem)}"
    return f"{spec.source_id}:{safe_name(source_path)}"


def write_frame(
    image: np.ndarray,
    images_dir: Path,
    source_id: str,
    frame_id: str,
    sample_index: int,
) -> Path:
    filename = f"{source_id}__{safe_name(frame_id)}__{sample_index:06d}.png"
    out_path = images_dir / filename
    Image.fromarray(image, mode="RGBA").save(out_path)
    return out_path


def build_curated_dataset(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        if not args.force:
            raise FileExistsError(f"{args.output_dir} exists; pass --force to overwrite it")
        shutil.rmtree(args.output_dir)

    images_dir = args.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    args.download_dir.mkdir(parents=True, exist_ok=True)

    samples: list[dict] = []
    source_records: list[dict] = []

    for spec in tqdm(SOURCES, desc="OpenGameArt sources"):
        download_path, sha256, size_bytes = download_file(spec, args.download_dir, args.timeout)
        source_count = 0

        if spec.mode in {"grid", "components"}:
            image = load_rgba(download_path.read_bytes())
            if spec.mode == "grid":
                frame_iter = iter_grid_frames(image, spec, download_path.name, args.min_opaque_pixels)
            else:
                frame_iter = iter_component_frames(image, spec, download_path.name, args.min_opaque_pixels)
        elif spec.mode in {"zip_grid", "zip_single"}:
            frame_iter = iter_zip_frames(download_path, spec, args.min_opaque_pixels)
        else:
            raise ValueError(f"Unsupported mode: {spec.mode}")

        for frame in frame_iter:
            if args.max_frames is not None and len(samples) >= args.max_frames:
                break
            image_path = write_frame(
                image=frame["image"],
                images_dir=images_dir,
                source_id=spec.source_id,
                frame_id=frame["frame_id"],
                sample_index=len(samples),
            )
            samples.append(
                {
                    "index": len(samples),
                    "image_path": str(image_path.relative_to(args.output_dir)),
                    "source_path": frame["source_path"],
                    "group_id": frame["group_id"],
                    "frame_id": frame["frame_id"],
                    "variant": frame["variant"],
                    "source_kind": "opengameart_public",
                    "source_id": spec.source_id,
                    "license": spec.license,
                    "author": spec.author,
                    "page_url": spec.page_url,
                }
            )
            source_count += 1

        source_record = asdict(spec)
        source_record.update(
            {
                "download_path": str(download_path),
                "sha256": sha256,
                "size_bytes": size_bytes,
                "curated_frames": source_count,
            }
        )
        source_records.append(source_record)

    manifest = {
        "dataset": "opengameart",
        "source_kind": "opengameart_public_curated",
        "images_dir": "images",
        "num_samples": len(samples),
        "group_count": len({sample["group_id"] for sample in samples}),
        "samples": samples,
    }
    source_manifest = {
        "dataset": "opengameart",
        "source_count": len(source_records),
        "sources": source_records,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.output_dir / "source_manifest.json").write_text(json.dumps(source_manifest, indent=2))

    print("OpenGameArt public curation complete")
    print(f"  output: {args.output_dir}")
    print(f"  frames: {len(samples)}")
    print(f"  groups: {manifest['group_count']}")
    for source in source_records:
        print(f"  {source['source_id']}: {source['curated_frames']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and curate public OpenGameArt sprite assets")
    parser.add_argument("--output-dir", type=Path, default=CURATED_DIR)
    parser.add_argument("--download-dir", type=Path, default=DOWNLOAD_DIR)
    parser.add_argument("--min-opaque-pixels", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build_curated_dataset(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
