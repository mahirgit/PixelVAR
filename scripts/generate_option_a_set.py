#!/usr/bin/env python3
"""Generate and package a larger Option A sprite sample set."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixelvar.data.palette import PaletteExtractor
from pixelvar.tokenizers import DeterministicPyramidTokenizer
from pixelvar.training import LitVAR
from pixelvar.utils import load_yaml, save_rgba_grid


def edge_density(index_maps: np.ndarray) -> np.ndarray:
    maps = np.asarray(index_maps)
    height, width = maps.shape[-2:]
    edge_h = maps[:, :, 1:] != maps[:, :, :-1]
    edge_v = maps[:, 1:, :] != maps[:, :-1, :]
    denominator = height * (width - 1) + (height - 1) * width
    return (edge_h.sum(axis=(1, 2)) + edge_v.sum(axis=(1, 2))) / denominator


def render_maps(index_maps: np.ndarray, palette: PaletteExtractor) -> np.ndarray:
    return np.stack([palette.render_index_map(index_map) for index_map in index_maps], axis=0)


def save_images(images: np.ndarray, output_dir: Path, start_index: int) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for offset, image in enumerate(images):
        index = start_index + offset
        path = output_dir / f"sample_{index:06d}.png"
        Image.fromarray(image, mode="RGBA").save(path)
        records.append({"index": index, "image_path": path.relative_to(output_dir.parent).as_posix()})
    return records


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_report(path: Path, manifest: dict, summary: dict) -> None:
    lines = [
        "# Option A Generated Set",
        "",
        "This package was generated from the selected Sprites full checkpoint setting.",
        "",
        "## Sampling",
        "",
        f"- Samples: `{manifest['num_samples']}`",
        f"- Temperature: `{manifest['temperature']}`",
        f"- Top-k: `{manifest['top_k']}`",
        f"- Seed: `{manifest['seed']}`",
        "",
        "## Summary",
        "",
        f"- Opaque ratio mean: `{summary['opaque_ratio_mean']:.4f}`",
        f"- Opaque ratio std: `{summary['opaque_ratio_std']:.4f}`",
        f"- Edge density mean: `{summary['edge_density_mean']:.4f}`",
        f"- Edge density std: `{summary['edge_density_std']:.4f}`",
        f"- Token min/max: `{summary['token_min']}` / `{summary['token_max']}`",
        "",
        "## Files",
        "",
        "- `tokens.npy`: generated token sequences, shape `(N, 1365)`, uint8",
        "- `index_maps.npy`: generated final 32x32 token maps, shape `(N, 32, 32)`, uint8",
        "- `images/`: first generated PNGs for inspection",
        "- `grids/`: contact-sheet PNGs",
        "- `manifest.json`: generation metadata",
        "- `summary.json`: aggregate generated-set metrics",
        f"- `{manifest['files']['zip']}`: zipped generated-set package",
        "",
        "## Contact Sheets",
        "",
    ]
    for grid_path in manifest["files"]["grids"]:
        lines.append(f"![{Path(grid_path).stem}]({grid_path})")
        lines.append("")
    path.write_text("\n".join(lines))


def zip_directory(source_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path == zip_path or path.is_dir():
                continue
            archive.write(path, path.relative_to(source_dir))


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a packaged PixelVAR Option A sample set")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/generated/option_a"))
    parser.add_argument("--num-samples", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-samples", type=int, default=64)
    parser.add_argument("--num-grids", type=int, default=16)
    parser.add_argument("--max-image-files", type=int, default=2048)
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = args.output_dir / "images"
    grids_dir = args.output_dir / "grids"
    for stale_dir in (images_dir, grids_dir):
        if stale_dir.exists():
            shutil.rmtree(stale_dir)
    for stale_file in (
        args.output_dir / "tokens.npy",
        args.output_dir / "index_maps.npy",
        args.output_dir / "summary.json",
        args.output_dir / "manifest.json",
        args.output_dir / "generation_report.md",
        args.output_dir.with_suffix(".zip"),
    ):
        if stale_file.exists():
            stale_file.unlink()
    images_dir.mkdir(parents=True, exist_ok=True)
    grids_dir.mkdir(parents=True, exist_ok=True)

    config = load_yaml(args.config)
    processed_dir = Path(config["data"]["processed_dir"])
    palette = PaletteExtractor()
    palette.load(processed_dir / "palette.json")
    tokenizer = DeterministicPyramidTokenizer(config["model"].get("scale_resolutions"))

    module = LitVAR.load_from_checkpoint(args.checkpoint)
    module.eval()
    if torch.cuda.is_available():
        module = module.cuda()

    tokens_path = args.output_dir / "tokens.npy"
    index_maps_path = args.output_dir / "index_maps.npy"
    tokens_out = np.lib.format.open_memmap(
        tokens_path,
        mode="w+",
        dtype=np.uint8,
        shape=(args.num_samples, tokenizer.sequence_length),
    )
    maps_out = np.lib.format.open_memmap(
        index_maps_path,
        mode="w+",
        dtype=np.uint8,
        shape=(args.num_samples, 32, 32),
    )

    image_records: list[dict] = []
    grid_paths: list[str] = []
    opaque_ratios = []
    edge_densities = []
    token_hist = np.zeros(int(config["model"]["vocab_size"]), dtype=np.int64)
    saved_grid_images: list[np.ndarray] = []
    max_grid_images = args.grid_samples * args.num_grids

    for start in range(0, args.num_samples, args.batch_size):
        end = min(start + args.batch_size, args.num_samples)
        current = end - start
        tokens = module.sample(batch_size=current, temperature=args.temperature, top_k=args.top_k).cpu()
        index_maps = tokenizer.from_sequence(tokens)[-1].cpu().numpy().astype(np.uint8)
        token_array = tokens.numpy().astype(np.uint8)

        tokens_out[start:end] = token_array
        maps_out[start:end] = index_maps

        opaque = index_maps != 0
        opaque_ratios.append(opaque.mean(axis=(1, 2)))
        edge_densities.append(edge_density(index_maps))
        token_hist += np.bincount(token_array.reshape(-1), minlength=len(token_hist))

        remaining_images = max(0, args.max_image_files - len(image_records))
        remaining_grids = max(0, max_grid_images - len(saved_grid_images))
        render_count = max(remaining_images, remaining_grids)
        if render_count:
            images = render_maps(index_maps[:render_count], palette)
            if remaining_images:
                image_records.extend(save_images(images[:remaining_images], images_dir, start))
            if remaining_grids:
                saved_grid_images.extend(list(images[:remaining_grids]))

        print(f"generated {end}/{args.num_samples}", flush=True)

    tokens_out.flush()
    maps_out.flush()

    for grid_idx in range(args.num_grids):
        start = grid_idx * args.grid_samples
        end = min(start + args.grid_samples, len(saved_grid_images))
        if start >= end:
            break
        grid_path = grids_dir / f"grid_{grid_idx:03d}.png"
        save_rgba_grid(np.stack(saved_grid_images[start:end], axis=0), grid_path, columns=8)
        grid_paths.append(grid_path.relative_to(args.output_dir).as_posix())

    opaque_all = np.concatenate(opaque_ratios)
    edge_all = np.concatenate(edge_densities)
    summary = {
        "num_samples": args.num_samples,
        "opaque_ratio_mean": float(opaque_all.mean()),
        "opaque_ratio_std": float(opaque_all.std()),
        "edge_density_mean": float(edge_all.mean()),
        "edge_density_std": float(edge_all.std()),
        "token_min": int(tokens_out.min()),
        "token_max": int(tokens_out.max()),
        "token_histogram": token_hist.tolist(),
        "saved_image_files": len(image_records),
        "saved_grid_files": len(grid_paths),
    }

    zip_path = args.output_dir.with_suffix(".zip")
    manifest = {
        "kind": "pixelvar_option_a_generated_set",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "processed_dir": str(processed_dir),
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "files": {
            "tokens": "tokens.npy",
            "index_maps": "index_maps.npy",
            "images": "images/",
            "grids": grid_paths,
            "summary": "summary.json",
            "report": "generation_report.md",
            "zip": zip_path.name,
        },
        "samples": image_records,
    }

    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "manifest.json", manifest)
    write_report(args.output_dir / "generation_report.md", manifest, summary)

    if not args.no_zip:
        zip_directory(args.output_dir, zip_path)
        print(f"wrote package {zip_path}", flush=True)

    print(f"wrote generated set to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
