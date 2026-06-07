#!/usr/bin/env python3
"""Export validation and generated sprites as individual PNG files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixelvar.data.palette import PaletteExtractor
from pixelvar.tokenizers import DeterministicPyramidTokenizer
from pixelvar.training import (
    load_flat_ar_model_from_checkpoint,
    load_flat_maskgit_model_from_checkpoint,
    load_hmar_model_from_checkpoint,
    load_var_model_from_checkpoint,
)
from pixelvar.utils import load_yaml

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_top_k(value: str | None) -> int | None:
    if value is None:
        return None
    text = value.strip().lower()
    if text in {"", "none", "null", "0"}:
        return None
    return int(text)


def split_indices(manifest: dict, split: str) -> list[int]:
    indices = [int(sample.get("index", idx)) for idx, sample in enumerate(manifest.get("samples", [])) if sample.get("split") == split]
    if not indices:
        raise ValueError(f"No {split!r} samples found in manifest")
    return indices


def save_images(images: np.ndarray, output_dir: Path, prefix: str) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            path.unlink()
    rows = []
    for idx, image in enumerate(images):
        path = output_dir / f"{prefix}_{idx:06d}.png"
        Image.fromarray(image.astype(np.uint8), mode="RGBA").save(path)
        rows.append({"index": idx, "path": path.as_posix()})
    return rows


def render_index_maps(index_maps: np.ndarray, palette: PaletteExtractor) -> np.ndarray:
    return np.stack([palette.render_index_map(index_map) for index_map in index_maps], axis=0)


@torch.no_grad()
def sample_model(
    checkpoint: Path,
    config: dict,
    model_kind: str,
    num_samples: int,
    batch_size: int,
    temperature: float,
    top_k: int | None,
    refinement_steps: int,
    mask_schedule: str,
) -> torch.Tensor:
    if model_kind == "hmar":
        module = load_hmar_model_from_checkpoint(checkpoint)
        sample_kwargs = {"refinement_steps": refinement_steps, "mask_schedule": mask_schedule}
    elif model_kind == "flat_ar":
        module = load_flat_ar_model_from_checkpoint(checkpoint)
        sample_kwargs = {}
    elif model_kind == "flat_maskgit":
        module = load_flat_maskgit_model_from_checkpoint(checkpoint)
        sample_kwargs = {"refinement_steps": refinement_steps, "mask_schedule": mask_schedule}
    else:
        module = load_var_model_from_checkpoint(checkpoint)
        sample_kwargs = {}

    module.eval()
    if torch.cuda.is_available():
        module = module.cuda()

    chunks = []
    remaining = num_samples
    while remaining > 0:
        current = min(batch_size, remaining)
        chunk = module.sample(batch_size=current, temperature=temperature, top_k=top_k, **sample_kwargs).cpu()
        chunks.append(chunk)
        remaining -= current
    tokens = torch.cat(chunks, dim=0)
    if model_kind in {"flat_ar", "flat_maskgit"}:
        tokens = module.full_sequence_from_final_tokens(tokens)
    return tokens


def load_token_files(paths: list[Path]) -> torch.Tensor:
    chunks = []
    for path in paths:
        array = np.load(path)
        chunks.append(torch.as_tensor(array, dtype=torch.long))
    if not chunks:
        raise ValueError("No token files were provided")
    return torch.cat(chunks, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PixelVAR reference/generated images for external evaluation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval_images"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-reference", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-kind", choices=["var", "hmar", "flat_ar", "flat_maskgit"], default="var")
    parser.add_argument("--generated-name", default="generated")
    parser.add_argument("--num-generated", type=int, default=4096)
    parser.add_argument("--sample-batch-size", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", default="8")
    parser.add_argument("--refinement-steps", type=int, default=1)
    parser.add_argument("--mask-schedule", choices=["cosine", "linear"], default="cosine")
    parser.add_argument("--token-file", type=Path, action="append", default=[])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    config = load_yaml(args.config)
    processed_dir = Path(config["data"]["processed_dir"])

    palette = PaletteExtractor()
    palette.load(processed_dir / "palette.json")

    manifest = load_yaml(processed_dir / "manifest.json")
    indices = split_indices(manifest, args.split)
    if len(indices) > args.num_reference:
        indices = sorted(rng.choice(indices, size=args.num_reference, replace=False).tolist())

    index_maps = np.load(processed_dir / "index_maps.npy", mmap_mode="r")
    reference_images = render_index_maps(np.asarray(index_maps[indices]), palette)

    manifest_rows = {
        "config": args.config.as_posix(),
        "processed_dir": processed_dir.as_posix(),
        "split": args.split,
        "seed": args.seed,
        "reference": save_images(reference_images, args.output_dir / "reference", "ref"),
        "generated": [],
    }

    if args.checkpoint or args.token_file:
        tokenizer = DeterministicPyramidTokenizer(config["model"].get("scale_resolutions"))
        if args.token_file:
            tokens = load_token_files(args.token_file)
            if len(tokens) > args.num_generated:
                tokens = tokens[: args.num_generated]
        else:
            tokens = sample_model(
                checkpoint=args.checkpoint,
                config=config,
                model_kind=args.model_kind,
                num_samples=args.num_generated,
                batch_size=args.sample_batch_size,
                temperature=args.temperature,
                top_k=parse_top_k(args.top_k),
                refinement_steps=args.refinement_steps,
                mask_schedule=args.mask_schedule,
            )
        generated_maps = tokenizer.from_sequence(tokens)[-1].cpu().numpy()
        generated_images = render_index_maps(generated_maps, palette)
        manifest_rows["generated"] = save_images(generated_images, args.output_dir / args.generated_name, "gen")

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest_rows, indent=2) + "\n")
    print(f"Wrote exported images to {args.output_dir}")


if __name__ == "__main__":
    main()
