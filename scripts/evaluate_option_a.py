#!/usr/bin/env python3
"""Evaluate Option A samples against processed validation sprites."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixelvar.data.palette import PaletteExtractor
from pixelvar.training import load_hmar_model_from_checkpoint, load_var_model_from_checkpoint
from pixelvar.tokenizers import DeterministicPyramidTokenizer
from pixelvar.utils import load_yaml, save_rgba_grid


def parse_top_k(value: str) -> int | None:
    text = value.strip().lower()
    if text in {"none", "null", "0"}:
        return None
    return int(text)


def split_indices(manifest: dict, split: str) -> list[int]:
    indices = [int(sample.get("index", idx)) for idx, sample in enumerate(manifest.get("samples", [])) if sample.get("split") == split]
    if not indices:
        raise ValueError(f"No {split!r} samples found in manifest")
    return indices


def entropy(probs: np.ndarray) -> np.ndarray:
    safe = np.where(probs > 0, probs, 1.0)
    return -(probs * np.log2(safe)).sum(axis=1)


def sprite_features(index_maps: np.ndarray, palette_size: int) -> np.ndarray:
    maps = np.asarray(index_maps, dtype=np.int64)
    opaque = maps != 0
    n, height, width = maps.shape

    hist = np.stack([np.bincount(frame.reshape(-1), minlength=palette_size + 1) for frame in maps], axis=0).astype(np.float64)
    hist = hist / hist.sum(axis=1, keepdims=True)
    opaque_hist = hist[:, 1:]
    opaque_hist_sum = opaque_hist.sum(axis=1, keepdims=True)
    opaque_hist_norm = np.divide(opaque_hist, opaque_hist_sum, out=np.zeros_like(opaque_hist), where=opaque_hist_sum > 0)

    edge_h = maps[:, :, 1:] != maps[:, :, :-1]
    edge_v = maps[:, 1:, :] != maps[:, :-1, :]
    edge_density_h = edge_h.mean(axis=(1, 2))
    edge_density_v = edge_v.mean(axis=(1, 2))
    edge_density = (edge_h.sum(axis=(1, 2)) + edge_v.sum(axis=(1, 2))) / (height * (width - 1) + (height - 1) * width)

    bbox = np.zeros((n, 7), dtype=np.float64)
    for idx, mask in enumerate(opaque):
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        box_w = (x1 - x0 + 1) / width
        box_h = (y1 - y0 + 1) / height
        bbox[idx] = [
            box_w,
            box_h,
            box_w * box_h,
            xs.mean() / (width - 1),
            ys.mean() / (height - 1),
            x0 / (width - 1),
            y0 / (height - 1),
        ]

    scalar = np.stack(
        [
            opaque.mean(axis=(1, 2)),
            1.0 - opaque.mean(axis=(1, 2)),
            edge_density,
            edge_density_h,
            edge_density_v,
            entropy(opaque_hist_norm),
        ],
        axis=1,
    )
    return np.concatenate([scalar, bbox, hist], axis=1)


def frechet_distance(real_features: np.ndarray, gen_features: np.ndarray) -> float:
    real_features = np.asarray(real_features, dtype=np.float64)
    gen_features = np.asarray(gen_features, dtype=np.float64)
    mu_real = real_features.mean(axis=0)
    mu_gen = gen_features.mean(axis=0)
    cov_real = np.cov(real_features, rowvar=False)
    cov_gen = np.cov(gen_features, rowvar=False)
    eps = 1e-6
    cov_real = cov_real + np.eye(cov_real.shape[0]) * eps
    cov_gen = cov_gen + np.eye(cov_gen.shape[0]) * eps

    try:
        from scipy.linalg import sqrtm

        covmean = sqrtm(cov_real @ cov_gen)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
    except Exception:
        values, vectors = np.linalg.eigh(cov_real @ cov_gen)
        values = np.clip(values, 0.0, None)
        covmean = (vectors * np.sqrt(values)) @ vectors.T

    diff = mu_real - mu_gen
    score = float(diff @ diff + np.trace(cov_real + cov_gen - 2.0 * covmean))
    return max(score, 0.0)


def render_maps(index_maps: np.ndarray, palette: PaletteExtractor) -> np.ndarray:
    return np.stack([palette.render_index_map(index_map) for index_map in index_maps], axis=0)


def palette_consistency(images: np.ndarray, palette: PaletteExtractor) -> float:
    colors = {tuple(map(int, color)) for color in palette.palette}
    total = 0
    valid = 0
    for image in images:
        opaque = image[:, :, 3] > 0
        rgb = image[:, :, :3][opaque]
        total += len(rgb)
        valid += sum(tuple(map(int, color)) in colors for color in rgb)
    if total == 0:
        return 1.0
    return valid / total


@torch.no_grad()
def sample_tokens(
    module,
    num_samples: int,
    batch_size: int,
    temperature: float,
    top_k: int | None,
    sample_kwargs: dict | None = None,
) -> torch.Tensor:
    sample_kwargs = sample_kwargs or {}
    chunks = []
    remaining = num_samples
    while remaining > 0:
        current = min(batch_size, remaining)
        chunks.append(module.sample(batch_size=current, temperature=temperature, top_k=top_k, **sample_kwargs).cpu())
        remaining -= current
    return torch.cat(chunks, dim=0)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict], reference_summary: dict, grid_paths: list[Path], model_label: str) -> None:
    best = min(rows, key=lambda row: float(row["feature_fid"]))
    lines = [
        "# PixelVAR Evaluation",
        "",
        f"This report evaluates generated {model_label} sprites against the validation split.",
        "The FID-style score below is a lightweight Frechet distance over handcrafted",
        "palette, transparency, silhouette, and edge features. It is not Inception FID.",
        "",
        "## Reference Validation Summary",
        "",
        f"- Samples: `{reference_summary['num_reference']}`",
        f"- Opaque ratio: `{reference_summary['opaque_ratio']:.4f}`",
        f"- Edge density: `{reference_summary['edge_density']:.4f}`",
        f"- Palette consistency: `{reference_summary['palette_consistency']:.4f}`",
        "",
        "## Sweep Results",
        "",
        "| Temp | Top-k | Feature FID | Palette consistency | Opaque ratio | Edge density |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: float(item["feature_fid"])):
        top_k = row["top_k"] if row["top_k"] else "none"
        lines.append(
            f"| {row['temperature']} | {top_k} | {float(row['feature_fid']):.5f} | "
            f"{float(row['palette_consistency']):.4f} | {float(row['opaque_ratio']):.4f} | "
            f"{float(row['edge_density']):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Best Setting",
            "",
            f"- Temperature: `{best['temperature']}`",
            f"- Top-k: `{best['top_k'] or 'none'}`",
            f"- Feature FID: `{float(best['feature_fid']):.5f}`",
            "",
            "## Sample Grids",
            "",
        ]
    )
    for grid_path in grid_paths:
        lines.append(f"![{grid_path.stem}]({grid_path.as_posix()})")
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Option A checkpoint sample quality")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-kind", choices=["var", "hmar"], default="var")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval/option_a"))
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.6, 0.8, 1.0])
    parser.add_argument("--top-k-values", nargs="+", default=["8", "16", "none"])
    parser.add_argument("--refinement-steps", type=int, default=4)
    parser.add_argument("--mask-schedule", choices=["cosine", "linear"], default="cosine")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--sample-batch-size", type=int, default=64)
    parser.add_argument("--grid-samples", type=int, default=64)
    parser.add_argument("--reference-samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = load_yaml(args.config)
    processed_dir = Path(config["data"]["processed_dir"])
    palette = PaletteExtractor()
    palette.load(processed_dir / "palette.json")
    palette_size = len(palette.palette)

    manifest = load_yaml(processed_dir / "manifest.json")
    val_indices = split_indices(manifest, "val")
    if len(val_indices) > args.reference_samples:
        rng = np.random.default_rng(args.seed)
        val_indices = sorted(rng.choice(val_indices, size=args.reference_samples, replace=False).tolist())

    index_maps = np.load(processed_dir / "index_maps.npy", mmap_mode="r")
    reference_maps = np.asarray(index_maps[val_indices])
    reference_images = render_maps(reference_maps, palette)
    reference_features = sprite_features(reference_maps, palette_size)
    reference_summary = {
        "num_reference": len(reference_maps),
        "opaque_ratio": float((reference_maps != 0).mean()),
        "edge_density": float(reference_features[:, 2].mean()),
        "palette_consistency": palette_consistency(reference_images, palette),
    }

    if args.model_kind == "hmar":
        module = load_hmar_model_from_checkpoint(args.checkpoint)
        sample_kwargs = {
            "refinement_steps": args.refinement_steps,
            "mask_schedule": args.mask_schedule,
        }
    else:
        module = load_var_model_from_checkpoint(args.checkpoint)
        sample_kwargs = {}
    module.eval()
    if torch.cuda.is_available():
        module = module.cuda()

    tokenizer = DeterministicPyramidTokenizer(config["model"].get("scale_resolutions"))
    rows = []
    grid_paths = []
    for temperature in args.temperatures:
        for top_k_text in args.top_k_values:
            top_k = parse_top_k(top_k_text)
            tokens = sample_tokens(
                module=module,
                num_samples=args.num_samples,
                batch_size=args.sample_batch_size,
                temperature=temperature,
                top_k=top_k,
                sample_kwargs=sample_kwargs,
            )
            maps = tokenizer.from_sequence(tokens)[-1].cpu().numpy()
            images = render_maps(maps, palette)
            features = sprite_features(maps, palette_size)

            top_k_label = "none" if top_k is None else str(top_k)
            stem = f"temp_{temperature:g}_topk_{top_k_label}"
            np.save(args.output_dir / f"{stem}_tokens.npy", tokens.numpy().astype(np.uint8))
            save_rgba_grid(images[: args.grid_samples], args.output_dir / f"{stem}_grid.png", columns=8)
            grid_paths.append(Path(f"{stem}_grid.png"))

            rows.append(
                {
                    "temperature": f"{temperature:g}",
                    "top_k": "" if top_k is None else str(top_k),
                    "num_samples": args.num_samples,
                    "feature_fid": f"{frechet_distance(reference_features, features):.8f}",
                    "palette_consistency": f"{palette_consistency(images, palette):.8f}",
                    "opaque_ratio": f"{float((maps != 0).mean()):.8f}",
                    "opaque_ratio_delta": f"{float((maps != 0).mean() - reference_summary['opaque_ratio']):.8f}",
                    "edge_density": f"{float(features[:, 2].mean()):.8f}",
                    "edge_density_delta": f"{float(features[:, 2].mean() - reference_summary['edge_density']):.8f}",
                    "token_entropy": f"{float(features[:, 5].mean()):.8f}",
                }
            )

    write_csv(args.output_dir / "metrics.csv", rows)
    (args.output_dir / "reference_summary.json").write_text(json.dumps(reference_summary, indent=2) + "\n")
    model_label = "HMAR Option B" if args.model_kind == "hmar" else "Option A"
    write_report(args.output_dir / "evaluation_report.md", rows, reference_summary, grid_paths, model_label)
    print(f"Wrote evaluation to {args.output_dir}")


if __name__ == "__main__":
    main()
