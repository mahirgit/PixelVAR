#!/usr/bin/env python3
"""Evaluate decoded patch-VQ VAR samples against real validation sprites."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixelvar.tokenizers import DeterministicPyramidTokenizer
from pixelvar.training import load_var_model_from_checkpoint
from pixelvar.utils import load_yaml, save_rgba_grid


def parse_top_k(value: str) -> int | None:
    text = value.strip().lower()
    if text in {"none", "null", "0"}:
        return None
    return int(text)


def split_indices(manifest: dict, split: str) -> list[int]:
    indices = [
        int(sample.get("index", idx))
        for idx, sample in enumerate(manifest.get("samples", []))
        if sample.get("split") == split
    ]
    if not indices:
        raise ValueError(f"No {split!r} samples found in manifest")
    return indices


def clean_rgba(images: np.ndarray) -> np.ndarray:
    rgba = np.asarray(images, dtype=np.uint8).copy()
    if rgba.shape[-1] == 3:
        alpha = np.full((*rgba.shape[:-1], 1), 255, dtype=np.uint8)
        rgba = np.concatenate([rgba, alpha], axis=-1)
    alpha_f = rgba[..., 3:4].astype(np.float32) / 255.0
    rgba[..., :3] = np.round(rgba[..., :3].astype(np.float32) * alpha_f).astype(np.uint8)
    return rgba


def decode_patch_codes(code_maps: np.ndarray, codebook: np.ndarray, patch_size: int) -> np.ndarray:
    code_maps = np.asarray(code_maps, dtype=np.int64)
    patches = codebook[code_maps]
    n, grid_h, grid_w = code_maps.shape
    channels = patches.shape[-1] // (patch_size * patch_size)
    patches = patches.reshape(n, grid_h, grid_w, patch_size, patch_size, channels)
    images = patches.transpose(0, 1, 3, 2, 4, 5).reshape(n, grid_h * patch_size, grid_w * patch_size, channels)
    return np.round(np.clip(images, 0.0, 1.0) * 255.0).astype(np.uint8)


def entropy(probs: np.ndarray) -> np.ndarray:
    safe = np.where(probs > 0, probs, 1.0)
    return -(probs * np.log2(safe)).sum(axis=1)


def image_features(images: np.ndarray, alpha_threshold: int = 128) -> np.ndarray:
    rgba = clean_rgba(images)
    n, height, width, _ = rgba.shape
    alpha = rgba[..., 3] >= alpha_threshold
    rgb = rgba[..., :3].astype(np.float32) / 255.0

    edge_alpha_h = alpha[:, :, 1:] != alpha[:, :, :-1]
    edge_alpha_v = alpha[:, 1:, :] != alpha[:, :-1, :]
    total_edges = height * (width - 1) + (height - 1) * width
    alpha_edge_density = (edge_alpha_h.sum(axis=(1, 2)) + edge_alpha_v.sum(axis=(1, 2))) / total_edges

    rgb_diff_h = np.abs(rgb[:, :, 1:, :] - rgb[:, :, :-1, :]).mean(axis=-1)
    rgb_diff_v = np.abs(rgb[:, 1:, :, :] - rgb[:, :-1, :, :]).mean(axis=-1)
    both_h = alpha[:, :, 1:] & alpha[:, :, :-1]
    both_v = alpha[:, 1:, :] & alpha[:, :-1, :]
    rgb_edge_density = ((rgb_diff_h > 0.08) & both_h).sum(axis=(1, 2))
    rgb_edge_density += ((rgb_diff_v > 0.08) & both_v).sum(axis=(1, 2))
    rgb_edge_density = rgb_edge_density / total_edges

    color_hist = np.zeros((n, 64), dtype=np.float64)
    rgb_mean = np.zeros((n, 3), dtype=np.float64)
    rgb_std = np.zeros((n, 3), dtype=np.float64)
    bbox = np.zeros((n, 7), dtype=np.float64)
    for idx in range(n):
        mask = alpha[idx]
        if not mask.any():
            continue
        opaque_rgb = rgba[idx, :, :, :3][mask]
        bins = np.clip(opaque_rgb // 64, 0, 3)
        bin_ids = bins[:, 0] * 16 + bins[:, 1] * 4 + bins[:, 2]
        hist = np.bincount(bin_ids, minlength=64).astype(np.float64)
        color_hist[idx] = hist / hist.sum()
        rgb_mean[idx] = opaque_rgb.mean(axis=0) / 255.0
        rgb_std[idx] = opaque_rgb.std(axis=0) / 255.0

        ys, xs = np.nonzero(mask)
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
            alpha.mean(axis=(1, 2)),
            1.0 - alpha.mean(axis=(1, 2)),
            alpha_edge_density,
            rgb_edge_density,
            entropy(color_hist),
        ],
        axis=1,
    )
    return np.concatenate([scalar, rgb_mean, rgb_std, bbox, color_hist], axis=1)


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


def summarize_images(images: np.ndarray) -> dict:
    features = image_features(images)
    alpha = clean_rgba(images)[..., 3] >= 128
    return {
        "num_images": int(len(images)),
        "opaque_ratio": float(alpha.mean()),
        "alpha_edge_density": float(features[:, 2].mean()),
        "rgb_edge_density": float(features[:, 3].mean()),
        "color_entropy": float(features[:, 4].mean()),
    }


@torch.no_grad()
def sample_tokens(module, num_samples: int, batch_size: int, temperature: float, top_k: int | None) -> torch.Tensor:
    chunks = []
    remaining = num_samples
    while remaining > 0:
        current = min(batch_size, remaining)
        chunks.append(module.sample(batch_size=current, temperature=temperature, top_k=top_k).cpu())
        remaining -= current
    return torch.cat(chunks, dim=0)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict], reference_summary: dict, grid_paths: list[Path]) -> None:
    best = min(rows, key=lambda row: float(row["feature_fid"]))
    lines = [
        "# Decoded Patch-VQ Evaluation",
        "",
        "This report evaluates decoded RGBA samples against real validation sprites.",
        "The score is a lightweight Frechet distance over alpha, silhouette, edge,",
        "bounding-box, and coarse RGB-histogram features. It is not Inception FID.",
        "",
        "## Reference Validation Summary",
        "",
        f"- Samples: `{reference_summary['num_images']}`",
        f"- Opaque ratio: `{reference_summary['opaque_ratio']:.4f}`",
        f"- Alpha edge density: `{reference_summary['alpha_edge_density']:.4f}`",
        f"- RGB edge density: `{reference_summary['rgb_edge_density']:.4f}`",
        f"- Color entropy: `{reference_summary['color_entropy']:.4f}`",
        "",
        "## Sweep Results",
        "",
        "| Temp | Top-k | Feature FID | Opaque | Alpha edge | RGB edge | Color entropy |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: float(item["feature_fid"])):
        top_k = row["top_k"] if row["top_k"] else "none"
        lines.append(
            f"| {row['temperature']} | {top_k} | {float(row['feature_fid']):.5f} | "
            f"{float(row['opaque_ratio']):.4f} | {float(row['alpha_edge_density']):.4f} | "
            f"{float(row['rgb_edge_density']):.4f} | {float(row['color_entropy']):.4f} |"
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
    parser = argparse.ArgumentParser(description="Evaluate decoded patch-VQ sample quality")
    parser.add_argument("--var-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("data/processed/sprites_patchvq16"))
    parser.add_argument("--reference-dir", type=Path, default=Path("data/processed/sprites"))
    parser.add_argument("--image-array", type=str, default="originals_rgba.npy")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval/sprites_patchvq16_decoded"))
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.6, 0.8, 1.0])
    parser.add_argument("--top-k-values", nargs="+", default=["16", "32", "64"])
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
    tokenizer_manifest = json.loads((args.tokenizer_dir / "manifest.json").read_text())
    scale_resolutions = config["model"].get("scale_resolutions", tokenizer_manifest.get("scale_resolutions"))
    tokenizer = DeterministicPyramidTokenizer(scale_resolutions)
    codebook = np.load(args.tokenizer_dir / "codebook.npy")
    patch_size = int(tokenizer_manifest.get("patch_size", 2))

    reference_manifest = load_yaml(args.reference_dir / "manifest.json")
    val_indices = split_indices(reference_manifest, "val")
    if len(val_indices) > args.reference_samples:
        rng = np.random.default_rng(args.seed)
        val_indices = sorted(rng.choice(val_indices, size=args.reference_samples, replace=False).tolist())
    reference_images_all = np.load(args.reference_dir / args.image_array, mmap_mode="r")
    reference_images = clean_rgba(np.asarray(reference_images_all[val_indices]))
    reference_features = image_features(reference_images)
    reference_summary = summarize_images(reference_images)
    save_rgba_grid(reference_images[: args.grid_samples], args.output_dir / "reference_grid.png", columns=8)

    model = load_var_model_from_checkpoint(args.var_checkpoint)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    rows = []
    grid_paths = [Path("reference_grid.png")]
    for temperature in args.temperatures:
        for top_k_text in args.top_k_values:
            top_k = parse_top_k(top_k_text)
            tokens = sample_tokens(
                module=model,
                num_samples=args.num_samples,
                batch_size=args.sample_batch_size,
                temperature=temperature,
                top_k=top_k,
            )
            code_maps = tokenizer.from_sequence(tokens)[-1].cpu().numpy()
            images = decode_patch_codes(code_maps, codebook, patch_size=patch_size)
            features = image_features(images)
            summary = summarize_images(images)

            top_k_label = "none" if top_k is None else str(top_k)
            stem = f"temp_{temperature:g}_topk_{top_k_label}"
            np.save(args.output_dir / f"{stem}_tokens.npy", tokens.numpy().astype(np.uint16))
            np.save(args.output_dir / f"{stem}_code_maps.npy", code_maps.astype(np.uint16))
            save_rgba_grid(images[: args.grid_samples], args.output_dir / f"{stem}_grid.png", columns=8)
            grid_paths.append(Path(f"{stem}_grid.png"))

            rows.append(
                {
                    "temperature": f"{temperature:g}",
                    "top_k": "" if top_k is None else str(top_k),
                    "num_samples": args.num_samples,
                    "feature_fid": f"{frechet_distance(reference_features, features):.8f}",
                    "opaque_ratio": f"{summary['opaque_ratio']:.8f}",
                    "opaque_ratio_delta": f"{summary['opaque_ratio'] - reference_summary['opaque_ratio']:.8f}",
                    "alpha_edge_density": f"{summary['alpha_edge_density']:.8f}",
                    "alpha_edge_density_delta": f"{summary['alpha_edge_density'] - reference_summary['alpha_edge_density']:.8f}",
                    "rgb_edge_density": f"{summary['rgb_edge_density']:.8f}",
                    "rgb_edge_density_delta": f"{summary['rgb_edge_density'] - reference_summary['rgb_edge_density']:.8f}",
                    "color_entropy": f"{summary['color_entropy']:.8f}",
                    "color_entropy_delta": f"{summary['color_entropy'] - reference_summary['color_entropy']:.8f}",
                }
            )

    write_csv(args.output_dir / "metrics.csv", rows)
    (args.output_dir / "reference_summary.json").write_text(json.dumps(reference_summary, indent=2) + "\n")
    write_report(args.output_dir / "evaluation_report.md", rows, reference_summary, grid_paths)
    print(f"Wrote decoded patch-VQ evaluation to {args.output_dir}")


if __name__ == "__main__":
    main()
