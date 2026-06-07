#!/usr/bin/env python3
"""Evaluate generated sprite folders against a common reference image folder."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass(frozen=True)
class ImageSet:
    name: str
    paths: list[Path]
    rgba: np.ndarray


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" in value:
        name, path = value.split("=", 1)
        return name.strip(), Path(path)
    path = Path(value)
    return path.name, path


def image_paths(directory: Path) -> list[Path]:
    paths = [path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS]
    if not paths:
        raise ValueError(f"No image files found in {directory}")
    return sorted(paths)


def choose_paths(paths: list[Path], max_images: int | None, seed: int) -> list[Path]:
    if max_images is None or max_images <= 0 or len(paths) <= max_images:
        return paths
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(paths), size=max_images, replace=False)
    return [paths[int(idx)] for idx in sorted(selected)]


def load_rgba_images(paths: list[Path], image_size: int) -> np.ndarray:
    images = []
    for path in paths:
        image = Image.open(path).convert("RGBA")
        if image.size != (image_size, image_size):
            image = image.resize((image_size, image_size), Image.Resampling.NEAREST)
        images.append(np.asarray(image, dtype=np.uint8))
    return np.stack(images, axis=0)


def load_image_set(name: str, directory: Path, image_size: int, max_images: int | None, seed: int) -> ImageSet:
    paths = choose_paths(image_paths(directory), max_images, seed)
    return ImageSet(name=name, paths=paths, rgba=load_rgba_images(paths, image_size))


def composite_rgb(rgba: np.ndarray, background: str) -> np.ndarray:
    rgb = rgba[:, :, :, :3].astype(np.float32)
    alpha = rgba[:, :, :, 3:4].astype(np.float32) / 255.0
    if background == "black":
        bg = np.zeros_like(rgb)
    elif background == "gray":
        bg = np.full_like(rgb, 127.5)
    else:
        bg = np.full_like(rgb, 255.0)
    return (rgb * alpha + bg * (1.0 - alpha)).astype(np.uint8)


def rgba_to_inception_tensor(rgba: np.ndarray, background: str) -> torch.Tensor:
    rgb = composite_rgb(rgba, background).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(0, 3, 1, 2)
    tensor = F.interpolate(tensor, size=(299, 299), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(1, 3, 1, 1)
    return (tensor - mean) / std


def build_inception(device: torch.device) -> torch.nn.Module:
    from torchvision.models import Inception_V3_Weights, inception_v3

    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights, transform_input=False)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    return model


@torch.no_grad()
def inception_features(rgba: np.ndarray, batch_size: int, background: str, device: torch.device) -> np.ndarray:
    model = build_inception(device)
    chunks = []
    for start in range(0, len(rgba), batch_size):
        batch = rgba_to_inception_tensor(rgba[start : start + batch_size], background).to(device)
        features = model(batch)
        if isinstance(features, tuple):
            features = features[0]
        chunks.append(features.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(chunks, axis=0)


def pixel_features(rgba: np.ndarray) -> np.ndarray:
    rgba_f = rgba.astype(np.float64) / 255.0
    alpha = rgba_f[:, :, :, 3]
    rgb = rgba_f[:, :, :, :3]
    opaque = alpha >= 0.5
    edge_h = np.abs(rgba_f[:, :, 1:, :] - rgba_f[:, :, :-1, :]).mean(axis=(1, 2, 3))
    edge_v = np.abs(rgba_f[:, 1:, :, :] - rgba_f[:, :-1, :, :]).mean(axis=(1, 2, 3))
    bins = []
    for channel in range(3):
        channel_values = rgb[:, :, :, channel]
        hist = np.stack([np.histogram(frame[mask], bins=8, range=(0, 1))[0] if mask.any() else np.zeros(8) for frame, mask in zip(channel_values, opaque)], axis=0)
        bins.append(hist / np.maximum(hist.sum(axis=1, keepdims=True), 1))
    scalar = np.stack(
        [
            alpha.mean(axis=(1, 2)),
            opaque.mean(axis=(1, 2)),
            edge_h,
            edge_v,
            rgb.mean(axis=(1, 2, 3)),
            rgb.std(axis=(1, 2, 3)),
        ],
        axis=1,
    )
    return np.concatenate([scalar, *bins], axis=1)


def frechet_distance(real_features: np.ndarray, gen_features: np.ndarray) -> float:
    real = np.asarray(real_features, dtype=np.float64)
    gen = np.asarray(gen_features, dtype=np.float64)
    mu_real = real.mean(axis=0)
    mu_gen = gen.mean(axis=0)
    cov_real = np.cov(real, rowvar=False) + np.eye(real.shape[1]) * 1e-6
    cov_gen = np.cov(gen, rowvar=False) + np.eye(gen.shape[1]) * 1e-6
    try:
        from scipy.linalg import sqrtm

        covmean = sqrtm(cov_real @ cov_gen)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
    except Exception:
        values, vectors = np.linalg.eig(cov_real @ cov_gen)
        covmean = (vectors @ np.diag(np.sqrt(np.clip(values.real, 0.0, None))) @ np.linalg.inv(vectors)).real
    diff = mu_real - mu_gen
    score = float(diff @ diff + np.trace(cov_real + cov_gen - 2.0 * covmean))
    return max(score, 0.0)


def polynomial_mmd(real: np.ndarray, gen: np.ndarray, subsets: int, subset_size: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    count = min(len(real), len(gen), subset_size)
    if count < 2:
        return math.nan, math.nan
    dim = real.shape[1]
    values = []
    for _ in range(subsets):
        real_idx = rng.choice(len(real), size=count, replace=False)
        gen_idx = rng.choice(len(gen), size=count, replace=False)
        x = real[real_idx]
        y = gen[gen_idx]
        kxx = ((x @ x.T) / dim + 1.0) ** 3
        kyy = ((y @ y.T) / dim + 1.0) ** 3
        kxy = ((x @ y.T) / dim + 1.0) ** 3
        mmd = (kxx.sum() - np.trace(kxx)) / (count * (count - 1))
        mmd += (kyy.sum() - np.trace(kyy)) / (count * (count - 1))
        mmd -= 2.0 * kxy.mean()
        values.append(float(mmd))
    return float(np.mean(values)), float(np.std(values, ddof=1) if len(values) > 1 else 0.0)


def prdc(real: np.ndarray, gen: np.ndarray, nearest_k: int, device: torch.device) -> dict[str, float]:
    real_t = torch.from_numpy(real.astype(np.float32)).to(device)
    gen_t = torch.from_numpy(gen.astype(np.float32)).to(device)
    real_nn_k = min(nearest_k, len(real_t) - 1)
    gen_nn_k = min(nearest_k, len(gen_t) - 1)
    if real_nn_k < 1 or gen_nn_k < 1:
        return {"precision": math.nan, "recall": math.nan, "density": math.nan, "coverage": math.nan}

    real_real = torch.cdist(real_t, real_t)
    gen_gen = torch.cdist(gen_t, gen_t)
    real_radii = real_real.kthvalue(k=real_nn_k + 1, dim=1).values
    gen_radii = gen_gen.kthvalue(k=gen_nn_k + 1, dim=1).values
    real_gen = torch.cdist(real_t, gen_t)

    precision = (real_gen <= real_radii[:, None]).any(dim=0).float().mean().item()
    recall = (real_gen <= gen_radii[None, :]).any(dim=1).float().mean().item()
    density = (real_gen <= real_radii[:, None]).sum(dim=0).float().mean().item() / real_nn_k
    nearest_gen_dist = real_gen.min(dim=1).values
    coverage = (nearest_gen_dist <= real_radii).float().mean().item()
    return {"precision": precision, "recall": recall, "density": density, "coverage": coverage}


def ssim_components(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = x.mean(dim=(-2, -1), keepdim=True)
    mu_y = y.mean(dim=(-2, -1), keepdim=True)
    var_x = ((x - mu_x) ** 2).mean(dim=(-2, -1), keepdim=True)
    var_y = ((y - mu_y) ** 2).mean(dim=(-2, -1), keepdim=True)
    cov = ((x - mu_x) * (y - mu_y)).mean(dim=(-2, -1), keepdim=True)
    luminance = (2 * mu_x * mu_y + c1) / (mu_x**2 + mu_y**2 + c1)
    contrast_structure = (2 * cov + c2) / (var_x + var_y + c2)
    return luminance.flatten(), contrast_structure.flatten()


def ms_ssim_pair(x: torch.Tensor, y: torch.Tensor, levels: int = 4) -> float:
    weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363], dtype=x.dtype)
    weights = weights[:levels] / weights[:levels].sum()
    mcs = []
    current_x = x
    current_y = y
    for level in range(levels):
        luminance, contrast_structure = ssim_components(current_x, current_y)
        if level == levels - 1:
            value = torch.clamp(luminance, min=1e-6) ** weights[level]
        else:
            value = torch.clamp(contrast_structure, min=1e-6) ** weights[level]
            mcs.append(value)
            current_x = F.avg_pool2d(current_x, kernel_size=2, stride=2)
            current_y = F.avg_pool2d(current_y, kernel_size=2, stride=2)
    for value_part in mcs:
        value = value * value_part
    return float(torch.clamp(value, 0.0, 1.0).item())


def mean_ms_ssim(rgba: np.ndarray, pairs: int, seed: int) -> float:
    if len(rgba) < 2 or pairs <= 0:
        return math.nan
    rgb = composite_rgb(rgba, "white").astype(np.float32) / 255.0
    gray = (0.299 * rgb[:, :, :, 0] + 0.587 * rgb[:, :, :, 1] + 0.114 * rgb[:, :, :, 2])
    tensor = torch.from_numpy(gray[:, None, :, :])
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(pairs):
        i, j = rng.choice(len(tensor), size=2, replace=False)
        values.append(ms_ssim_pair(tensor[int(i) : int(i) + 1], tensor[int(j) : int(j) + 1]))
    return float(np.mean(values))


def domain_metrics(rgba: np.ndarray, palette: np.ndarray | None) -> dict[str, float]:
    alpha = rgba[:, :, :, 3]
    opaque = alpha >= 128
    rgb = rgba[:, :, :, :3]
    edge_h = np.any(rgba[:, :, 1:, :] != rgba[:, :, :-1, :], axis=3).mean(axis=(1, 2))
    edge_v = np.any(rgba[:, 1:, :, :] != rgba[:, :-1, :, :], axis=3).mean(axis=(1, 2))
    unique_colors = []
    palette_consistent = []
    palette_set = {tuple(map(int, color)) for color in palette} if palette is not None else None
    for image_rgb, mask in zip(rgb, opaque):
        colors = image_rgb[mask]
        if len(colors) == 0:
            unique_colors.append(0)
            palette_consistent.append(1.0)
            continue
        unique_colors.append(len(np.unique(colors, axis=0)))
        if palette_set is not None:
            valid = sum(tuple(map(int, color)) in palette_set for color in colors)
            palette_consistent.append(valid / len(colors))
    result = {
        "opaque_ratio": float(opaque.mean()),
        "edge_density": float(((edge_h + edge_v) / 2.0).mean()),
        "unique_colors_mean": float(np.mean(unique_colors)),
        "unique_colors_p95": float(np.percentile(unique_colors, 95)),
    }
    if palette_set is not None:
        result["palette_consistency"] = float(np.mean(palette_consistent))
    return result


def pixel_nearest_neighbor(real_rgba: np.ndarray, gen_rgba: np.ndarray, device: torch.device, batch_size: int) -> dict[str, float]:
    real = torch.from_numpy((real_rgba.astype(np.float32) / 255.0).reshape(len(real_rgba), -1)).to(device)
    gen = torch.from_numpy((gen_rgba.astype(np.float32) / 255.0).reshape(len(gen_rgba), -1)).to(device)
    nearest_chunks = []
    for start in range(0, len(gen), batch_size):
        dist = torch.cdist(gen[start : start + batch_size], real)
        nearest_chunks.append(dist.min(dim=1).values.detach().cpu())
    nearest = torch.cat(nearest_chunks).numpy()
    return {
        "pixel_nn_l2_mean": float(nearest.mean()),
        "pixel_nn_l2_p05": float(np.percentile(nearest, 5)),
        "pixel_exact_match_rate": float((nearest == 0).mean()),
    }


def load_palette(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    data = json.loads(path.read_text())
    return np.asarray(data["colors"], dtype=np.uint8)


def write_outputs(output_dir: Path, rows: list[dict], reference_name: str, feature_space: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# External Image-Folder Evaluation",
        "",
        f"Reference folder: `{reference_name}`",
        f"Feature space: `{feature_space}`",
        "",
        "Lower is better for FID, KID, mean MS-SSIM, and pixel nearest-neighbor distance.",
        "Higher is better for precision, recall, density, coverage, and palette consistency.",
        "",
        "| Method | Images | FID | KID mean | Precision | Recall | Density | Coverage | MS-SSIM | Palette consistency | Opaque ratio | Edge density |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['num_images']} | {float(row['fid']):.4f} | "
            f"{float(row['kid_mean']):.6f} | {float(row['precision']):.4f} | "
            f"{float(row['recall']):.4f} | {float(row['density']):.4f} | "
            f"{float(row['coverage']):.4f} | {float(row['mean_ms_ssim']):.4f} | "
            f"{float(row.get('palette_consistency', math.nan)):.4f} | "
            f"{float(row['opaque_ratio']):.4f} | {float(row['edge_density']):.4f} |"
        )
    (output_dir / "evaluation_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated image folders with standard and pixel-art metrics")
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--generated-dir", action="append", required=True, help="Either PATH or NAME=PATH")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/external_eval"))
    parser.add_argument("--palette-json", type=Path)
    parser.add_argument("--feature-space", choices=["inception", "pixel"], default="inception")
    parser.add_argument("--background", choices=["white", "gray", "black"], default="white")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--max-images", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--kid-subsets", type=int, default=50)
    parser.add_argument("--kid-subset-size", type=int, default=1000)
    parser.add_argument("--prdc-k", type=int, default=5)
    parser.add_argument("--msssim-pairs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    palette = load_palette(args.palette_json)
    reference = load_image_set("reference", args.reference_dir, args.image_size, args.max_images, args.seed)
    generated_sets = [
        load_image_set(name, path, args.image_size, args.max_images, args.seed + idx + 1)
        for idx, (name, path) in enumerate(parse_named_path(value) for value in args.generated_dir)
    ]

    device = torch.device(args.device)
    if args.feature_space == "inception":
        reference_features = inception_features(reference.rgba, args.batch_size, args.background, device)
        feature_sets = {
            item.name: inception_features(item.rgba, args.batch_size, args.background, device)
            for item in generated_sets
        }
    else:
        reference_features = pixel_features(reference.rgba)
        feature_sets = {item.name: pixel_features(item.rgba) for item in generated_sets}

    reference_domain = domain_metrics(reference.rgba, palette)
    reference_ms_ssim = mean_ms_ssim(reference.rgba, args.msssim_pairs, args.seed)
    rows = []
    for idx, item in enumerate(generated_sets):
        features = feature_sets[item.name]
        domain = domain_metrics(item.rgba, palette)
        nearest = pixel_nearest_neighbor(reference.rgba, item.rgba, device, args.batch_size)
        prdc_scores = prdc(reference_features, features, args.prdc_k, device)
        kid_mean, kid_std = polynomial_mmd(reference_features, features, args.kid_subsets, args.kid_subset_size, args.seed + idx)
        row = {
            "method": item.name,
            "num_reference": len(reference.rgba),
            "num_images": len(item.rgba),
            "feature_space": args.feature_space,
            "fid": f"{frechet_distance(reference_features, features):.8f}",
            "kid_mean": f"{kid_mean:.8f}",
            "kid_std": f"{kid_std:.8f}",
            "precision": f"{prdc_scores['precision']:.8f}",
            "recall": f"{prdc_scores['recall']:.8f}",
            "density": f"{prdc_scores['density']:.8f}",
            "coverage": f"{prdc_scores['coverage']:.8f}",
            "mean_ms_ssim": f"{mean_ms_ssim(item.rgba, args.msssim_pairs, args.seed + idx + 100):.8f}",
            "reference_mean_ms_ssim": f"{reference_ms_ssim:.8f}",
            "opaque_ratio": f"{domain['opaque_ratio']:.8f}",
            "reference_opaque_ratio": f"{reference_domain['opaque_ratio']:.8f}",
            "opaque_ratio_delta": f"{domain['opaque_ratio'] - reference_domain['opaque_ratio']:.8f}",
            "edge_density": f"{domain['edge_density']:.8f}",
            "reference_edge_density": f"{reference_domain['edge_density']:.8f}",
            "edge_density_delta": f"{domain['edge_density'] - reference_domain['edge_density']:.8f}",
            "unique_colors_mean": f"{domain['unique_colors_mean']:.8f}",
            "unique_colors_p95": f"{domain['unique_colors_p95']:.8f}",
            **{key: f"{value:.8f}" for key, value in nearest.items()},
        }
        if "palette_consistency" in domain:
            row["palette_consistency"] = f"{domain['palette_consistency']:.8f}"
            row["reference_palette_consistency"] = f"{reference_domain['palette_consistency']:.8f}"
        rows.append(row)

    write_outputs(args.output_dir, rows, args.reference_dir.as_posix(), args.feature_space)
    print(f"Wrote external evaluation to {args.output_dir}")


if __name__ == "__main__":
    main()
