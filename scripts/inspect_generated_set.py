#!/usr/bin/env python3
"""Inspect and conservatively filter a generated PixelVAR sample package."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixelvar.data.palette import PaletteExtractor
from pixelvar.utils import save_rgba_grid


def load_json(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text())


def edge_density(index_maps: np.ndarray) -> np.ndarray:
    height, width = index_maps.shape[-2:]
    edge_h = index_maps[:, :, 1:] != index_maps[:, :, :-1]
    edge_v = index_maps[:, 1:, :] != index_maps[:, :-1, :]
    return (edge_h.sum(axis=(1, 2)) + edge_v.sum(axis=(1, 2))) / (height * (width - 1) + (height - 1) * width)


def entropy(probs: np.ndarray) -> np.ndarray:
    safe = np.where(probs > 0, probs, 1.0)
    return -(probs * np.log2(safe)).sum(axis=1)


def largest_component_features(mask: np.ndarray) -> tuple[float, int]:
    total = int(mask.sum())
    if total == 0:
        return 0.0, 0

    seen = np.zeros(mask.shape, dtype=bool)
    largest = 0
    components = 0
    height, width = mask.shape
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        components += 1
        stack = [(int(y), int(x))]
        seen[y, x] = True
        size = 0
        while stack:
            cy, cx = stack.pop()
            size += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        largest = max(largest, size)
    return largest / total, components


def compute_features(index_maps: np.ndarray, vocab_size: int) -> dict[str, np.ndarray]:
    maps = np.asarray(index_maps, dtype=np.uint8)
    opaque = maps != 0
    height, width = maps.shape[-2:]

    bbox = np.zeros((len(maps), 7), dtype=np.float64)
    largest_ratio = np.zeros(len(maps), dtype=np.float64)
    component_count = np.zeros(len(maps), dtype=np.int32)
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
        largest_ratio[idx], component_count[idx] = largest_component_features(mask)

    hist = np.stack([np.bincount(frame.reshape(-1), minlength=vocab_size) for frame in maps], axis=0).astype(np.float64)
    hist = hist / hist.sum(axis=1, keepdims=True)
    opaque_hist = hist[:, 1:]
    opaque_hist_sum = opaque_hist.sum(axis=1, keepdims=True)
    opaque_hist_norm = np.divide(opaque_hist, opaque_hist_sum, out=np.zeros_like(opaque_hist), where=opaque_hist_sum > 0)

    return {
        "opaque_ratio": opaque.mean(axis=(1, 2)),
        "edge_density": edge_density(maps),
        "bbox_w": bbox[:, 0],
        "bbox_h": bbox[:, 1],
        "bbox_area": bbox[:, 2],
        "center_x": bbox[:, 3],
        "center_y": bbox[:, 4],
        "bbox_x0": bbox[:, 5],
        "bbox_y0": bbox[:, 6],
        "largest_component_ratio": largest_ratio,
        "component_count": component_count.astype(np.float64),
        "token_entropy": entropy(opaque_hist_norm),
        "unique_opaque_tokens": (opaque_hist > 0).sum(axis=1).astype(np.float64),
    }


def robust_z(values: np.ndarray, target: float | None = None) -> np.ndarray:
    center = float(np.median(values)) if target is None else float(target)
    mad = float(np.median(np.abs(values - np.median(values))))
    scale = 1.4826 * mad if mad > 0 else float(values.std())
    if scale <= 1e-9:
        return np.zeros_like(values, dtype=np.float64)
    return np.abs(values - center) / scale


def reasons_for_sample(features: dict[str, np.ndarray], idx: int, thresholds: dict) -> list[str]:
    reasons = []
    if features["opaque_ratio"][idx] < thresholds["opaque_min"]:
        reasons.append("too_sparse")
    if features["opaque_ratio"][idx] > thresholds["opaque_max"]:
        reasons.append("too_dense")
    if features["edge_density"][idx] < thresholds["edge_min"]:
        reasons.append("too_smooth")
    if features["edge_density"][idx] > thresholds["edge_max"]:
        reasons.append("too_noisy")
    if features["bbox_w"][idx] < thresholds["bbox_w_min"]:
        reasons.append("bbox_too_narrow")
    if features["bbox_h"][idx] < thresholds["bbox_h_min"]:
        reasons.append("bbox_too_short")
    if features["bbox_area"][idx] > thresholds["bbox_area_max"]:
        reasons.append("bbox_too_large")
    if features["center_x"][idx] < thresholds["center_x_min"] or features["center_x"][idx] > thresholds["center_x_max"]:
        reasons.append("off_center_x")
    if features["center_y"][idx] < thresholds["center_y_min"] or features["center_y"][idx] > thresholds["center_y_max"]:
        reasons.append("off_center_y")
    if features["largest_component_ratio"][idx] < thresholds["largest_component_ratio_min"]:
        reasons.append("fragmented")
    if features["component_count"][idx] > thresholds["component_count_max"]:
        reasons.append("too_many_components")
    if features["unique_opaque_tokens"][idx] < thresholds["unique_opaque_tokens_min"]:
        reasons.append("too_few_colors")
    return reasons


def feature_stats(features: dict[str, np.ndarray]) -> dict:
    stats = {}
    for name, values in features.items():
        stats[name] = {
            "min": float(np.min(values)),
            "p01": float(np.quantile(values, 0.01)),
            "p05": float(np.quantile(values, 0.05)),
            "median": float(np.quantile(values, 0.50)),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
    return stats


def render_grid(index_maps: np.ndarray, indices: np.ndarray, palette: PaletteExtractor, path: Path, columns: int = 8) -> None:
    if len(indices) == 0:
        return
    images = np.stack([palette.render_index_map(index_maps[int(idx)]) for idx in indices], axis=0)
    save_rgba_grid(images, path, columns=columns)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: dict) -> None:
    lines = [
        "# Generated Set Inspection",
        "",
        "This report conservatively filters obvious failures and flags unusual samples for human review.",
        "",
        "## Counts",
        "",
        f"- Total: `{summary['counts']['total']}`",
        f"- Keep: `{summary['counts']['keep']}`",
        f"- Review: `{summary['counts']['review']}`",
        f"- Reject: `{summary['counts']['reject']}`",
        "",
        "## Key Metrics",
        "",
        f"- Opaque ratio mean: `{summary['feature_stats']['opaque_ratio']['mean']:.4f}`",
        f"- Edge density mean: `{summary['feature_stats']['edge_density']['mean']:.4f}`",
        f"- Largest component ratio mean: `{summary['feature_stats']['largest_component_ratio']['mean']:.4f}`",
        f"- Component count p95: `{summary['feature_stats']['component_count']['p95']:.1f}`",
        "",
        "## Contact Sheets",
        "",
        "Keep/random sample:",
        "",
        "![keep_random](grids/keep_random.png)",
        "",
        "Highest-score review samples:",
        "",
        "![review_highest_score](grids/review_highest_score.png)",
        "",
    ]
    if summary["counts"]["reject"] > 0:
        lines.extend(["Rejected samples:", "", "![rejects](grids/rejects.png)", ""])
    path.write_text("\n".join(lines))


def write_keep_package(
    output_dir: Path,
    index_maps: np.ndarray,
    tokens: np.ndarray | None,
    keep_indices: np.ndarray,
    summary: dict,
) -> None:
    package_dir = output_dir / "keep_package"
    package_dir.mkdir(parents=True, exist_ok=True)
    np.save(package_dir / "keep_index_maps.npy", np.asarray(index_maps[keep_indices], dtype=np.uint8))
    if tokens is not None:
        np.save(package_dir / "keep_tokens.npy", np.asarray(tokens[keep_indices], dtype=np.uint8))

    keep_manifest = {
        "kind": "pixelvar_option_a_filtered_keep_set",
        "source_generated_dir": summary["generated_dir"],
        "num_samples": int(len(keep_indices)),
        "filter_counts": summary["counts"],
        "files": {
            "keep_indices": "../keep_indices.npy",
            "keep_index_maps": "keep_index_maps.npy",
            "keep_tokens": "keep_tokens.npy" if tokens is not None else None,
        },
    }
    (package_dir / "keep_manifest.json").write_text(json.dumps(keep_manifest, indent=2) + "\n")

    zip_path = output_dir / "keep_package.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(package_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(package_dir))
    shutil.rmtree(package_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and filter a generated PixelVAR sample set")
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--palette", type=Path, default=Path("data/processed/sprites/palette.json"))
    parser.add_argument("--reference-summary", type=Path)
    parser.add_argument("--review-fraction", type=float, default=0.05)
    parser.add_argument("--grid-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write-keep-package", action="store_true")
    args = parser.parse_args()

    index_maps_path = args.generated_dir / "index_maps.npy"
    if not index_maps_path.exists():
        raise FileNotFoundError(index_maps_path)
    index_maps = np.load(index_maps_path, mmap_mode="r")
    manifest = load_json(args.generated_dir / "manifest.json")
    reference = load_json(args.reference_summary)
    vocab_size = 17
    tokens = None
    if manifest.get("files", {}).get("tokens"):
        tokens = np.load(args.generated_dir / manifest["files"]["tokens"], mmap_mode="r")
        vocab_size = int(tokens.max()) + 1
    vocab_size = max(vocab_size, 17)

    features = compute_features(index_maps, vocab_size=vocab_size)
    ref_opaque = reference.get("opaque_ratio")
    ref_edge = reference.get("edge_density")

    score = (
        robust_z(features["opaque_ratio"], ref_opaque)
        + robust_z(features["edge_density"], ref_edge)
        + 0.75 * robust_z(features["bbox_area"])
        + 0.50 * robust_z(features["center_x"])
        + 0.50 * robust_z(features["center_y"])
        + 2.00 * np.maximum(0.0, 1.0 - features["largest_component_ratio"]) * 100.0
        + 0.25 * np.maximum(0.0, features["component_count"] - 1.0)
    )

    thresholds = {
        "opaque_min": 0.16,
        "opaque_max": 0.36,
        "edge_min": 0.11,
        "edge_max": 0.31,
        "bbox_w_min": 0.25,
        "bbox_h_min": 0.65,
        "bbox_area_max": 0.75,
        "center_x_min": 0.38,
        "center_x_max": 0.62,
        "center_y_min": 0.45,
        "center_y_max": 0.70,
        "largest_component_ratio_min": 0.90,
        "component_count_max": 8,
        "unique_opaque_tokens_min": 3,
    }

    rows = []
    reject_indices = []
    for idx in range(len(index_maps)):
        reasons = reasons_for_sample(features, idx, thresholds)
        if reasons:
            status = "reject"
            reject_indices.append(idx)
        else:
            status = "keep"
        rows.append(
            {
                "index": idx,
                "status": status,
                "score": f"{float(score[idx]):.8f}",
                "reasons": "|".join(reasons),
                **{name: f"{float(values[idx]):.8f}" for name, values in features.items()},
            }
        )

    reject_set = set(reject_indices)
    eligible = np.array([idx for idx in range(len(index_maps)) if idx not in reject_set], dtype=np.int64)
    review_count = min(len(eligible), int(round(len(index_maps) * args.review_fraction)))
    review_indices = np.array([], dtype=np.int64)
    if review_count:
        ranked = eligible[np.argsort(score[eligible])[::-1]]
        review_indices = np.sort(ranked[:review_count])
        for idx in review_indices:
            rows[int(idx)]["status"] = "review"

    keep_indices = np.array([idx for idx in eligible if idx not in set(review_indices.tolist())], dtype=np.int64)
    reject_indices = np.array(reject_indices, dtype=np.int64)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    grids_dir = args.output_dir / "grids"
    grids_dir.mkdir(exist_ok=True)
    np.save(args.output_dir / "keep_indices.npy", keep_indices.astype(np.int32))
    np.save(args.output_dir / "review_indices.npy", review_indices.astype(np.int32))
    np.save(args.output_dir / "reject_indices.npy", reject_indices.astype(np.int32))
    write_csv(args.output_dir / "quality_scores.csv", rows)

    rng = np.random.default_rng(args.seed)
    keep_for_grid = keep_indices
    if len(keep_for_grid) > args.grid_samples:
        keep_for_grid = np.sort(rng.choice(keep_for_grid, size=args.grid_samples, replace=False))
    review_for_grid = review_indices[np.argsort(score[review_indices])[::-1]][: args.grid_samples]

    palette = PaletteExtractor()
    palette.load(args.palette)
    render_grid(index_maps, keep_for_grid, palette, grids_dir / "keep_random.png")
    render_grid(index_maps, review_for_grid, palette, grids_dir / "review_highest_score.png")
    render_grid(index_maps, reject_indices[: args.grid_samples], palette, grids_dir / "rejects.png")

    summary = {
        "generated_dir": str(args.generated_dir),
        "reference_summary": str(args.reference_summary) if args.reference_summary else None,
        "review_fraction": args.review_fraction,
        "counts": {
            "total": int(len(index_maps)),
            "keep": int(len(keep_indices)),
            "review": int(len(review_indices)),
            "reject": int(len(reject_indices)),
        },
        "thresholds": thresholds,
        "feature_stats": feature_stats(features),
        "score_stats": {
            "min": float(score.min()),
            "median": float(np.median(score)),
            "p95": float(np.quantile(score, 0.95)),
            "p99": float(np.quantile(score, 0.99)),
            "max": float(score.max()),
        },
    }
    (args.output_dir / "inspection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(args.output_dir / "inspection_report.md", summary)
    if args.write_keep_package:
        write_keep_package(args.output_dir, index_maps, tokens, keep_indices, summary)
    print(json.dumps(summary["counts"], indent=2))
    print(f"wrote inspection to {args.output_dir}")


if __name__ == "__main__":
    main()
