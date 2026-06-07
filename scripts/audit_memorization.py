#!/usr/bin/env python3
"""Audit generated sprites for duplicates, split leakage, and memorization."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixelvar.data.palette import PaletteExtractor


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def image_paths(directory: Path) -> list[Path]:
    paths = [path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS]
    if not paths:
        raise ValueError(f"No image files found in {directory}")
    return sorted(paths)


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def split_indices(manifest: dict) -> dict[str, list[int]]:
    splits: dict[str, list[int]] = defaultdict(list)
    for idx, sample in enumerate(manifest.get("samples", [])):
        split = str(sample.get("split", "unsplit"))
        splits[split].append(int(sample.get("index", idx)))
    if not splits:
        raise ValueError("Manifest does not contain samples")
    return dict(splits)


def hash_map(index_map: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(index_map, dtype=np.uint8).tobytes()).hexdigest()


def maps_to_hashes(index_maps: np.ndarray) -> list[str]:
    return [hash_map(frame) for frame in index_maps]


def rgba_to_index_map(image: np.ndarray, palette: PaletteExtractor) -> tuple[np.ndarray, int]:
    rgba = np.asarray(image, dtype=np.uint8)
    if rgba.shape != (32, 32, 4):
        raise ValueError(f"Expected 32x32 RGBA image, got {rgba.shape}")

    color_to_token = {tuple(map(int, color)): idx + 1 for idx, color in enumerate(palette.palette)}
    result = np.zeros((32, 32), dtype=np.uint8)
    opaque = rgba[:, :, 3] >= palette.alpha_threshold
    invalid = 0
    if opaque.any():
        rgb = rgba[:, :, :3]
        for y, x in np.argwhere(opaque):
            color = tuple(map(int, rgb[y, x]))
            token = color_to_token.get(color)
            if token is None:
                invalid += 1
                distances = np.sum((palette.palette.astype(np.int16) - np.asarray(color, dtype=np.int16)) ** 2, axis=1)
                token = int(np.argmin(distances)) + 1
            result[y, x] = token
    return result, invalid


def load_generated_maps(generated_dir: Path, palette: PaletteExtractor, max_images: int | None) -> tuple[list[Path], np.ndarray, int]:
    paths = image_paths(generated_dir)
    if max_images is not None and max_images > 0:
        paths = paths[:max_images]
    maps = []
    invalid_pixels = 0
    for path in paths:
        image = Image.open(path).convert("RGBA")
        if image.size != (32, 32):
            image = image.resize((32, 32), Image.Resampling.NEAREST)
        index_map, invalid = rgba_to_index_map(np.asarray(image), palette)
        maps.append(index_map)
        invalid_pixels += invalid
    return paths, np.stack(maps, axis=0), invalid_pixels


def duplicate_summary(hashes: list[str]) -> dict:
    counts = Counter(hashes)
    duplicate_groups = [count for count in counts.values() if count > 1]
    return {
        "count": len(hashes),
        "unique": len(counts),
        "duplicate_groups": len(duplicate_groups),
        "duplicate_samples": int(sum(count - 1 for count in duplicate_groups)),
        "max_group_size": int(max(duplicate_groups) if duplicate_groups else 1),
    }


def dataset_duplicate_summary(index_maps: np.ndarray, splits: dict[str, list[int]]) -> dict:
    result = {}
    hash_to_splits: dict[str, set[str]] = defaultdict(set)
    hash_to_split_counts: dict[str, Counter] = defaultdict(Counter)
    for split, indices in splits.items():
        split_hashes = maps_to_hashes(index_maps[indices])
        result[split] = duplicate_summary(split_hashes)
        for item_hash in split_hashes:
            hash_to_splits[item_hash].add(split)
            hash_to_split_counts[item_hash][split] += 1

    cross_split = {item_hash: dict(hash_to_split_counts[item_hash]) for item_hash, names in hash_to_splits.items() if len(names) > 1}
    result["cross_split_duplicate_hashes"] = len(cross_split)
    result["cross_split_duplicate_samples"] = int(sum(sum(counts.values()) for counts in cross_split.values()))
    return result


def exact_match_summary(generated_hashes: list[str], split_hashes: dict[str, dict[str, list[int]]]) -> tuple[dict, list[dict]]:
    rows = []
    summary = {}
    for split, lookup in split_hashes.items():
        matched_generated = []
        matched_hashes = set()
        for gen_idx, item_hash in enumerate(generated_hashes):
            if item_hash in lookup:
                matched_generated.append(gen_idx)
                matched_hashes.add(item_hash)
                rows.append(
                    {
                        "generated_index": gen_idx,
                        "hash": item_hash,
                        "matched_split": split,
                        "matched_dataset_indices": " ".join(str(idx) for idx in lookup[item_hash][:20]),
                        "num_dataset_matches": len(lookup[item_hash]),
                    }
                )
        summary[split] = {
            "generated_matches": len(matched_generated),
            "generated_match_rate": len(matched_generated) / len(generated_hashes),
            "unique_matching_hashes": len(matched_hashes),
        }
    return summary, rows


def nearest_by_split(
    generated_maps: np.ndarray,
    index_maps: np.ndarray,
    splits: dict[str, list[int]],
    device: torch.device,
    gen_batch_size: int,
    ref_batch_size: int,
) -> tuple[dict, list[dict]]:
    generated = torch.from_numpy(generated_maps.reshape(len(generated_maps), -1).astype(np.uint8)).to(device)
    result = {}
    nearest_rows = []
    for split, indices in splits.items():
        reference_np = np.asarray(index_maps[indices]).reshape(len(indices), -1).astype(np.uint8)
        best_dist = torch.full((len(generated),), generated.shape[1] + 1, dtype=torch.int32, device=device)
        best_ref_local = torch.full((len(generated),), -1, dtype=torch.long, device=device)

        for ref_start in range(0, len(reference_np), ref_batch_size):
            ref_chunk_np = reference_np[ref_start : ref_start + ref_batch_size]
            reference = torch.from_numpy(ref_chunk_np).to(device)
            for gen_start in range(0, len(generated), gen_batch_size):
                gen_chunk = generated[gen_start : gen_start + gen_batch_size]
                distances = (gen_chunk[:, None, :] != reference[None, :, :]).sum(dim=2, dtype=torch.int32)
                chunk_min, chunk_argmin = distances.min(dim=1)
                update = chunk_min < best_dist[gen_start : gen_start + len(gen_chunk)]
                if update.any():
                    target_slice = slice(gen_start, gen_start + len(gen_chunk))
                    best_dist[target_slice] = torch.where(update, chunk_min, best_dist[target_slice])
                    candidate_indices = chunk_argmin.to(torch.long) + ref_start
                    best_ref_local[target_slice] = torch.where(update, candidate_indices, best_ref_local[target_slice])

        best_dist_cpu = best_dist.cpu().numpy()
        best_ref_local_cpu = best_ref_local.cpu().numpy()
        nearest_fraction = best_dist_cpu / generated.shape[1]
        result[split] = {
            "mean_hamming_fraction": float(nearest_fraction.mean()),
            "median_hamming_fraction": float(np.median(nearest_fraction)),
            "p01_hamming_fraction": float(np.percentile(nearest_fraction, 1)),
            "p05_hamming_fraction": float(np.percentile(nearest_fraction, 5)),
            "min_hamming_fraction": float(nearest_fraction.min()),
            "exact_count": int((best_dist_cpu == 0).sum()),
            "within_1pct": int((nearest_fraction <= 0.01).sum()),
            "within_2pct": int((nearest_fraction <= 0.02).sum()),
            "within_5pct": int((nearest_fraction <= 0.05).sum()),
            "within_10pct": int((nearest_fraction <= 0.10).sum()),
        }
        for gen_idx, (dist, local_idx) in enumerate(zip(best_dist_cpu, best_ref_local_cpu)):
            nearest_rows.append(
                {
                    "generated_index": gen_idx,
                    "split": split,
                    "nearest_dataset_index": int(indices[int(local_idx)]),
                    "hamming_distance": int(dist),
                    "hamming_fraction": f"{float(dist / generated.shape[1]):.8f}",
                }
            )
    return result, nearest_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_pair_sheet(
    path: Path,
    generated_paths: list[Path],
    index_maps: np.ndarray,
    palette: PaletteExtractor,
    nearest_rows: list[dict],
    max_pairs: int,
) -> None:
    if not nearest_rows:
        return
    sorted_rows = sorted(nearest_rows, key=lambda row: (int(row["hamming_distance"]), row["split"], int(row["generated_index"])))
    selected = sorted_rows[:max_pairs]
    scale = 4
    tile = 32
    columns = 4
    rows = math.ceil(len(selected) / columns)
    canvas = Image.new("RGBA", (columns * tile * 2 * scale, rows * tile * scale), (0, 0, 0, 0))
    for out_idx, row in enumerate(selected):
        gen_idx = int(row["generated_index"])
        ref_idx = int(row["nearest_dataset_index"])
        gen_img = Image.open(generated_paths[gen_idx]).convert("RGBA").resize((tile * scale, tile * scale), Image.Resampling.NEAREST)
        ref_arr = palette.render_index_map(index_maps[ref_idx])
        ref_img = Image.fromarray(ref_arr, mode="RGBA").resize((tile * scale, tile * scale), Image.Resampling.NEAREST)
        grid_row, grid_col = divmod(out_idx, columns)
        x0 = grid_col * tile * 2 * scale
        y0 = grid_row * tile * scale
        canvas.alpha_composite(gen_img, (x0, y0))
        canvas.alpha_composite(ref_img, (x0 + tile * scale, y0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def write_report(path: Path, summary: dict) -> None:
    lines = [
        "# Memorization Audit",
        "",
        f"Generated folder: `{summary['generated_dir']}`",
        f"Processed dataset: `{summary['processed_dir']}`",
        f"Generated images: `{summary['num_generated']}`",
        f"Off-palette opaque pixels remapped: `{summary['invalid_generated_pixels']}`",
        "",
        "## Exact Matches",
        "",
        "| Split | Generated matches | Match rate | Unique hashes |",
        "| --- | ---: | ---: | ---: |",
    ]
    for split, values in summary["exact_matches"].items():
        lines.append(
            f"| {split} | {values['generated_matches']} | {values['generated_match_rate']:.6f} | "
            f"{values['unique_matching_hashes']} |"
        )
    lines.extend(
        [
            "",
            "## Nearest Neighbors",
            "",
            "| Split | Mean Hamming | P05 Hamming | Min Hamming | Exact | <=1% | <=2% | <=5% |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split, values in summary["nearest"].items():
        lines.append(
            f"| {split} | {values['mean_hamming_fraction']:.6f} | {values['p05_hamming_fraction']:.6f} | "
            f"{values['min_hamming_fraction']:.6f} | {values['exact_count']} | {values['within_1pct']} | "
            f"{values['within_2pct']} | {values['within_5pct']} |"
        )
    lines.extend(
        [
            "",
            "## Dataset Duplicates",
            "",
            f"- Cross-split duplicate hashes: `{summary['dataset_duplicates']['cross_split_duplicate_hashes']}`",
            f"- Cross-split duplicate samples: `{summary['dataset_duplicates']['cross_split_duplicate_samples']}`",
            "",
            "| Split | Samples | Unique | Duplicate groups | Duplicate samples | Max group |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split, values in summary["dataset_duplicates"].items():
        if not isinstance(values, dict):
            continue
        lines.append(
            f"| {split} | {values['count']} | {values['unique']} | {values['duplicate_groups']} | "
            f"{values['duplicate_samples']} | {values['max_group_size']} |"
        )
    lines.extend(
        [
            "",
            "## Generated Duplicates",
            "",
            f"- Unique generated images: `{summary['generated_duplicates']['unique']}`",
            f"- Duplicate groups: `{summary['generated_duplicates']['duplicate_groups']}`",
            f"- Duplicate generated samples: `{summary['generated_duplicates']['duplicate_samples']}`",
            f"- Max generated duplicate group size: `{summary['generated_duplicates']['max_group_size']}`",
            "",
            "## Interpretation",
            "",
        ]
    )
    val_matches = summary["exact_matches"].get("val", {}).get("generated_matches", 0)
    train_matches = summary["exact_matches"].get("train", {}).get("generated_matches", 0)
    cross_split = summary["dataset_duplicates"]["cross_split_duplicate_hashes"]
    if val_matches > 0 and train_matches > 0 and cross_split > 0:
        lines.append(
            "The generated exact matches overlap multiple dataset splits, and the processed dataset contains cross-split exact duplicates. "
            "This means some validation exact matches can be explained by duplicate images already present across splits."
        )
    elif val_matches > 0:
        lines.append(
            "There are exact validation matches. Review the matching images before claiming this model is fully non-memorizing."
        )
    else:
        lines.append("No exact validation matches were found.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit generated sprites for memorization")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/sprites"))
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/memorization_audit"))
    parser.add_argument("--max-generated", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gen-batch-size", type=int, default=128)
    parser.add_argument("--ref-batch-size", type=int, default=4096)
    parser.add_argument("--nearest-sheet-pairs", type=int, default=32)
    args = parser.parse_args()

    palette = PaletteExtractor()
    palette.load(args.processed_dir / "palette.json")
    index_maps = np.load(args.processed_dir / "index_maps.npy", mmap_mode="r")
    manifest = load_manifest(args.processed_dir / "manifest.json")
    splits = split_indices(manifest)
    generated_paths, generated_maps, invalid_pixels = load_generated_maps(
        args.generated_dir,
        palette,
        max_images=args.max_generated if args.max_generated > 0 else None,
    )

    generated_hashes = maps_to_hashes(generated_maps)
    dataset_hash_lookup: dict[str, dict[str, list[int]]] = {}
    for split, indices in splits.items():
        lookup: dict[str, list[int]] = defaultdict(list)
        for dataset_idx, item_hash in zip(indices, maps_to_hashes(index_maps[indices])):
            lookup[item_hash].append(int(dataset_idx))
        dataset_hash_lookup[split] = dict(lookup)

    exact_summary, exact_rows = exact_match_summary(generated_hashes, dataset_hash_lookup)
    nearest_summary, nearest_rows = nearest_by_split(
        generated_maps=generated_maps,
        index_maps=index_maps,
        splits=splits,
        device=torch.device(args.device),
        gen_batch_size=args.gen_batch_size,
        ref_batch_size=args.ref_batch_size,
    )

    summary = {
        "processed_dir": args.processed_dir.as_posix(),
        "generated_dir": args.generated_dir.as_posix(),
        "num_generated": len(generated_maps),
        "invalid_generated_pixels": invalid_pixels,
        "generated_duplicates": duplicate_summary(generated_hashes),
        "dataset_duplicates": dataset_duplicate_summary(index_maps, splits),
        "exact_matches": exact_summary,
        "nearest": nearest_summary,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(args.output_dir / "exact_matches.csv", exact_rows)
    write_csv(args.output_dir / "nearest_neighbors.csv", nearest_rows)
    write_report(args.output_dir / "audit_report.md", summary)
    render_pair_sheet(
        args.output_dir / "nearest_pairs.png",
        generated_paths,
        index_maps,
        palette,
        nearest_rows,
        args.nearest_sheet_pairs,
    )
    print(f"Wrote memorization audit to {args.output_dir}")


if __name__ == "__main__":
    main()
