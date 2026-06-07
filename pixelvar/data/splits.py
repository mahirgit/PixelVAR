"""Dataset split helpers."""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Iterable


POKEMON_ID_RE = re.compile(r"^(\d+)")


def parse_pokemon_id(path: str | Path) -> str | None:
    """Extract a Pokemon ID from filenames like ``25.png`` or ``25_back.png``."""
    stem = Path(path).stem
    match = POKEMON_ID_RE.match(stem)
    return match.group(1) if match else None


def infer_pokemon_variant(path: str | Path) -> str:
    """Infer a loose Pokemon sprite variant from path components."""
    p = Path(path)
    text = "/".join(p.parts).lower()
    stem = p.stem.lower()
    pieces = []
    if "shiny" in text:
        pieces.append("shiny")
    if "back" in text or stem.endswith("_back"):
        pieces.append("back")
    return "_".join(pieces) if pieces else "front"


def make_id_splits(
    ids: Iterable[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, str]:
    """Create deterministic train/val/test assignments for unique IDs."""
    return make_group_splits(ids, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)


def make_group_splits(
    group_ids: Iterable[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, str]:
    """Create deterministic train/val/test assignments for arbitrary asset groups."""
    unique_ids = sorted({str(i) for i in group_ids if i is not None}, key=_stable_group_sort_key)
    rng = random.Random(seed)
    rng.shuffle(unique_ids)

    n_total = len(unique_ids)
    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * val_ratio))
    if n_train + n_val > n_total:
        n_val = max(0, n_total - n_train)

    split_map: dict[str, str] = {}
    for idx, pokemon_id in enumerate(unique_ids):
        if idx < n_train:
            split_map[pokemon_id] = "train"
        elif idx < n_train + n_val:
            split_map[pokemon_id] = "val"
        else:
            split_map[pokemon_id] = "test"
    return split_map


def assert_no_group_split_leakage(
    sample_records: list[dict],
    group_key: str = "group_id",
    split_key: str = "split",
) -> None:
    """Raise if one asset group appears in multiple splits."""
    by_group: dict[str, set[str]] = {}
    for record in sample_records:
        group_id = record.get(group_key)
        split = record.get(split_key)
        if group_id is None or split is None:
            continue
        by_group.setdefault(str(group_id), set()).add(str(split))

    leaked = {group_id: splits for group_id, splits in by_group.items() if len(splits) > 1}
    if leaked:
        preview = ", ".join(f"{gid}:{sorted(splits)}" for gid, splits in list(leaked.items())[:5])
        raise ValueError(f"Split leakage detected for {group_key}: {preview}")


def assert_no_split_leakage(sample_records: list[dict]) -> None:
    """Raise if one Pokemon ID appears in multiple splits."""
    try:
        assert_no_group_split_leakage(sample_records, group_key="pokemon_id")
    except ValueError as exc:
        raise ValueError(str(exc).replace("Split leakage detected for pokemon_id", "Pokemon split leakage detected")) from exc


def _stable_group_sort_key(group_id: str) -> tuple[int, int | str]:
    text = str(group_id)
    return (0, int(text)) if text.isdigit() else (1, text)
