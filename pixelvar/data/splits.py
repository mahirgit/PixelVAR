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
    unique_ids = sorted({str(i) for i in ids if i is not None}, key=lambda x: int(x) if x.isdigit() else x)
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


def assert_no_split_leakage(sample_records: list[dict]) -> None:
    """Raise if one Pokemon ID appears in multiple splits."""
    by_id: dict[str, set[str]] = {}
    for record in sample_records:
        pokemon_id = record.get("pokemon_id")
        split = record.get("split")
        if pokemon_id is None or split is None:
            continue
        by_id.setdefault(str(pokemon_id), set()).add(str(split))

    leaked = {pokemon_id: splits for pokemon_id, splits in by_id.items() if len(splits) > 1}
    if leaked:
        preview = ", ".join(f"{pid}:{sorted(splits)}" for pid, splits in list(leaked.items())[:5])
        raise ValueError(f"Pokemon split leakage detected: {preview}")
