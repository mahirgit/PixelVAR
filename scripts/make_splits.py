#!/usr/bin/env python3
"""Create deterministic Pokemon ID splits for an existing raw or processed dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pixelvar.data.splits import make_id_splits, parse_pokemon_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Pokemon ID train/val/test splits")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/pokemon"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/pokemon/splits.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = sorted(args.raw_dir.rglob("*.png"))
    pokemon_ids = [parse_pokemon_id(path) for path in paths]
    pokemon_ids = [pid for pid in pokemon_ids if pid is not None]
    if not pokemon_ids:
        raise SystemExit(f"No Pokemon IDs found under {args.raw_dir}")

    split_map = make_id_splits(pokemon_ids, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(split_map, indent=2, sort_keys=True))

    counts = {split: list(split_map.values()).count(split) for split in ("train", "val", "test")}
    print(f"Wrote {len(split_map)} Pokemon ID splits to {args.output}")
    print(counts)


if __name__ == "__main__":
    main()
