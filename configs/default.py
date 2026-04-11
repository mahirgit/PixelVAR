"""Default configuration for PixelVAR."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    target_size: int = 32
    palette_size: int = 16
    num_scales: int = 6  # 1x1, 2x2, 4x4, 8x8, 16x16, 32x32
    scale_resolutions: List[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32]
    )


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    seed: int = 42
