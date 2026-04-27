"""Base tokenizer contract for discrete PixelVAR tokenizers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseTokenizer(ABC):
    """Interface shared by deterministic and future learned tokenizers."""

    @abstractmethod
    def encode(self, index_map: torch.Tensor) -> list[torch.Tensor]:
        """Convert an index map into multi-scale maps."""

    @abstractmethod
    def decode(self, multi_scale_maps: list[torch.Tensor]) -> torch.Tensor:
        """Convert multi-scale maps back to the final-resolution index map."""

    @abstractmethod
    def to_sequence(self, multi_scale_maps: list[torch.Tensor]) -> torch.Tensor:
        """Flatten multi-scale maps into a single token sequence."""

    @abstractmethod
    def from_sequence(self, token_sequence: torch.Tensor) -> list[torch.Tensor]:
        """Split a flat token sequence into multi-scale maps."""
