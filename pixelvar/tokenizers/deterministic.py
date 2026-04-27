"""Deterministic palette-index pyramid tokenizer."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from pixelvar.tokenizers.base import BaseTokenizer


DEFAULT_SCALE_RESOLUTIONS = [1, 2, 4, 8, 16, 32]


@dataclass(frozen=True)
class DeterministicPyramidTokenizer(BaseTokenizer):
    """
    Mode-pooling tokenizer over literal palette-index maps.

    Token values are not learned codes. Token 0 is transparent, and tokens
    1..K are palette colors. Ties in mode pooling are resolved by the smallest
    token value because ``torch.unique(..., sorted=True)`` returns sorted values.
    """

    scale_resolutions: list[int] = field(default_factory=lambda: DEFAULT_SCALE_RESOLUTIONS.copy())

    def __post_init__(self) -> None:
        if not self.scale_resolutions:
            raise ValueError("scale_resolutions must be non-empty")
        if self.scale_resolutions[-1] != 32:
            raise ValueError("The final deterministic tokenizer scale must be 32")
        if any(s <= 0 for s in self.scale_resolutions):
            raise ValueError("All scale resolutions must be positive")

    @property
    def token_counts(self) -> list[int]:
        return [res * res for res in self.scale_resolutions]

    @property
    def offsets(self) -> list[int]:
        offsets = [0]
        for count in self.token_counts[:-1]:
            offsets.append(offsets[-1] + count)
        return offsets

    @property
    def boundaries(self) -> list[tuple[int, int]]:
        return [(start, start + count) for start, count in zip(self.offsets, self.token_counts)]

    @property
    def sequence_length(self) -> int:
        return sum(self.token_counts)

    def encode(self, index_map: torch.Tensor | np.ndarray) -> list[torch.Tensor]:
        """Create mode-pooled scale maps from ``(32, 32)`` or ``(B, 32, 32)``."""
        tensor = torch.as_tensor(index_map, dtype=torch.long)
        squeeze = False
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
            squeeze = True
        if tensor.ndim != 3:
            raise ValueError(f"index_map must have shape (32, 32) or (B, 32, 32), got {tuple(tensor.shape)}")

        batch, height, width = tensor.shape
        if height != width:
            raise ValueError(f"index_map must be square, got {height}x{width}")
        if self.scale_resolutions[-1] != height:
            raise ValueError(
                f"final scale {self.scale_resolutions[-1]} does not match input size {height}"
            )

        maps: list[torch.Tensor] = []
        for res in self.scale_resolutions:
            if height % res != 0 or width % res != 0:
                raise ValueError(f"scale {res} does not evenly divide input size {height}x{width}")
            if res == height:
                scale_map = tensor.clone()
            else:
                scale_map = self._mode_pool(tensor, res)
            maps.append(scale_map.squeeze(0) if squeeze else scale_map)
        return maps

    def decode(self, multi_scale_maps: list[torch.Tensor]) -> torch.Tensor:
        """Return the final-resolution index map from the pyramid."""
        if len(multi_scale_maps) != len(self.scale_resolutions):
            raise ValueError(f"expected {len(self.scale_resolutions)} scale maps, got {len(multi_scale_maps)}")
        return torch.as_tensor(multi_scale_maps[-1], dtype=torch.long).clone()

    def to_sequence(self, multi_scale_maps: list[torch.Tensor]) -> torch.Tensor:
        """Flatten maps in coarse-to-fine order."""
        if len(multi_scale_maps) != len(self.scale_resolutions):
            raise ValueError(f"expected {len(self.scale_resolutions)} scale maps, got {len(multi_scale_maps)}")

        first = torch.as_tensor(multi_scale_maps[0])
        if first.ndim == 2:
            parts = [torch.as_tensor(m, dtype=torch.long).reshape(-1) for m in multi_scale_maps]
            sequence = torch.cat(parts, dim=0)
        elif first.ndim == 3:
            batch = first.shape[0]
            parts = [torch.as_tensor(m, dtype=torch.long).reshape(batch, -1) for m in multi_scale_maps]
            sequence = torch.cat(parts, dim=1)
        else:
            raise ValueError(f"scale maps must have rank 2 or 3, got {first.ndim}")

        if sequence.shape[-1] != self.sequence_length:
            raise ValueError(f"token sequence length {sequence.shape[-1]} != {self.sequence_length}")
        return sequence

    def from_sequence(self, token_sequence: torch.Tensor | np.ndarray) -> list[torch.Tensor]:
        """Split a flat sequence into scale maps."""
        sequence = torch.as_tensor(token_sequence, dtype=torch.long)
        squeeze = False
        if sequence.ndim == 1:
            sequence = sequence.unsqueeze(0)
            squeeze = True
        if sequence.ndim != 2:
            raise ValueError(f"token_sequence must have shape (1365,) or (B, 1365), got {tuple(sequence.shape)}")
        if sequence.shape[1] != self.sequence_length:
            raise ValueError(f"token sequence length {sequence.shape[1]} != {self.sequence_length}")

        maps: list[torch.Tensor] = []
        for (start, end), res in zip(self.boundaries, self.scale_resolutions):
            scale_map = sequence[:, start:end].reshape(sequence.shape[0], res, res)
            maps.append(scale_map.squeeze(0) if squeeze else scale_map)
        return maps

    @staticmethod
    def _mode_pool(index_maps: torch.Tensor, res: int) -> torch.Tensor:
        batch, height, width = index_maps.shape
        block_h = height // res
        block_w = width // res
        pooled = torch.empty((batch, res, res), dtype=torch.long, device=index_maps.device)

        for i in range(res):
            for j in range(res):
                block = index_maps[:, i * block_h : (i + 1) * block_h, j * block_w : (j + 1) * block_w]
                flat = block.reshape(batch, -1)
                for b in range(batch):
                    values, counts = torch.unique(flat[b], sorted=True, return_counts=True)
                    pooled[b, i, j] = values[torch.argmax(counts)]
        return pooled
