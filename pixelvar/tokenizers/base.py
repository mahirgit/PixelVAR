"""
Base tokenizer interface for PixelVAR.

All tokenizers (deterministic, VQ-VAE, etc.) must implement this interface.
Downstream code (AR model, loss, sampling) only interacts through this contract.

See docs/design_contract.md for tensor shapes and vocabulary layout.
"""

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class BaseTokenizer(ABC):
    """
    Abstract base class for PixelVAR tokenizers.

    Contract:
        - encode: (B, 32, 32) -> list of K tensors, each (B, h_k, w_k)
        - decode: list of K tensors -> (B, 32, 32)
        - to_sequence: list of K tensors -> (B, 1365)
        - from_sequence: (B, 1365) -> list of K tensors

    Vocabulary (v0):
        Token 0:     transparent
        Token 1-16:  palette colors
        Token 17:    [MASK] (reserved, unused in v0)
    """

    TRANSPARENT_TOKEN: int = 0
    MASK_TOKEN: int = -1  # Set properly by subclass based on palette_size

    def __init__(self, scale_resolutions: list[int] = None, palette_size: int = 16):
        if scale_resolutions is None:
            self.scale_resolutions = [1, 2, 4, 8, 16, 32]
        else:
            self.scale_resolutions = scale_resolutions

        self.palette_size = palette_size
        self.MASK_TOKEN = palette_size + 1  # e.g., 17 for palette_size=16
        self.vocab_size = palette_size + 1  # palette colors + transparent (MASK not counted in v0)

        # Precompute total token count
        self.total_tokens = sum(r * r for r in self.scale_resolutions)
        # scale_lengths[i] = number of tokens at scale i
        self.scale_lengths = [r * r for r in self.scale_resolutions]
        # scale_offsets[i] = starting index of scale i in the flat sequence
        self.scale_offsets = []
        offset = 0
        for length in self.scale_lengths:
            self.scale_offsets.append(offset)
            offset += length

    @abstractmethod
    def encode(self, index_map: Tensor) -> list[Tensor]:
        """
        Encode a full-resolution index map into multi-scale token maps.

        Args:
            index_map: (B, 32, 32) long tensor, values in [0, vocab_size)

        Returns:
            List of K tensors: [(B, 1, 1), (B, 2, 2), (B, 4, 4),
                                (B, 8, 8), (B, 16, 16), (B, 32, 32)]
        """
        ...

    @abstractmethod
    def decode(self, multi_scale_maps: list[Tensor]) -> Tensor:
        """
        Decode multi-scale token maps back to full resolution.

        Args:
            multi_scale_maps: List of K tensors, each (B, h_k, w_k)

        Returns:
            (B, 32, 32) long tensor — reconstructed index map
            (For deterministic tokenizer, the finest scale IS the reconstruction.)
        """
        ...

    def to_sequence(self, multi_scale_maps: list[Tensor]) -> Tensor:
        """
        Flatten multi-scale maps into a single token sequence.

        Args:
            multi_scale_maps: List of K tensors, each (B, h_k, w_k)

        Returns:
            (B, total_tokens) long tensor — concatenated flat sequence
        """
        return torch.cat([m.reshape(m.shape[0], -1) for m in multi_scale_maps], dim=1)

    def from_sequence(self, token_sequence: Tensor) -> list[Tensor]:
        """
        Unflatten a token sequence back into multi-scale maps.

        Args:
            token_sequence: (B, total_tokens) long tensor

        Returns:
            List of K tensors: [(B, 1, 1), (B, 2, 2), ..., (B, 32, 32)]
        """
        B = token_sequence.shape[0]
        maps = []
        for res, offset, length in zip(
            self.scale_resolutions, self.scale_offsets, self.scale_lengths
        ):
            flat = token_sequence[:, offset : offset + length]
            maps.append(flat.reshape(B, res, res))
        return maps

    def get_scale_info(self) -> list[dict]:
        """Return metadata about each scale (useful for AR model)."""
        return [
            {
                "resolution": res,
                "num_tokens": length,
                "offset": offset,
            }
            for res, length, offset in zip(
                self.scale_resolutions, self.scale_lengths, self.scale_offsets
            )
        ]
