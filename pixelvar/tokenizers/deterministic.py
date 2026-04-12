"""
Deterministic palette-index pyramid tokenizer.

Encodes 32x32 palette index maps into multi-scale token maps via mode-pooling
(most frequent index in each block). This is the v0 baseline tokenizer —
no learning required.

See docs/design_contract.md for vocabulary layout and tensor shapes.
"""

import torch
from torch import Tensor

from pixelvar.tokenizers.base import BaseTokenizer


class DeterministicTokenizer(BaseTokenizer):
    """
    Deterministic multi-scale tokenizer using mode-pooling.

    For each scale resolution r < 32, divides the 32x32 index map into
    (32/r)x(32/r) blocks and takes the most frequent palette index per block.

    This mirrors how pixel artists work: coarse silhouette first, then detail.
    """

    def __init__(self, scale_resolutions: list[int] = None, palette_size: int = 16):
        super().__init__(scale_resolutions=scale_resolutions, palette_size=palette_size)

    def encode(self, index_map: Tensor) -> list[Tensor]:
        """
        Encode full-resolution index map into multi-scale token maps.

        Args:
            index_map: (B, 32, 32) long tensor, values in [0, vocab_size)

        Returns:
            List of K tensors at increasing resolutions.
        """
        B, H, W = index_map.shape
        assert H == 32 and W == 32, f"Expected 32x32, got {H}x{W}"

        maps = []
        for res in self.scale_resolutions:
            if res == H:
                # Finest scale — identity
                maps.append(index_map.clone())
            else:
                maps.append(self._mode_pool(index_map, res))
        return maps

    def decode(self, multi_scale_maps: list[Tensor]) -> Tensor:
        """
        Decode by returning the finest scale map.

        For the deterministic tokenizer, the 32x32 scale IS the full
        reconstruction. No upsampling or learned decoding needed.
        """
        return multi_scale_maps[-1]

    def _mode_pool(self, index_map: Tensor, target_res: int) -> Tensor:
        """
        Downsample via mode-pooling (most frequent index per block).

        Args:
            index_map: (B, 32, 32) long tensor
            target_res: target spatial resolution

        Returns:
            (B, target_res, target_res) long tensor
        """
        B, H, W = index_map.shape
        block_h = H // target_res
        block_w = W // target_res

        # Reshape into blocks: (B, target_res, block_h, target_res, block_w)
        reshaped = index_map.reshape(B, target_res, block_h, target_res, block_w)
        # Merge block dims: (B, target_res, target_res, block_h * block_w)
        reshaped = reshaped.permute(0, 1, 3, 2, 4).reshape(
            B, target_res, target_res, block_h * block_w
        )

        # Mode: most frequent value in each block
        # Use one-hot bincount approach for batched mode
        result = torch.zeros(B, target_res, target_res, dtype=torch.long, device=index_map.device)

        for b in range(B):
            for i in range(target_res):
                for j in range(target_res):
                    block = reshaped[b, i, j]  # (block_h * block_w,)
                    # bincount and argmax
                    counts = torch.bincount(block, minlength=self.vocab_size)
                    result[b, i, j] = counts.argmax()

        return result

    def _mode_pool_numpy(self, index_map: Tensor, target_res: int) -> Tensor:
        """
        Faster numpy-based mode-pooling for CPU tensors.
        Used during preprocessing (not in training loop).
        """
        import numpy as np

        device = index_map.device
        arr = index_map.cpu().numpy()
        B, H, W = arr.shape
        block_h = H // target_res
        block_w = W // target_res

        result = np.zeros((B, target_res, target_res), dtype=np.int64)

        for b in range(B):
            for i in range(target_res):
                for j in range(target_res):
                    block = arr[
                        b,
                        i * block_h : (i + 1) * block_h,
                        j * block_w : (j + 1) * block_w,
                    ]
                    values, counts = np.unique(block, return_counts=True)
                    result[b, i, j] = values[np.argmax(counts)]

        return torch.from_numpy(result).long().to(device)
