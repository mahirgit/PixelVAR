"""Flat 32x32 token baselines for PixelVAR."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from pixelvar.tokenizers import DeterministicPyramidTokenizer


class CausalSelfAttention(nn.Module):
    """Causal multi-head attention with an optional KV cache for sampling."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch, length, channels = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if cache is not None:
            cached_k, cached_v = cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)

        dropout_p = self.dropout if self.training else 0.0
        is_causal = cache is None and length > 1
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=is_causal)
        y = y.transpose(1, 2).reshape(batch, length, channels)
        new_cache = (k, v) if use_cache else None
        return self.output(y), new_cache


class CausalTransformerBlock(nn.Module):
    """Pre-norm transformer block used by the flat raster AR baseline."""

    def __init__(self, d_model: int, n_heads: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attended, new_cache = self.attn(self.attn_norm(x), cache=cache, use_cache=use_cache)
        x = x + attended
        x = x + self.mlp(self.mlp_norm(x))
        return x, new_cache


class FlatARTransformer(nn.Module):
    """Raster-order autoregressive baseline over final 32x32 palette tokens."""

    def __init__(
        self,
        vocab_size: int = 17,
        scale_resolutions: list[int] | None = None,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        mlp_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.bos_token_id = vocab_size
        self.tokenizer = DeterministicPyramidTokenizer(scale_resolutions or [1, 2, 4, 8, 16, 32])
        self.scale_resolutions = self.tokenizer.scale_resolutions
        self.final_resolution = self.scale_resolutions[-1]
        self.sequence_length = self.final_resolution * self.final_resolution

        self.token_embedding = nn.Embedding(vocab_size + 1, d_model)
        self.position_embedding = nn.Embedding(self.sequence_length, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList(
            [CausalTransformerBlock(d_model=d_model, n_heads=n_heads, mlp_dim=mlp_dim, dropout=dropout) for _ in range(n_layers)]
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, final_tokens: torch.Tensor) -> torch.Tensor:
        """Predict logits for final-scale tokens with teacher forcing."""
        self._validate_final_tokens(final_tokens)
        batch, length = final_tokens.shape
        bos = torch.full((batch, 1), self.bos_token_id, dtype=torch.long, device=final_tokens.device)
        inputs = torch.cat([bos, final_tokens[:, :-1]], dim=1)
        positions = torch.arange(length, device=final_tokens.device)
        hidden = self.token_embedding(inputs) + self.position_embedding(positions).unsqueeze(0)
        hidden = self.input_norm(hidden)
        for block in self.blocks:
            hidden, _ = block(hidden)
        hidden = self.output_norm(hidden)
        return self.output_head(hidden)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        temperature: float = 1.0,
        top_k: int | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Generate final-scale tokens in raster order."""
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        was_training = self.training
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        tokens = torch.empty((batch_size, self.sequence_length), dtype=torch.long, device=device)
        previous = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=device)
        caches: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(self.blocks)
        for position_idx in range(self.sequence_length):
            position = torch.full((1,), position_idx, dtype=torch.long, device=device)
            hidden = self.token_embedding(previous) + self.position_embedding(position).unsqueeze(0)
            hidden = self.input_norm(hidden)
            next_caches = []
            for block, cache in zip(self.blocks, caches):
                hidden, next_cache = block(hidden, cache=cache, use_cache=True)
                next_caches.append(next_cache)
            caches = next_caches
            logits = self.output_head(self.output_norm(hidden[:, -1, :])) / temperature
            if top_k is not None:
                logits = self._top_k_logits(logits, top_k)
            sampled = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            tokens[:, position_idx] = sampled.squeeze(1)
            previous = sampled

        if was_training:
            self.train()
        return tokens

    def full_sequence_from_final_tokens(self, final_tokens: torch.Tensor) -> torch.Tensor:
        """Place final-scale flat tokens into a full PixelVAR pyramid sequence."""
        self._validate_final_tokens(final_tokens)
        start, end = self.tokenizer.boundaries[-1]
        sequence = torch.zeros((final_tokens.shape[0], self.tokenizer.sequence_length), dtype=torch.long, device=final_tokens.device)
        sequence[:, start:end] = final_tokens
        return sequence

    def _validate_final_tokens(self, final_tokens: torch.Tensor) -> None:
        if final_tokens.ndim != 2:
            raise ValueError(f"final_tokens must have shape (B, {self.sequence_length}), got {tuple(final_tokens.shape)}")
        if final_tokens.shape[1] != self.sequence_length:
            raise ValueError(f"final token length {final_tokens.shape[1]} != {self.sequence_length}")
        if final_tokens.min() < 0 or final_tokens.max() >= self.vocab_size:
            raise ValueError(
                f"token range [{int(final_tokens.min())}, {int(final_tokens.max())}] outside [0, {self.vocab_size - 1}]"
            )

    @staticmethod
    def _top_k_logits(logits: torch.Tensor, top_k: int) -> torch.Tensor:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if top_k >= logits.shape[-1]:
            return logits
        values, _ = torch.topk(logits, k=top_k, dim=-1)
        threshold = values[..., -1, None]
        return logits.masked_fill(logits < threshold, torch.finfo(logits.dtype).min)


class FlatMaskGITTransformer(nn.Module):
    """Flat masked-token baseline over final 32x32 palette tokens."""

    def __init__(
        self,
        vocab_size: int = 17,
        mask_token_id: int | None = None,
        scale_resolutions: list[int] | None = None,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        mlp_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token_id = vocab_size if mask_token_id is None else int(mask_token_id)
        self.tokenizer = DeterministicPyramidTokenizer(scale_resolutions or [1, 2, 4, 8, 16, 32])
        self.scale_resolutions = self.tokenizer.scale_resolutions
        self.final_resolution = self.scale_resolutions[-1]
        self.sequence_length = self.final_resolution * self.final_resolution

        input_vocab_size = max(vocab_size, self.mask_token_id + 1)
        self.token_embedding = nn.Embedding(input_vocab_size, d_model)
        self.position_embedding = nn.Embedding(self.sequence_length, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        """Predict logits for final-scale tokens from masked final-scale input."""
        self._validate_input_tokens(input_tokens)
        positions = torch.arange(self.sequence_length, device=input_tokens.device)
        hidden = self.token_embedding(input_tokens) + self.position_embedding(positions).unsqueeze(0)
        hidden = self.input_norm(hidden)
        hidden = self.transformer(hidden)
        hidden = self.output_norm(hidden)
        return self.output_head(hidden)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        refinement_steps: int = 8,
        temperature: float = 1.0,
        top_k: int | None = None,
        mask_schedule: str = "cosine",
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Generate final-scale tokens by iterative masked refinement."""
        if refinement_steps <= 0:
            raise ValueError("refinement_steps must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        was_training = self.training
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        tokens = torch.full((batch_size, self.sequence_length), self.mask_token_id, dtype=torch.long, device=device)
        kept = torch.zeros_like(tokens, dtype=torch.bool)
        for step_idx in range(refinement_steps):
            logits = self(tokens) / temperature
            if top_k is not None:
                logits = self._top_k_logits(logits, top_k)
            probs = torch.softmax(logits, dim=-1)
            sampled = torch.multinomial(probs.reshape(-1, self.vocab_size), num_samples=1).reshape(batch_size, self.sequence_length)
            confidence = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

            tokens = torch.where(kept, tokens, sampled)
            confidence = confidence.masked_fill(kept, torch.inf)
            keep_count = self._keep_count(self.sequence_length, step_idx + 1, refinement_steps, mask_schedule)
            keep_indices = confidence.topk(k=keep_count, dim=1).indices
            next_kept = torch.zeros_like(kept)
            next_kept.scatter_(1, keep_indices, True)
            kept = next_kept
            tokens = tokens.masked_fill(~kept, self.mask_token_id)

        tokens = tokens.clamp(max=self.vocab_size - 1)
        if was_training:
            self.train()
        return tokens

    def full_sequence_from_final_tokens(self, final_tokens: torch.Tensor) -> torch.Tensor:
        """Place final-scale flat tokens into a full PixelVAR pyramid sequence."""
        self._validate_final_tokens(final_tokens)
        start, end = self.tokenizer.boundaries[-1]
        sequence = torch.zeros((final_tokens.shape[0], self.tokenizer.sequence_length), dtype=torch.long, device=final_tokens.device)
        sequence[:, start:end] = final_tokens
        return sequence

    def _validate_final_tokens(self, final_tokens: torch.Tensor) -> None:
        if final_tokens.ndim != 2:
            raise ValueError(f"final_tokens must have shape (B, {self.sequence_length}), got {tuple(final_tokens.shape)}")
        if final_tokens.shape[1] != self.sequence_length:
            raise ValueError(f"final token length {final_tokens.shape[1]} != {self.sequence_length}")
        if final_tokens.min() < 0 or final_tokens.max() >= self.vocab_size:
            raise ValueError(
                f"token range [{int(final_tokens.min())}, {int(final_tokens.max())}] outside [0, {self.vocab_size - 1}]"
            )

    def _validate_input_tokens(self, input_tokens: torch.Tensor) -> None:
        if input_tokens.ndim != 2:
            raise ValueError(f"input_tokens must have shape (B, {self.sequence_length}), got {tuple(input_tokens.shape)}")
        if input_tokens.shape[1] != self.sequence_length:
            raise ValueError(f"input token length {input_tokens.shape[1]} != {self.sequence_length}")
        if input_tokens.min() < 0 or input_tokens.max() > self.mask_token_id:
            raise ValueError(f"input token range must be within [0, {self.mask_token_id}]")

    @staticmethod
    def _keep_count(target_len: int, step: int, refinement_steps: int, mask_schedule: str) -> int:
        if step >= refinement_steps:
            return target_len
        progress = step / refinement_steps
        if mask_schedule == "linear":
            keep_ratio = progress
        elif mask_schedule == "cosine":
            keep_ratio = 1.0 - math.cos(progress * math.pi / 2.0)
        else:
            raise ValueError(f"Unsupported mask_schedule: {mask_schedule}")
        return max(1, min(target_len, int(math.ceil(target_len * keep_ratio))))

    @staticmethod
    def _top_k_logits(logits: torch.Tensor, top_k: int) -> torch.Tensor:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if top_k >= logits.shape[-1]:
            return logits
        values, _ = torch.topk(logits, k=top_k, dim=-1)
        threshold = values[..., -1, None]
        return logits.masked_fill(logits < threshold, torch.finfo(logits.dtype).min)
