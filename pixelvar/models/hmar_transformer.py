"""Hierarchical masked autoregressive Transformer for PixelVAR."""

from __future__ import annotations

import math

import torch
from torch import nn

from pixelvar.tokenizers import DeterministicPyramidTokenizer


class HMARTransformer(nn.Module):
    """
    Coarse-to-fine masked refiner over PixelVAR palette tokens.

    For each scale, coarser scales are fixed context and target-scale tokens are
    partially masked. The model predicts every target position in parallel.
    Sampling starts each scale fully masked and repeatedly keeps the most
    confident predictions.
    """

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
        self.sequence_length = self.tokenizer.sequence_length
        self.boundaries = self.tokenizer.boundaries

        input_vocab_size = max(vocab_size, self.mask_token_id + 1)
        self.token_embedding = nn.Embedding(input_vocab_size, d_model)
        self.position_embedding = nn.Embedding(self.sequence_length, d_model)
        self.scale_embedding = nn.Embedding(len(self.scale_resolutions), d_model)
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

        scale_ids = torch.empty(self.sequence_length, dtype=torch.long)
        for scale_idx, (start, end) in enumerate(self.boundaries):
            scale_ids[start:end] = scale_idx
        self.register_buffer("scale_ids", scale_ids, persistent=False)

    def forward(self, token_sequence: torch.Tensor) -> torch.Tensor:
        """Return one-pass fully masked logits for every scale."""
        self._validate_context_sequence(token_sequence)
        logits = token_sequence.new_zeros(
            (token_sequence.shape[0], self.sequence_length, self.vocab_size),
            dtype=torch.float32,
        )
        for scale_idx, (start, end) in enumerate(self.boundaries):
            target_input = torch.full(
                (token_sequence.shape[0], end - start),
                self.mask_token_id,
                dtype=torch.long,
                device=token_sequence.device,
            )
            logits[:, start:end, :] = self.predict_scale(token_sequence, target_input, scale_idx)
        return logits

    def predict_scale(
        self,
        token_sequence: torch.Tensor,
        target_input: torch.Tensor,
        scale_idx: int,
    ) -> torch.Tensor:
        """Predict target-scale logits from coarser context and masked target input."""
        self._validate_context_sequence(token_sequence)
        start, end = self.boundaries[scale_idx]
        target_len = end - start
        if target_input.shape != (token_sequence.shape[0], target_len):
            raise ValueError(f"target_input must have shape {(token_sequence.shape[0], target_len)}, got {tuple(target_input.shape)}")
        if target_input.min() < 0 or target_input.max() > self.mask_token_id:
            raise ValueError(f"target_input values must be in [0, {self.mask_token_id}]")

        device = token_sequence.device
        pieces = []
        if start > 0:
            context_positions = torch.arange(start, device=device)
            context = (
                self.token_embedding(token_sequence[:, :start])
                + self.position_embedding(context_positions).unsqueeze(0)
                + self.scale_embedding(self.scale_ids[:start]).unsqueeze(0)
            )
            pieces.append(context)

        target_positions = torch.arange(start, end, device=device)
        target = (
            self.token_embedding(target_input)
            + self.position_embedding(target_positions).unsqueeze(0)
            + self.scale_embedding(self.scale_ids[start:end]).unsqueeze(0)
        )
        pieces.append(target)

        hidden = torch.cat(pieces, dim=1)
        hidden = self.input_norm(hidden)
        hidden = self.transformer(hidden)
        target_hidden = hidden[:, -target_len:, :]
        target_hidden = self.output_norm(target_hidden)
        return self.output_head(target_hidden)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        refinement_steps: int = 4,
        temperature: float = 1.0,
        top_k: int | None = None,
        mask_schedule: str = "cosine",
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Generate token sequences with iterative masked refinement per scale."""
        if refinement_steps <= 0:
            raise ValueError("refinement_steps must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        was_training = self.training
        self.eval()

        if device is None:
            device = next(self.parameters()).device
        tokens = torch.zeros((batch_size, self.sequence_length), dtype=torch.long, device=device)
        for scale_idx, (start, end) in enumerate(self.boundaries):
            target_len = end - start
            target = torch.full((batch_size, target_len), self.mask_token_id, dtype=torch.long, device=device)
            kept = torch.zeros((batch_size, target_len), dtype=torch.bool, device=device)

            for step_idx in range(refinement_steps):
                logits = self.predict_scale(tokens, target, scale_idx) / temperature
                if top_k is not None:
                    logits = self._top_k_logits(logits, top_k)
                probs = torch.softmax(logits, dim=-1)
                sampled = torch.multinomial(probs.reshape(-1, self.vocab_size), num_samples=1).reshape(batch_size, target_len)
                confidence = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

                target = torch.where(kept, target, sampled)
                confidence = confidence.masked_fill(kept, torch.inf)

                keep_count = self._keep_count(target_len, step_idx + 1, refinement_steps, mask_schedule)
                keep_indices = confidence.topk(k=keep_count, dim=1).indices
                next_kept = torch.zeros_like(kept)
                next_kept.scatter_(1, keep_indices, True)
                kept = next_kept
                target = target.masked_fill(~kept, self.mask_token_id)

            tokens[:, start:end] = target.clamp(max=self.vocab_size - 1)

        if was_training:
            self.train()
        return tokens

    def _validate_context_sequence(self, token_sequence: torch.Tensor) -> None:
        if token_sequence.ndim != 2:
            raise ValueError(f"token_sequence must have shape (B, {self.sequence_length}), got {tuple(token_sequence.shape)}")
        if token_sequence.shape[1] != self.sequence_length:
            raise ValueError(f"token_sequence length {token_sequence.shape[1]} != {self.sequence_length}")
        if token_sequence.min() < 0 or token_sequence.max() >= self.vocab_size:
            raise ValueError(
                f"token range [{int(token_sequence.min())}, {int(token_sequence.max())}] outside [0, {self.vocab_size - 1}]"
            )

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
