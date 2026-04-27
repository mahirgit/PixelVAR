"""Option A scale-wise VAR Transformer for PixelVAR."""

from __future__ import annotations

import torch
from torch import nn

from pixelvar.tokenizers import DeterministicPyramidTokenizer


class VARTransformer(nn.Module):
    """
    Decoder-only-style scale predictor for deterministic PixelVAR tokens.

    For scale ``s``, the model embeds all coarser tokens ``< s`` and appends
    learned query embeddings for scale ``s``. The outputs at query positions
    predict every token in that target scale in parallel.
    """

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
        self.tokenizer = DeterministicPyramidTokenizer(scale_resolutions or [1, 2, 4, 8, 16, 32])
        self.scale_resolutions = self.tokenizer.scale_resolutions
        self.sequence_length = self.tokenizer.sequence_length
        self.boundaries = self.tokenizer.boundaries

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.query_embedding = nn.Embedding(self.sequence_length, d_model)
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
        """Return logits for all scale positions as ``(B, 1365, vocab_size)``."""
        logits = token_sequence.new_zeros(
            (token_sequence.shape[0], self.sequence_length, self.vocab_size),
            dtype=torch.float32,
        )
        for scale_idx, scale_logits in enumerate(self.forward_by_scale(token_sequence)):
            start, end = self.boundaries[scale_idx]
            logits[:, start:end, :] = scale_logits
        return logits

    def forward_by_scale(self, token_sequence: torch.Tensor) -> list[torch.Tensor]:
        """Return one logits tensor per scale."""
        self._validate_sequence(token_sequence)
        return [self.predict_scale(token_sequence, scale_idx) for scale_idx in range(len(self.boundaries))]

    def predict_scale(self, token_sequence: torch.Tensor, scale_idx: int) -> torch.Tensor:
        """Predict target-scale logits from all coarser token context."""
        self._validate_sequence(token_sequence)
        start, end = self.boundaries[scale_idx]
        device = token_sequence.device
        batch = token_sequence.shape[0]

        pieces = []
        if start > 0:
            context_positions = torch.arange(start, device=device)
            context_tokens = token_sequence[:, :start]
            context = (
                self.token_embedding(context_tokens)
                + self.position_embedding(context_positions).unsqueeze(0)
                + self.scale_embedding(self.scale_ids[:start]).unsqueeze(0)
            )
            pieces.append(context)

        target_positions = torch.arange(start, end, device=device)
        queries = (
            self.query_embedding(target_positions)
            + self.position_embedding(target_positions)
            + self.scale_embedding(self.scale_ids[start:end])
        )
        pieces.append(queries.unsqueeze(0).expand(batch, -1, -1))

        hidden = torch.cat(pieces, dim=1)
        hidden = self.input_norm(hidden)
        hidden = self.transformer(hidden)
        target_hidden = hidden[:, -int(end - start) :, :]
        target_hidden = self.output_norm(target_hidden)
        return self.output_head(target_hidden)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        temperature: float = 1.0,
        top_k: int | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Generate token sequences scale by scale."""
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        was_training = self.training
        self.eval()

        if device is None:
            device = next(self.parameters()).device
        tokens = torch.zeros((batch_size, self.sequence_length), dtype=torch.long, device=device)
        for scale_idx, (start, end) in enumerate(self.boundaries):
            logits = self.predict_scale(tokens, scale_idx) / temperature
            if top_k is not None:
                logits = self._top_k_logits(logits, top_k)
            probs = torch.softmax(logits, dim=-1)
            sampled = torch.multinomial(probs.reshape(-1, self.vocab_size), num_samples=1)
            tokens[:, start:end] = sampled.reshape(batch_size, end - start)

        if was_training:
            self.train()
        return tokens

    def _validate_sequence(self, token_sequence: torch.Tensor) -> None:
        if token_sequence.ndim != 2:
            raise ValueError(f"token_sequence must have shape (B, {self.sequence_length}), got {tuple(token_sequence.shape)}")
        if token_sequence.shape[1] != self.sequence_length:
            raise ValueError(f"token_sequence length {token_sequence.shape[1]} != {self.sequence_length}")
        if token_sequence.min() < 0 or token_sequence.max() >= self.vocab_size:
            raise ValueError(
                f"token range [{int(token_sequence.min())}, {int(token_sequence.max())}] outside [0, {self.vocab_size - 1}]"
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
