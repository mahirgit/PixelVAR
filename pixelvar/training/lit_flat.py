"""Lightning modules for flat PixelVAR baselines."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

try:
    import lightning as L
except ModuleNotFoundError:  # pragma: no cover - only reached before deps are installed.
    L = None

from pixelvar.models import FlatARTransformer, FlatMaskGITTransformer


if L is None:
    _LightningModule = torch.nn.Module
else:
    _LightningModule = L.LightningModule


class LitFlatAR(_LightningModule):
    """Lightning wrapper for the flat raster autoregressive baseline."""

    def __init__(
        self,
        model_config: dict[str, Any] | None = None,
        optimizer_config: dict[str, Any] | None = None,
    ):
        if L is None:
            raise ModuleNotFoundError("Install Lightning with `pip install lightning` to use LitFlatAR")
        super().__init__()
        self.model_config = model_config or {}
        self.optimizer_config = optimizer_config or {}
        self.model = FlatARTransformer(**self.model_config)
        self.save_hyperparameters()

    def forward(self, final_tokens: torch.Tensor) -> torch.Tensor:
        return self.model(final_tokens)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        metrics = self._shared_step(batch, stage="train")
        return metrics["loss"]

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        metrics = self._shared_step(batch, stage="val")
        return metrics["loss"]

    @torch.no_grad()
    def sample(self, batch_size: int = 16, temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        final_tokens = self.model.sample(batch_size=batch_size, temperature=temperature, top_k=top_k, device=self.device)
        return self.model.full_sequence_from_final_tokens(final_tokens)

    def configure_optimizers(self):
        return _configure_optimizers(self.parameters(), self.optimizer_config)

    def _shared_step(self, batch: dict, stage: str) -> dict[str, torch.Tensor]:
        final_tokens = _final_tokens(batch["token_sequence"].long(), self.model)
        logits = self.model(final_tokens)
        loss = F.cross_entropy(logits.reshape(-1, self.model.vocab_size), final_tokens.reshape(-1))
        pred = logits.argmax(dim=-1)
        acc = (pred == final_tokens).float().mean()
        self.log(f"{stage}/loss", loss, on_step=stage == "train", on_epoch=True, prog_bar=True, batch_size=final_tokens.shape[0])
        self.log(f"{stage}/acc", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=final_tokens.shape[0])
        return {"loss": loss, "acc": acc}


class LitFlatMaskGIT(_LightningModule):
    """Lightning wrapper for the flat MaskGIT-style baseline."""

    def __init__(
        self,
        model_config: dict[str, Any] | None = None,
        optimizer_config: dict[str, Any] | None = None,
        masking_config: dict[str, Any] | None = None,
    ):
        if L is None:
            raise ModuleNotFoundError("Install Lightning with `pip install lightning` to use LitFlatMaskGIT")
        super().__init__()
        self.model_config = model_config or {}
        self.optimizer_config = optimizer_config or {}
        self.masking_config = masking_config or {}
        self.model = FlatMaskGITTransformer(**self.model_config)
        self.save_hyperparameters()

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        return self.model(input_tokens)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        metrics = self._shared_step(batch, stage="train")
        return metrics["loss"]

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        metrics = self._shared_step(batch, stage="val")
        return metrics["loss"]

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 16,
        refinement_steps: int = 8,
        temperature: float = 1.0,
        top_k: int | None = None,
        mask_schedule: str = "cosine",
    ) -> torch.Tensor:
        final_tokens = self.model.sample(
            batch_size=batch_size,
            refinement_steps=refinement_steps,
            temperature=temperature,
            top_k=top_k,
            mask_schedule=mask_schedule,
            device=self.device,
        )
        return self.model.full_sequence_from_final_tokens(final_tokens)

    def configure_optimizers(self):
        return _configure_optimizers(self.parameters(), self.optimizer_config)

    def _shared_step(self, batch: dict, stage: str) -> dict[str, torch.Tensor]:
        final_tokens = _final_tokens(batch["token_sequence"].long(), self.model)
        mask = self._sample_mask(final_tokens.shape, final_tokens.device, stage)
        input_tokens = final_tokens.clone()
        input_tokens[mask] = self.model.mask_token_id
        logits = self.model(input_tokens)
        masked_logits = logits[mask]
        masked_target = final_tokens[mask]
        loss = F.cross_entropy(masked_logits, masked_target)
        pred = masked_logits.argmax(dim=-1)
        acc = (pred == masked_target).float().mean()
        self.log(f"{stage}/loss", loss, on_step=stage == "train", on_epoch=True, prog_bar=True, batch_size=final_tokens.shape[0])
        self.log(f"{stage}/acc", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=final_tokens.shape[0])
        return {"loss": loss, "acc": acc}

    def _sample_mask(self, shape: torch.Size | tuple[int, int], device: torch.device, stage: str) -> torch.Tensor:
        batch, target_len = int(shape[0]), int(shape[1])
        if stage == "train":
            min_ratio = float(self.masking_config.get("mask_ratio_min", 0.1))
            max_ratio = float(self.masking_config.get("mask_ratio_max", 1.0))
            full_mask_prob = float(self.masking_config.get("full_mask_prob", 0.25))
            ratios = torch.empty((batch, 1), device=device).uniform_(min_ratio, max_ratio)
            if full_mask_prob > 0:
                full = torch.rand((batch, 1), device=device) < full_mask_prob
                ratios = torch.where(full, torch.ones_like(ratios), ratios)
            mask = torch.rand((batch, target_len), device=device) < ratios
        else:
            ratio = float(self.masking_config.get("validation_mask_ratio", 1.0))
            mask = torch.ones((batch, target_len), dtype=torch.bool, device=device) if ratio >= 1.0 else torch.rand((batch, target_len), device=device) < ratio

        empty_rows = ~mask.any(dim=1)
        if empty_rows.any():
            rows = empty_rows.nonzero(as_tuple=False).flatten()
            cols = torch.randint(0, target_len, (rows.numel(),), device=device)
            mask[rows, cols] = True
        return mask


def _final_tokens(token_sequence: torch.Tensor, model: FlatARTransformer | FlatMaskGITTransformer) -> torch.Tensor:
    start, end = model.tokenizer.boundaries[-1]
    return token_sequence[:, start:end]


def _configure_optimizers(parameters, optimizer_config: dict[str, Any]):
    lr = float(optimizer_config.get("lr", 3e-4))
    weight_decay = float(optimizer_config.get("weight_decay", 0.01))
    betas = tuple(optimizer_config.get("betas", (0.9, 0.95)))
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay, betas=betas)

    scheduler_name = optimizer_config.get("scheduler")
    if scheduler_name is None:
        return optimizer
    if scheduler_name != "cosine":
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    max_epochs = int(optimizer_config.get("max_epochs", 100))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
