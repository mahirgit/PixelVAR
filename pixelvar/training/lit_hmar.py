"""Lightning module for PixelVAR HMAR masked refinement."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

try:
    import lightning as L
except ModuleNotFoundError:  # pragma: no cover - only reached before deps are installed.
    L = None

from pixelvar.models import HMARTransformer


if L is None:
    _LightningModule = torch.nn.Module
else:
    _LightningModule = L.LightningModule


class LitHMAR(_LightningModule):
    """Lightning wrapper for hierarchical masked refinement training."""

    def __init__(
        self,
        model_config: dict[str, Any] | None = None,
        optimizer_config: dict[str, Any] | None = None,
        masking_config: dict[str, Any] | None = None,
    ):
        if L is None:
            raise ModuleNotFoundError("Install Lightning with `pip install lightning` to use LitHMAR")
        super().__init__()
        self.model_config = model_config or {}
        self.optimizer_config = optimizer_config or {}
        self.masking_config = masking_config or {}
        self.model = HMARTransformer(**self.model_config)
        self.save_hyperparameters()

    def forward(self, token_sequence: torch.Tensor) -> torch.Tensor:
        return self.model(token_sequence)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        metrics = self._shared_step(batch, stage="train")
        return metrics["loss"]

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        metrics = self._shared_step(batch, stage="val")
        return metrics["loss"]

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        metrics = self._shared_step(batch, stage="test")
        return metrics["loss"]

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 16,
        refinement_steps: int = 4,
        temperature: float = 1.0,
        top_k: int | None = None,
        mask_schedule: str = "cosine",
    ) -> torch.Tensor:
        return self.model.sample(
            batch_size=batch_size,
            refinement_steps=refinement_steps,
            temperature=temperature,
            top_k=top_k,
            mask_schedule=mask_schedule,
            device=self.device,
        )

    def configure_optimizers(self):
        lr = float(self.optimizer_config.get("lr", 3e-4))
        weight_decay = float(self.optimizer_config.get("weight_decay", 0.01))
        betas = tuple(self.optimizer_config.get("betas", (0.9, 0.95)))
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)

        scheduler_name = self.optimizer_config.get("scheduler")
        if scheduler_name is None:
            return optimizer
        if scheduler_name != "cosine":
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
        max_epochs = int(self.optimizer_config.get("max_epochs", 100))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def _shared_step(self, batch: dict, stage: str) -> dict[str, torch.Tensor]:
        tokens = batch["token_sequence"].long()

        total_loss = tokens.new_tensor(0.0, dtype=torch.float32)
        total_correct = tokens.new_tensor(0.0, dtype=torch.float32)
        total_tokens = 0

        for scale_idx, (start, end) in enumerate(self.model.boundaries):
            target = tokens[:, start:end]
            mask = self._sample_mask(target.shape, target.device, stage)
            target_input = target.clone()
            target_input[mask] = self.model.mask_token_id

            logits = self.model.predict_scale(tokens, target_input, scale_idx)
            masked_logits = logits[mask]
            masked_target = target[mask]
            scale_loss = F.cross_entropy(masked_logits, masked_target)
            pred = masked_logits.argmax(dim=-1)
            correct = (pred == masked_target).float().sum()
            count = int(masked_target.numel())

            total_loss = total_loss + scale_loss * count
            total_correct = total_correct + correct
            total_tokens += count

            self.log(
                f"{stage}/loss_s{scale_idx}_{self.model.scale_resolutions[scale_idx]}",
                scale_loss,
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=False,
                batch_size=tokens.shape[0],
            )
            self.log(
                f"{stage}/acc_s{scale_idx}_{self.model.scale_resolutions[scale_idx]}",
                correct / count,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=tokens.shape[0],
            )

        loss = total_loss / total_tokens
        acc = total_correct / total_tokens
        self.log(
            f"{stage}/loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            batch_size=tokens.shape[0],
        )
        self.log(
            f"{stage}/acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=tokens.shape[0],
        )
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
            if ratio >= 1.0:
                return torch.ones((batch, target_len), dtype=torch.bool, device=device)
            mask = torch.rand((batch, target_len), device=device) < ratio

        empty_rows = ~mask.any(dim=1)
        if empty_rows.any():
            rows = empty_rows.nonzero(as_tuple=False).flatten()
            cols = torch.randint(0, target_len, (rows.numel(),), device=device)
            mask[rows, cols] = True
        return mask
