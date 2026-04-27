"""Lightning module for PixelVAR Option A training."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

try:
    import lightning as L
except ModuleNotFoundError:  # pragma: no cover - only reached before deps are installed.
    L = None

from pixelvar.models import VARTransformer


if L is None:
    _LightningModule = torch.nn.Module
else:
    _LightningModule = L.LightningModule


class LitVAR(_LightningModule):
    """Lightning wrapper with token-weighted scale losses and metrics."""

    def __init__(
        self,
        model_config: dict[str, Any] | None = None,
        optimizer_config: dict[str, Any] | None = None,
    ):
        if L is None:
            raise ModuleNotFoundError("Install Lightning with `pip install lightning` to use LitVAR")
        super().__init__()
        self.model_config = model_config or {}
        self.optimizer_config = optimizer_config or {}
        self.model = VARTransformer(**self.model_config)
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
    def sample(self, batch_size: int = 16, temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        return self.model.sample(batch_size=batch_size, temperature=temperature, top_k=top_k, device=self.device)

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
        logits_by_scale = self.model.forward_by_scale(tokens)

        total_loss = tokens.new_tensor(0.0, dtype=torch.float32)
        total_correct = tokens.new_tensor(0.0, dtype=torch.float32)
        total_tokens = 0

        for scale_idx, logits in enumerate(logits_by_scale):
            start, end = self.model.boundaries[scale_idx]
            target = tokens[:, start:end]
            scale_loss = F.cross_entropy(logits.reshape(-1, self.model.vocab_size), target.reshape(-1))
            pred = logits.argmax(dim=-1)
            correct = (pred == target).float().sum()
            count = target.numel()

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
