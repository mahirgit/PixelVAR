"""Lightning module for VQ-VAE tokenizer training."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

try:
    import lightning as L
except ModuleNotFoundError:  # pragma: no cover
    L = None

from pixelvar.models import VQVAE


if L is None:
    _LightningModule = torch.nn.Module
else:
    _LightningModule = L.LightningModule


class LitVQVAE(_LightningModule):
    """Train a learned discrete tokenizer over RGBA sprites."""

    def __init__(
        self,
        model_config: dict[str, Any] | None = None,
        optimizer_config: dict[str, Any] | None = None,
        loss_config: dict[str, Any] | None = None,
    ):
        if L is None:
            raise ModuleNotFoundError("Install Lightning with `pip install lightning` to use LitVQVAE")
        super().__init__()
        self.model_config = model_config or {}
        self.optimizer_config = optimizer_config or {}
        self.loss_config = loss_config or {}
        self.model = VQVAE(**self.model_config)
        self.save_hyperparameters()

    def forward(self, image: torch.Tensor):
        return self.model(image)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        metrics = self._shared_step(batch, "train")
        return metrics["loss"]

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        metrics = self._shared_step(batch, "val")
        return metrics["loss"]

    def configure_optimizers(self):
        lr = float(self.optimizer_config.get("lr", 2e-4))
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
        target = batch["image"].float()
        output = self.model(target)
        recon = output.recon

        alpha = target[:, 3:4]
        opaque_weight = float(self.loss_config.get("opaque_rgb_weight", 1.0))
        transparent_weight = float(self.loss_config.get("transparent_rgb_weight", 0.05))
        alpha_weight = float(self.loss_config.get("alpha_weight", 1.0))
        code_entropy_weight = float(self.loss_config.get("code_entropy_weight", 0.0))

        rgb_weight = alpha * opaque_weight + (1.0 - alpha) * transparent_weight
        rgb_loss = (torch.abs(recon[:, :3] - target[:, :3]) * rgb_weight).mean()
        alpha_loss = F.binary_cross_entropy(recon[:, 3:4].clamp(1e-5, 1.0 - 1e-5), alpha)
        code_probs = F.one_hot(output.code_indices.reshape(-1), self.model.num_codes).float().mean(dim=0)
        code_entropy = -torch.sum(code_probs * torch.log(code_probs + 1e-10))
        loss = rgb_loss + alpha_weight * alpha_loss + output.loss_vq + output.loss_commit - code_entropy_weight * code_entropy

        hard_alpha = recon[:, 3:4] >= 0.5
        target_alpha = alpha >= 0.5
        alpha_acc = (hard_alpha == target_alpha).float().mean()

        self.log(f"{stage}/loss", loss, on_step=stage == "train", on_epoch=True, prog_bar=True, batch_size=target.shape[0])
        self.log(f"{stage}/rgb_l1", rgb_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=target.shape[0])
        self.log(f"{stage}/alpha_bce", alpha_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=target.shape[0])
        self.log(f"{stage}/vq_loss", output.loss_vq, on_step=False, on_epoch=True, prog_bar=False, batch_size=target.shape[0])
        self.log(f"{stage}/commit_loss", output.loss_commit, on_step=False, on_epoch=True, prog_bar=False, batch_size=target.shape[0])
        self.log(f"{stage}/perplexity", output.perplexity, on_step=False, on_epoch=True, prog_bar=True, batch_size=target.shape[0])
        self.log(f"{stage}/usage_fraction", output.usage_fraction, on_step=False, on_epoch=True, prog_bar=False, batch_size=target.shape[0])
        self.log(f"{stage}/code_entropy", code_entropy, on_step=False, on_epoch=True, prog_bar=False, batch_size=target.shape[0])
        self.log(f"{stage}/alpha_acc", alpha_acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=target.shape[0])
        return {"loss": loss}
