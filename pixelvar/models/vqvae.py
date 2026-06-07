"""Small VQ-VAE tokenizer for 32x32 RGBA sprites."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class VQVAEOutput:
    recon: torch.Tensor
    code_indices: torch.Tensor
    loss_vq: torch.Tensor
    loss_commit: torch.Tensor
    perplexity: torch.Tensor
    usage_fraction: torch.Tensor


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class VectorQuantizer(nn.Module):
    """Straight-through vector quantizer."""

    def __init__(self, num_codes: int = 256, embedding_dim: int = 64, commitment_weight: float = 0.25):
        super().__init__()
        self.num_codes = int(num_codes)
        self.embedding_dim = int(embedding_dim)
        self.commitment_weight = float(commitment_weight)
        self.embedding = nn.Embedding(self.num_codes, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0, 1.0)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if z.ndim != 4:
            raise ValueError(f"z must have shape (B, C, H, W), got {tuple(z.shape)}")
        z_perm = z.permute(0, 2, 3, 1).contiguous()
        flat = z_perm.reshape(-1, self.embedding_dim)
        codebook = self.embedding.weight

        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat @ codebook.t()
            + codebook.pow(2).sum(dim=1).unsqueeze(0)
        )
        indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(indices).reshape_as(z_perm)

        loss_vq = F.mse_loss(quantized, z_perm.detach())
        loss_commit = F.mse_loss(z_perm, quantized.detach())
        quantized = z_perm + (quantized - z_perm).detach()
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        encodings = F.one_hot(indices, self.num_codes).float()
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        usage_fraction = (avg_probs > 0).float().mean()
        code_indices = indices.reshape(z.shape[0], z.shape[2], z.shape[3])
        return quantized, code_indices, loss_vq, self.commitment_weight * loss_commit, perplexity, usage_fraction


class VQVAE(nn.Module):
    """Convolutional VQ-VAE producing an 8x8 learned token map by default."""

    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels: int = 128,
        embedding_dim: int = 64,
        num_codes: int = 256,
        latent_size: int = 8,
        commitment_weight: float = 0.25,
        residual_blocks: int = 2,
    ):
        super().__init__()
        if latent_size not in {4, 8, 16}:
            raise ValueError("latent_size must be one of 4, 8, or 16 for 32x32 sprites")
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.embedding_dim = int(embedding_dim)
        self.num_codes = int(num_codes)
        self.latent_size = int(latent_size)

        downsample_layers = int(torch.log2(torch.tensor(32 // latent_size)).item())
        if 32 // latent_size != 2**downsample_layers:
            raise ValueError("latent_size must divide 32 by a power of two")

        encoder: list[nn.Module] = [
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        ]
        for _ in range(downsample_layers):
            encoder.extend(
                [
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
                    nn.SiLU(),
                ]
            )
        encoder.extend(ResidualBlock(hidden_channels) for _ in range(residual_blocks))
        encoder.append(nn.Conv2d(hidden_channels, embedding_dim, kernel_size=1))
        self.encoder = nn.Sequential(*encoder)

        self.quantizer = VectorQuantizer(num_codes=num_codes, embedding_dim=embedding_dim, commitment_weight=commitment_weight)

        decoder: list[nn.Module] = [
            nn.Conv2d(embedding_dim, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        ]
        decoder.extend(ResidualBlock(hidden_channels) for _ in range(residual_blocks))
        for _ in range(downsample_layers):
            decoder.extend(
                [
                    nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
                    nn.SiLU(),
                ]
            )
        decoder.append(nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1))
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        quantized, indices, *_ = self.quantizer(z)
        return quantized, indices

    def decode(self, quantized: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decoder(quantized))

    def decode_code_indices(self, code_indices: torch.Tensor) -> torch.Tensor:
        indices = code_indices.long()
        if indices.ndim == 2:
            indices = indices.unsqueeze(0)
        if indices.ndim != 3:
            raise ValueError(f"code_indices must have shape (H, W) or (B, H, W), got {tuple(indices.shape)}")
        quantized = self.quantizer.embedding(indices).permute(0, 3, 1, 2).contiguous()
        return self.decode(quantized)

    def forward(self, x: torch.Tensor) -> VQVAEOutput:
        z = self.encoder(x)
        quantized, indices, loss_vq, loss_commit, perplexity, usage_fraction = self.quantizer(z)
        recon = self.decode(quantized)
        return VQVAEOutput(
            recon=recon,
            code_indices=indices,
            loss_vq=loss_vq,
            loss_commit=loss_commit,
            perplexity=perplexity,
            usage_fraction=usage_fraction,
        )
