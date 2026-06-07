"""Checkpoint loading helpers that do not require Lightning at inference time."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from pixelvar.models import FlatARTransformer, FlatMaskGITTransformer, HMARTransformer, VARTransformer, VQVAE


def _load_checkpoint_payload(checkpoint: str | Path, map_location: str | torch.device) -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    try:
        return torch.load(checkpoint, map_location=map_location)
    except TypeError:
        return torch.load(checkpoint, map_location=map_location, weights_only=False)


def load_var_model_from_checkpoint(checkpoint: str | Path, map_location: str | torch.device = "cpu") -> VARTransformer:
    """Load a ``VARTransformer`` from a Lightning checkpoint using plain PyTorch."""
    payload = _load_checkpoint_payload(checkpoint, map_location)

    hparams = payload.get("hyper_parameters", {})
    model_config = hparams.get("model_config", {})
    model = VARTransformer(**model_config)

    state_dict = payload.get("state_dict", payload)
    model_state = {
        key.removeprefix("model."): value
        for key, value in state_dict.items()
        if key.startswith("model.")
    }
    if not model_state:
        model_state = state_dict
    model.load_state_dict(model_state)
    model.eval()
    return model


def load_flat_ar_model_from_checkpoint(checkpoint: str | Path, map_location: str | torch.device = "cpu") -> FlatARTransformer:
    """Load a ``FlatARTransformer`` from a Lightning checkpoint using plain PyTorch."""
    payload = _load_checkpoint_payload(checkpoint, map_location)

    hparams = payload.get("hyper_parameters", {})
    model_config = hparams.get("model_config", {})
    model = FlatARTransformer(**model_config)

    state_dict = payload.get("state_dict", payload)
    model_state = {
        key.removeprefix("model."): value
        for key, value in state_dict.items()
        if key.startswith("model.")
    }
    if not model_state:
        model_state = state_dict
    model.load_state_dict(model_state)
    model.eval()
    return model


def load_flat_maskgit_model_from_checkpoint(
    checkpoint: str | Path,
    map_location: str | torch.device = "cpu",
) -> FlatMaskGITTransformer:
    """Load a ``FlatMaskGITTransformer`` from a Lightning checkpoint using plain PyTorch."""
    payload = _load_checkpoint_payload(checkpoint, map_location)

    hparams = payload.get("hyper_parameters", {})
    model_config = hparams.get("model_config", {})
    model = FlatMaskGITTransformer(**model_config)

    state_dict = payload.get("state_dict", payload)
    model_state = {
        key.removeprefix("model."): value
        for key, value in state_dict.items()
        if key.startswith("model.")
    }
    if not model_state:
        model_state = state_dict
    model.load_state_dict(model_state)
    model.eval()
    return model


def load_hmar_model_from_checkpoint(checkpoint: str | Path, map_location: str | torch.device = "cpu") -> HMARTransformer:
    """Load an ``HMARTransformer`` from a Lightning checkpoint using plain PyTorch."""
    payload = _load_checkpoint_payload(checkpoint, map_location)

    hparams = payload.get("hyper_parameters", {})
    model_config = hparams.get("model_config", {})
    model = HMARTransformer(**model_config)

    state_dict = payload.get("state_dict", payload)
    model_state = {
        key.removeprefix("model."): value
        for key, value in state_dict.items()
        if key.startswith("model.")
    }
    if not model_state:
        model_state = state_dict
    model.load_state_dict(model_state)
    model.eval()
    return model


def load_vqvae_model_from_checkpoint(checkpoint: str | Path, map_location: str | torch.device = "cpu") -> VQVAE:
    """Load a ``VQVAE`` from a Lightning checkpoint using plain PyTorch."""
    payload = _load_checkpoint_payload(checkpoint, map_location)

    hparams = payload.get("hyper_parameters", {})
    model_config = hparams.get("model_config", {})
    model = VQVAE(**model_config)

    state_dict = payload.get("state_dict", payload)
    model_state = {
        key.removeprefix("model."): value
        for key, value in state_dict.items()
        if key.startswith("model.")
    }
    if not model_state:
        model_state = state_dict
    model.load_state_dict(model_state)
    model.eval()
    return model
