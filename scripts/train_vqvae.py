#!/usr/bin/env python3
"""Train the learned VQ-VAE tokenizer."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import lightning as L
    import torch
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger, WandbLogger
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("Lightning is not installed. Run: pip install 'lightning>=2.6,<2.7'") from exc

from pixelvar.data.vqvae_dataset import get_sprite_image_dataloader
from pixelvar.training import LitVQVAE
from pixelvar.utils import load_yaml


def build_logger(config: dict, run_name: str, output_dir: Path):
    logger_config = config.get("logger", {})
    logger_name = logger_config.get("name", "csv")
    if logger_name == "wandb" and os.environ.get("WANDB_API_KEY"):
        return WandbLogger(
            project=logger_config.get("project", "pixelvar"),
            name=run_name,
            save_dir=str(output_dir),
        )
    return CSVLogger(save_dir=str(output_dir), name=run_name)


def build_callbacks(config: dict, run_name: str) -> list:
    checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints")) / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    monitor = config.get("checkpoint_monitor", "val/loss")
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best",
            monitor=monitor,
            mode="min",
            save_last=True,
            save_top_k=1,
            auto_insert_metric_name=False,
        )
    ]
    if config.get("log_lr", True):
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    early_stopping = config.get("early_stopping")
    if early_stopping:
        early_stopping_config = {} if early_stopping is True else dict(early_stopping)
        callbacks.append(
            EarlyStopping(
                monitor=early_stopping_config.get("monitor", monitor),
                mode=early_stopping_config.get("mode", "min"),
                patience=int(early_stopping_config.get("patience", 20)),
                min_delta=float(early_stopping_config.get("min_delta", 0.0)),
            )
        )
    return callbacks


def trainer_kwargs(config: dict) -> dict:
    trainer_config = config.get("trainer", {})
    allowed = {
        "accelerator",
        "devices",
        "precision",
        "max_epochs",
        "max_steps",
        "log_every_n_steps",
        "gradient_clip_val",
        "limit_train_batches",
        "limit_val_batches",
        "num_sanity_val_steps",
        "fast_dev_run",
        "detect_anomaly",
    }
    return {key: value for key, value in trainer_config.items() if key in allowed}


def build_loaders(config: dict):
    data_config = dict(config.get("data", {}))
    processed_dir = data_config.pop("processed_dir")
    batch_size = int(data_config.pop("batch_size", 256))
    num_workers = int(data_config.pop("num_workers", 4))
    image_array = data_config.pop("image_array", "originals_rgba.npy")
    max_train_samples = data_config.pop("max_train_samples", None)
    max_val_samples = data_config.pop("max_val_samples", None)
    if data_config:
        raise ValueError(f"Unsupported data config keys: {sorted(data_config)}")
    train_loader = get_sprite_image_dataloader(
        processed_dir=processed_dir,
        split="train",
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        image_array=image_array,
        max_samples=max_train_samples,
    )
    val_loader = get_sprite_image_dataloader(
        processed_dir=processed_dir,
        split="val",
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        image_array=image_array,
        max_samples=max_val_samples,
    )
    return train_loader, val_loader


def write_run_report(config: dict, run_name: str, output_dir: Path, checkpoint_callback: ModelCheckpoint) -> None:
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    report = [
        f"# {run_name}",
        "",
        "## Artifacts",
        f"- Best checkpoint: `{checkpoint_callback.best_model_path or 'not available'}`",
        f"- Last checkpoint: `{checkpoint_callback.last_model_path or 'not available'}`",
        "",
        "## Config",
        "```",
        str(config),
        "```",
        "",
        "## Notes",
        "- Learned VQ-VAE tokenizer training run.",
        "- Inspect reconstructions before exporting VAR tokens.",
    ]
    (run_dir / "run_report.md").write_text("\n".join(report))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PixelVAR VQ-VAE tokenizer")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Resume from the run's last checkpoint")
    args = parser.parse_args()

    config = load_yaml(args.config)
    run_name = config.get("run_name", args.config.stem)
    output_dir = Path(config.get("output_dir", "outputs/runs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config.get("seed", 42))
    L.seed_everything(seed, workers=True)
    matmul_precision = config.get("matmul_precision")
    if matmul_precision:
        torch.set_float32_matmul_precision(str(matmul_precision))

    train_loader, val_loader = build_loaders(config)
    optimizer_config = config.get("optimizer", {})
    optimizer_config.setdefault("max_epochs", config.get("trainer", {}).get("max_epochs", 100))
    module = LitVQVAE(
        model_config=config.get("model", {}),
        optimizer_config=optimizer_config,
        loss_config=config.get("loss", {}),
    )

    logger = build_logger(config, run_name, output_dir)
    callbacks = build_callbacks(config, run_name)
    trainer = L.Trainer(
        logger=logger,
        callbacks=callbacks,
        default_root_dir=str(output_dir / run_name),
        **trainer_kwargs(config),
    )

    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, run_dir / "config.yaml")

    ckpt_path = None
    if args.resume:
        last = Path(config.get("checkpoint_dir", "checkpoints")) / run_name / "last.ckpt"
        ckpt_path = str(last) if last.exists() else None
        if ckpt_path is None:
            print(f"[warn] --resume requested but no last checkpoint found at {last}")

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)

    checkpoint_callback = next(cb for cb in callbacks if isinstance(cb, ModelCheckpoint))
    write_run_report(config, run_name, output_dir, checkpoint_callback)
    print(f"Run report written to {run_dir / 'run_report.md'}")


if __name__ == "__main__":
    main()
