#!/usr/bin/env python3
"""Train PixelVAR Option A with Lightning."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import lightning as L
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger, WandbLogger
except ModuleNotFoundError as exc:  # pragma: no cover - CLI guard.
    raise SystemExit("Lightning is not installed. Run: pip install 'lightning>=2.6,<2.7'") from exc

from pixelvar.data.datamodule import PixelArtDataModule
from pixelvar.training import LitVAR
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
        "- Option A deterministic tokenizer training run.",
        "- Inspect per-scale loss and accuracy before promoting to the next ladder stage.",
    ]
    (run_dir / "run_report.md").write_text("\n".join(report))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PixelVAR Option A")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Resume from the run's last checkpoint")
    args = parser.parse_args()

    config = load_yaml(args.config)
    run_name = config.get("run_name", args.config.stem)
    output_dir = Path(config.get("output_dir", "outputs/runs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config.get("seed", 42))
    L.seed_everything(seed, workers=True)

    data_config = config.get("data", {})
    datamodule = PixelArtDataModule(**data_config)

    model_config = config.get("model", {})
    optimizer_config = config.get("optimizer", {})
    optimizer_config.setdefault("max_epochs", config.get("trainer", {}).get("max_epochs", 100))
    module = LitVAR(model_config=model_config, optimizer_config=optimizer_config)

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

    trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)

    checkpoint_callback = next(cb for cb in callbacks if isinstance(cb, ModelCheckpoint))
    write_run_report(config, run_name, output_dir, checkpoint_callback)
    print(f"Run report written to {run_dir / 'run_report.md'}")


if __name__ == "__main__":
    main()
