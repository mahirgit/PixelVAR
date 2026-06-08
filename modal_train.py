"""Run PixelVAR jobs on Modal with a single B200 GPU.

Local setup:

    python -m pip install modal
    modal setup

Examples:

    modal run modal_train.py --action cuda-check
    modal run modal_train.py --action prepare-pokemon
    modal run modal_train.py --action train-overfit32
    modal run modal_train.py --action train --config configs/train/debug1k.yaml
    modal run modal_train.py --action eval-sprites
    modal run modal_train.py --action generate-sprites-selected
    modal run modal_train.py --action prepare-sprites-generated-keep
    modal run modal_train.py --action train-sprites-generated-keep-v0-full
    modal run modal_train.py --action prepare-sprites-mixed
    modal run modal_train.py --action train-sprites-mixed-v0-full
    modal run modal_train.py --action prepare-opengameart-public
    modal run modal_train.py --action prepare-sprites-mixed-opengameart
    modal run modal_train.py --action train-sprites-mixed-oga-v0-full
    modal run modal_train.py --action train-vqvae-sprites-v0-full
    modal run modal_train.py --action export-vqvae-sprites
    modal run modal_train.py --action train-sprites-vqvae16-v0-full
    modal run modal_train.py --action prepare-patch-vq-sprites
    modal run modal_train.py --action train-sprites-patchvq16-v0-full
    modal run modal_train.py --action train-sprites-hmar-v0-full
    modal run modal_train.py --action eval-hmar-refinement-ablation
    modal run modal_train.py --action benchmark-main-hmar-known-metrics
    modal run modal_train.py --action train-flat-ar-ladder
    modal run modal_train.py --action train-flat-maskgit-ladder
    modal run modal_train.py --action benchmark-main-hmar-flat-known-metrics
    modal run modal_train.py --action audit-flat-ar-memorization
    modal run modal_train.py --action audit-main-hmar-memorization
    modal run modal_train.py --action build-four-way-sample-sheet
    modal run modal_train.py --action prepare-sd-pixl-baseline
    modal run modal_train.py --action run-sd-pixl-smoke --sd-pixl-steps 250
    modal run modal_train.py --action run-sd-pixl-batch --num-samples 4 --sd-pixl-steps 1000
    modal run modal_train.py --action run-practical-diffusion-smoke --num-samples 4

Persistent Modal volumes:

    pixelvar-data         -> /workspace/PixelVAR/data
    pixelvar-outputs      -> /workspace/PixelVAR/outputs
    pixelvar-checkpoints  -> /workspace/PixelVAR/checkpoints
"""

from __future__ import annotations

import shlex
from pathlib import Path

import modal


APP_NAME = "pixelvar-b200"
WORKDIR = "/workspace/PixelVAR"
GPU_TYPE = "B200"
DEFAULT_TIMEOUT = 6 * 60 * 60

REPO_ROOT = Path(__file__).resolve().parent

data_volume = modal.Volume.from_name("pixelvar-data", create_if_missing=True)
outputs_volume = modal.Volume.from_name("pixelvar-outputs", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("pixelvar-checkpoints", create_if_missing=True)

VOLUMES = {
    f"{WORKDIR}/data": data_volume,
    f"{WORKDIR}/outputs": outputs_volume,
    f"{WORKDIR}/checkpoints": checkpoints_volume,
}

IMAGE_IGNORE = [
    ".git/**",
    ".pytest_cache/**",
    "__pycache__/**",
    "**/__pycache__/**",
    "*.pyc",
    ".venv/**",
    "venv/**",
    "env/**",
    "data/**",
    "outputs/**",
    "checkpoints/**",
    "wandb/**",
    "lightning_logs/**",
    "visualizations/**",
    "*.ckpt",
    "*.pt",
    "*.pth",
]

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch",
        "torchvision",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "lightning>=2.6,<2.7",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "requests>=2.31.0",
        "pyyaml>=6.0.0",
        "pytest>=8.0.0",
        "gdown>=5.1.0",
        "kaggle>=1.6.0",
        "datasets>=2.20.0",
        "pyarrow>=15.0.0",
    )
    .env({"PYTHONPATH": WORKDIR, "PYTHONUNBUFFERED": "1"})
    .workdir(WORKDIR)
    .add_local_dir(REPO_ROOT, remote_path=WORKDIR, ignore=IMAGE_IGNORE)
)

sd_pixl_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "torchaudio==2.8.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "accelerate==0.33.0",
        "diffusers==0.31.0",
        "numpy==1.26.4",
        "omegaconf==2.3.0",
        "einops==0.8.0",
        "transformers==4.44.0",
        "scipy==1.14.0",
        "tensorboard==2.17.1",
        "openai-clip==1.0.1",
        "opencv-python==4.10.0.84",
        "scikit-learn==1.5.1",
        "peft==0.12.0",
        "matplotlib==3.9.2",
        "Pillow==10.4.0",
        "protobuf==5.27.3",
        "safetensors==0.4.4",
        "huggingface-hub==0.24.5",
        "requests==2.32.3",
        "pyyaml>=6.0.0",
        "tqdm>=4.66.5",
    )
    .env({"PYTHONPATH": WORKDIR, "PYTHONUNBUFFERED": "1"})
    .workdir(WORKDIR)
    .add_local_dir(REPO_ROOT, remote_path=WORKDIR, ignore=IMAGE_IGNORE)
)

app = modal.App(APP_NAME)

TRAIN_CONFIGS = {
    "train-overfit32": "configs/train/overfit32.yaml",
    "train-debug1k": "configs/train/debug1k.yaml",
    "train-v0-full": "configs/train/v0_full.yaml",
    "train-sprites-overfit32": "configs/train/sprites_overfit32.yaml",
    "train-sprites-debug1k": "configs/train/sprites_debug1k.yaml",
    "train-sprites-v0-full": "configs/train/sprites_v0_full.yaml",
    "train-sprites-generated-keep-overfit32": "configs/train/sprites_generated_keep_overfit32.yaml",
    "train-sprites-generated-keep-debug1k": "configs/train/sprites_generated_keep_debug1k.yaml",
    "train-sprites-generated-keep-v0-full": "configs/train/sprites_generated_keep_v0_full.yaml",
    "train-sprites-mixed-overfit32": "configs/train/sprites_mixed_overfit32.yaml",
    "train-sprites-mixed-debug1k": "configs/train/sprites_mixed_debug1k.yaml",
    "train-sprites-mixed-v0-full": "configs/train/sprites_mixed_v0_full.yaml",
    "train-sprites-mixed-oga-overfit32": "configs/train/sprites_mixed_oga_overfit32.yaml",
    "train-sprites-mixed-oga-debug1k": "configs/train/sprites_mixed_oga_debug1k.yaml",
    "train-sprites-mixed-oga-v0-full": "configs/train/sprites_mixed_oga_v0_full.yaml",
    "train-sprites-vqvae16-overfit32": "configs/train/sprites_vqvae16_overfit32.yaml",
    "train-sprites-vqvae16-debug1k": "configs/train/sprites_vqvae16_debug1k.yaml",
    "train-sprites-vqvae16-v0-full": "configs/train/sprites_vqvae16_v0_full.yaml",
    "train-sprites-patchvq16-overfit32": "configs/train/sprites_patchvq16_overfit32.yaml",
    "train-sprites-patchvq16-debug1k": "configs/train/sprites_patchvq16_debug1k.yaml",
    "train-sprites-patchvq16-v0-full": "configs/train/sprites_patchvq16_v0_full.yaml",
}

VQVAE_CONFIGS = {
    "train-vqvae-sprites-overfit32": "configs/train/vqvae_sprites_overfit32.yaml",
    "train-vqvae-sprites-debug1k": "configs/train/vqvae_sprites_debug1k.yaml",
    "train-vqvae-sprites-v0-full": "configs/train/vqvae_sprites_v0_full.yaml",
}

HMAR_CONFIGS = {
    "train-sprites-hmar-overfit32": "configs/train/sprites_hmar_overfit32.yaml",
    "train-sprites-hmar-debug1k": "configs/train/sprites_hmar_debug1k.yaml",
    "train-sprites-hmar-v0-full": "configs/train/sprites_hmar_v0_full.yaml",
}

FLAT_CONFIGS = {
    "train-sprites-flat-ar-overfit32": "configs/train/sprites_flat_ar_overfit32.yaml",
    "train-sprites-flat-ar-debug1k": "configs/train/sprites_flat_ar_debug1k.yaml",
    "train-sprites-flat-ar-v0-full": "configs/train/sprites_flat_ar_v0_full.yaml",
    "train-sprites-flat-maskgit-overfit32": "configs/train/sprites_flat_maskgit_overfit32.yaml",
    "train-sprites-flat-maskgit-debug1k": "configs/train/sprites_flat_maskgit_debug1k.yaml",
    "train-sprites-flat-maskgit-v0-full": "configs/train/sprites_flat_maskgit_v0_full.yaml",
}


def _quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def _commit_volumes() -> None:
    for name, volume in (
        ("data", data_volume),
        ("outputs", outputs_volume),
        ("checkpoints", checkpoints_volume),
    ):
        print(f"[modal] committing {name} volume")
        volume.commit()


def _reload_volumes() -> None:
    for name, volume in (
        ("data", data_volume),
        ("outputs", outputs_volume),
        ("checkpoints", checkpoints_volume),
    ):
        print(f"[modal] reloading {name} volume")
        volume.reload()


def _run_commands(commands: list[str]) -> None:
    import subprocess

    _reload_volumes()
    try:
        for command in commands:
            print(f"\n[modal] $ {command}", flush=True)
            subprocess.run(command, shell=True, cwd=WORKDIR, check=True)
    finally:
        _commit_volumes()


@app.function(image=image, volumes=VOLUMES, timeout=DEFAULT_TIMEOUT)
def run_cpu(commands: list[str]) -> None:
    """Run non-GPU PixelVAR setup/data commands."""
    _run_commands(commands)


@app.function(image=image, gpu=GPU_TYPE, volumes=VOLUMES, timeout=DEFAULT_TIMEOUT)
def run_b200(commands: list[str]) -> None:
    """Run PixelVAR commands on one Modal B200."""
    _run_commands(commands)


@app.function(image=sd_pixl_image, gpu=GPU_TYPE, volumes=VOLUMES, timeout=DEFAULT_TIMEOUT)
def run_sd_pixl_b200(commands: list[str]) -> None:
    """Run diffusion external-baseline commands on one Modal B200."""
    _run_commands(commands)


def commands_for_action(
    action: str,
    config: str,
    cmd: str,
    resume: bool,
    checkpoint: str,
    num_samples: int,
    temperature: float,
    top_k: int,
    refinement_steps: int,
    output: str,
    transparent_color: str,
    sheet_tile_size: int,
    sprites_frames_per_group: int,
    sd_pixl_steps: int,
    sd_pixl_model_id: str,
    sd_pixl_prompt_index: int,
    sd_pixl_prompt: str,
    diffusion_model_id: str,
    diffusion_lora_id: str,
    diffusion_lora_weight_name: str,
    diffusion_lora_scale: float,
    diffusion_prompt_index: int,
    diffusion_prompt: str,
    diffusion_steps: int,
    diffusion_guidance_scale: float,
    diffusion_height: int,
    diffusion_width: int,
    diffusion_seed: int,
    diffusion_dtype: str,
    diffusion_negative_prompt: str,
) -> tuple[str, list[str]]:
    if action == "cuda-check":
        return (
            "gpu",
            [
                "python - <<'PY'\n"
                "import torch\n"
                "print('torch', torch.__version__)\n"
                "print('cuda_available', torch.cuda.is_available())\n"
                "print('device_count', torch.cuda.device_count())\n"
                "if torch.cuda.is_available():\n"
                "    print('device_name', torch.cuda.get_device_name(0))\n"
                "PY"
            ],
        )

    if action == "prepare-pokemon":
        return (
            "cpu",
            [
                "python scripts/download_data.py --dataset pokemon",
                "python scripts/preprocess_data.py --dataset pokemon --palette-size 16",
                "python scripts/check_data.py --dataset pokemon",
            ],
        )

    if action == "check-pokemon":
        return ("cpu", ["python scripts/check_data.py --dataset pokemon"])

    if action == "download-sprites-public":
        return ("cpu", ["python scripts/download_data.py --dataset sprites"])

    if action == "prepare-sprites":
        return (
            "cpu",
            [
                "python scripts/curate_msd_sprites.py --dataset-name sprites",
                "python scripts/preprocess_data.py --dataset sprites --palette-size 16",
                "python scripts/check_data.py --dataset sprites",
            ],
        )

    if action == "prepare-raw-sprites":
        transparent_arg = f" --transparent-color {_quote(transparent_color)}" if transparent_color else ""
        return (
            "cpu",
            [
                "python scripts/curate_data.py "
                f"--dataset sprites --sprites-frames-per-group {int(sprites_frames_per_group)}"
                f"{transparent_arg}",
                "python scripts/preprocess_data.py --dataset sprites --palette-size 16",
                "python scripts/check_data.py --dataset sprites",
            ],
        )

    if action == "prepare-opengameart":
        transparent_arg = f" --transparent-color {_quote(transparent_color)}" if transparent_color else ""
        return (
            "cpu",
            [
                "python scripts/curate_data.py "
                f"--dataset opengameart --sheet-tile-size {int(sheet_tile_size)}"
                f"{transparent_arg}",
                "python scripts/preprocess_data.py --dataset opengameart --palette-size 16",
                "python scripts/check_data.py --dataset opengameart",
            ],
        )

    if action == "prepare-opengameart-public":
        return (
            "cpu",
            [
                "python scripts/download_opengameart_assets.py --force",
                "python scripts/preprocess_data.py "
                "--dataset opengameart --palette-size 16 "
                "--reference-palette data/processed/sprites/palette.json",
                "python scripts/check_data.py --dataset opengameart",
            ],
        )

    if action == "prepare-sprites-generated-keep":
        generated_name = f"sprites_v0_full_t08_top8_{int(num_samples) if num_samples != 16 else 170000}"
        return (
            "cpu",
            [
                "python scripts/inspect_generated_set.py "
                f"--generated-dir outputs/generated/{generated_name} "
                f"--output-dir outputs/generated/{generated_name}_inspection "
                "--palette data/processed/sprites/palette.json "
                "--reference-summary outputs/eval/sprites_v0_full/reference_summary.json "
                "--write-keep-package",
                "python scripts/import_generated_keep.py "
                f"--keep-package outputs/generated/{generated_name}_inspection/keep_package.zip "
                "--palette data/processed/sprites/palette.json "
                "--dataset-name sprites_generated_keep_170k",
                "python scripts/check_data.py --dataset sprites_generated_keep_170k",
            ],
        )

    if action == "prepare-sprites-mixed":
        return (
            "cpu",
            [
                "python scripts/mix_processed_datasets.py "
                "--dataset-name sprites_mixed_real_generated "
                "--sources sprites sprites_generated_keep_170k",
                "python scripts/check_data.py --dataset sprites_mixed_real_generated",
            ],
        )

    if action == "prepare-sprites-mixed-opengameart":
        return (
            "cpu",
            [
                "python scripts/mix_processed_datasets.py "
                "--dataset-name sprites_mixed_real_generated_opengameart "
                "--sources sprites sprites_generated_keep_170k opengameart",
                "python scripts/check_data.py --dataset sprites_mixed_real_generated_opengameart",
            ],
        )

    if action == "export-vqvae-sprites":
        vq_checkpoint = checkpoint or "checkpoints/vqvae_sprites_v0_full/best.ckpt"
        return (
            "gpu",
            [
                "python scripts/export_vq_tokens.py "
                f"--checkpoint {_quote(vq_checkpoint)} "
                "--source-dir data/processed/sprites "
                "--output-dir data/processed/sprites_vqvae16 "
                "--batch-size 512 "
                "--force",
                "python scripts/check_vq_tokens.py --dataset sprites_vqvae16",
            ],
        )

    if action == "prepare-patch-vq-sprites":
        return (
            "cpu",
            [
                "python scripts/train_patch_vq_tokenizer.py "
                "--source-dir data/processed/sprites "
                "--output-dir data/processed/sprites_patchvq16 "
                "--patch-size 2 "
                "--num-codes 512 "
                "--force",
                "python scripts/check_vq_tokens.py --dataset sprites_patchvq16",
            ],
        )

    if action == "smoke":
        return ("cpu", ["python scripts/smoke.py"])

    if action in VQVAE_CONFIGS:
        train_cmd = f"python scripts/train_vqvae.py --config {_quote(VQVAE_CONFIGS[action])}"
        if resume:
            train_cmd += " --resume"
        return ("gpu", [train_cmd])

    if action in HMAR_CONFIGS:
        train_cmd = f"python scripts/train_hmar.py --config {_quote(HMAR_CONFIGS[action])}"
        if resume:
            train_cmd += " --resume"
        return ("gpu", [train_cmd])

    if action in FLAT_CONFIGS:
        train_cmd = f"python scripts/train_flat_baseline.py --config {_quote(FLAT_CONFIGS[action])}"
        if resume:
            train_cmd += " --resume"
        return ("gpu", [train_cmd])

    if action in TRAIN_CONFIGS:
        train_cmd = f"python scripts/train_var.py --config {_quote(TRAIN_CONFIGS[action])}"
        if resume:
            train_cmd += " --resume"
        return ("gpu", [train_cmd])

    if action == "train":
        train_cmd = f"python scripts/train_var.py --config {_quote(config)}"
        if resume:
            train_cmd += " --resume"
        return ("gpu", [train_cmd])

    if action == "train-ladder":
        return (
            "gpu",
            [
                "python scripts/train_var.py --config configs/train/overfit32.yaml",
                "python scripts/train_var.py --config configs/train/debug1k.yaml",
                "python scripts/train_var.py --config configs/train/v0_full.yaml",
            ],
        )

    if action == "train-flat-ar-ladder":
        return (
            "gpu",
            [
                "python scripts/train_flat_baseline.py --config configs/train/sprites_flat_ar_overfit32.yaml",
                "python scripts/train_flat_baseline.py --config configs/train/sprites_flat_ar_debug1k.yaml",
                "python scripts/train_flat_baseline.py --config configs/train/sprites_flat_ar_v0_full.yaml",
            ],
        )

    if action == "train-flat-maskgit-ladder":
        return (
            "gpu",
            [
                "python scripts/train_flat_baseline.py --config configs/train/sprites_flat_maskgit_overfit32.yaml",
                "python scripts/train_flat_baseline.py --config configs/train/sprites_flat_maskgit_debug1k.yaml",
                "python scripts/train_flat_baseline.py --config configs/train/sprites_flat_maskgit_v0_full.yaml",
            ],
        )

    if action == "sample":
        if not checkpoint:
            raise ValueError("--checkpoint is required for --action sample")
        sample_cmd = (
            "python scripts/sample_var.py "
            f"--checkpoint {_quote(checkpoint)} "
            f"--config {_quote(config)} "
            f"--num-samples {int(num_samples)} "
            f"--temperature {float(temperature)} "
            f"--top-k {int(top_k)} "
            f"--output {_quote(output)}"
        )
        return ("gpu", [sample_cmd])

    if action == "sample-vq-var":
        var_checkpoint = checkpoint or "checkpoints/var_sprites_vqvae16_v0_full/best.ckpt"
        vq_top_k = 32 if int(top_k) == 8 else int(top_k)
        sample_cmd = (
            "python scripts/sample_vq_var.py "
            f"--var-checkpoint {_quote(var_checkpoint)} "
            "--vqvae-checkpoint checkpoints/vqvae_sprites_v0_full/best.ckpt "
            f"--config {_quote(config if config != 'configs/train/overfit32.yaml' else 'configs/train/sprites_vqvae16_v0_full.yaml')} "
            f"--num-samples {int(num_samples)} "
            f"--temperature {float(temperature)} "
            f"--top-k {vq_top_k} "
            f"--output {_quote(output if output != 'outputs/samples/modal_sample_grid.png' else 'outputs/samples/sprites_vqvae16_v0_full_t08_top32.png')}"
        )
        return ("gpu", [sample_cmd])

    if action == "sample-patch-vq-var":
        var_checkpoint = checkpoint or "checkpoints/var_sprites_patchvq16_v0_full/best.ckpt"
        patch_temperature = float(temperature)
        vq_top_k = 16 if int(top_k) == 8 else int(top_k)
        temperature_label = f"{patch_temperature:g}".replace(".", "")
        default_patch_output = f"outputs/samples/sprites_patchvq16_v0_full_t{temperature_label}_top{vq_top_k}.png"
        sample_cmd = (
            "python scripts/sample_patch_vq_var.py "
            f"--var-checkpoint {_quote(var_checkpoint)} "
            f"--config {_quote(config if config != 'configs/train/overfit32.yaml' else 'configs/train/sprites_patchvq16_v0_full.yaml')} "
            "--tokenizer-dir data/processed/sprites_patchvq16 "
            f"--num-samples {int(num_samples)} "
            f"--temperature {patch_temperature} "
            f"--top-k {vq_top_k} "
            f"--output {_quote(output if output != 'outputs/samples/modal_sample_grid.png' else default_patch_output)}"
        )
        return ("gpu", [sample_cmd])

    if action == "sample-hmar":
        hmar_checkpoint = checkpoint or "checkpoints/hmar_sprites_v0_full/best.ckpt"
        hmar_config = config if config != "configs/train/overfit32.yaml" else "configs/train/sprites_hmar_v0_full.yaml"
        temperature_label = f"{float(temperature):g}".replace(".", "")
        default_hmar_output = (
            f"outputs/samples/hmar_sprites_v0_full_t{temperature_label}_top{int(top_k)}_s{int(refinement_steps)}.png"
        )
        sample_cmd = (
            "python scripts/sample_hmar.py "
            f"--checkpoint {_quote(hmar_checkpoint)} "
            f"--config {_quote(hmar_config)} "
            f"--num-samples {int(num_samples)} "
            f"--refinement-steps {int(refinement_steps)} "
            f"--temperature {float(temperature)} "
            f"--top-k {int(top_k)} "
            f"--output {_quote(output if output != 'outputs/samples/modal_sample_grid.png' else default_hmar_output)}"
        )
        return ("gpu", [sample_cmd])

    if action == "eval-patch-vq-decoded":
        eval_checkpoint = checkpoint or "checkpoints/var_sprites_patchvq16_v0_full/best.ckpt"
        eval_config = config if config != "configs/train/overfit32.yaml" else "configs/train/sprites_patchvq16_v0_full.yaml"
        eval_output = output if output != "outputs/samples/modal_sample_grid.png" else "outputs/eval/sprites_patchvq16_decoded"
        eval_samples = 128 if num_samples == 16 else int(num_samples)
        eval_cmd = (
            "python scripts/evaluate_decoded_patch_vq.py "
            f"--var-checkpoint {_quote(eval_checkpoint)} "
            f"--config {_quote(eval_config)} "
            "--tokenizer-dir data/processed/sprites_patchvq16 "
            "--reference-dir data/processed/sprites "
            f"--output-dir {_quote(eval_output)} "
            f"--num-samples {eval_samples} "
            "--sample-batch-size 64 "
            "--grid-samples 64 "
            "--reference-samples 2048"
        )
        return ("gpu", [eval_cmd])

    if action == "eval-hmar-sprites":
        eval_config = config if config != "configs/train/overfit32.yaml" else "configs/train/sprites_hmar_v0_full.yaml"
        eval_checkpoint = checkpoint or "checkpoints/hmar_sprites_v0_full/best.ckpt"
        eval_output = output if output != "outputs/samples/modal_sample_grid.png" else "outputs/eval/hmar_sprites_v0_full"
        eval_samples = 128 if num_samples == 16 else int(num_samples)
        eval_cmd = (
            "python scripts/evaluate_option_a.py "
            "--model-kind hmar "
            f"--config {_quote(eval_config)} "
            f"--checkpoint {_quote(eval_checkpoint)} "
            f"--output-dir {_quote(eval_output)} "
            f"--num-samples {eval_samples} "
            f"--refinement-steps {int(refinement_steps)} "
            "--sample-batch-size 64 "
            "--grid-samples 64 "
            "--reference-samples 2048"
        )
        return ("gpu", [eval_cmd])

    if action == "eval-hmar-refinement-ablation":
        eval_config = config if config != "configs/train/overfit32.yaml" else "configs/train/sprites_hmar_v0_full.yaml"
        eval_checkpoint = checkpoint or "checkpoints/hmar_sprites_v0_full/best.ckpt"
        eval_root = output if output != "outputs/samples/modal_sample_grid.png" else "outputs/eval/hmar_sprites_refinement_ablation"
        eval_samples = 128 if num_samples == 16 else int(num_samples)
        commands = []
        for steps in (1, 2, 4, 8):
            step_output = f"{eval_root.rstrip('/')}/steps_{steps}"
            commands.append(
                "python scripts/evaluate_option_a.py "
                "--model-kind hmar "
                f"--config {_quote(eval_config)} "
                f"--checkpoint {_quote(eval_checkpoint)} "
                f"--output-dir {_quote(step_output)} "
                f"--num-samples {eval_samples} "
                f"--refinement-steps {steps} "
                "--sample-batch-size 64 "
                "--grid-samples 64 "
                "--reference-samples 2048"
            )
        return ("gpu", commands)

    if action == "eval-sprites":
        eval_config = config if config != "configs/train/overfit32.yaml" else "configs/train/sprites_v0_full.yaml"
        eval_checkpoint = checkpoint or "checkpoints/var_sprites_v0_full/best.ckpt"
        eval_output = output if output != "outputs/samples/modal_sample_grid.png" else "outputs/eval/sprites_v0_full"
        eval_samples = 128 if num_samples == 16 else int(num_samples)
        eval_cmd = (
            "python scripts/evaluate_option_a.py "
            f"--config {_quote(eval_config)} "
            f"--checkpoint {_quote(eval_checkpoint)} "
            f"--output-dir {_quote(eval_output)} "
            f"--num-samples {eval_samples} "
            "--sample-batch-size 64 "
            "--grid-samples 64 "
            "--reference-samples 2048"
        )
        return ("gpu", [eval_cmd])

    if action == "benchmark-main-hmar-known-metrics":
        benchmark_samples = 4096 if num_samples == 16 else int(num_samples)
        output_root = output if output != "outputs/samples/modal_sample_grid.png" else "outputs/eval_images"
        main_dir = f"{output_root.rstrip('/')}/pixelvar_main"
        hmar_dir = f"{output_root.rstrip('/')}/hmar_steps1"
        eval_output = "outputs/external_eval/main_vs_hmar_known_metrics"
        return (
            "gpu",
            [
                "python scripts/export_eval_images.py "
                "--config configs/train/sprites_v0_full.yaml "
                "--checkpoint checkpoints/var_sprites_v0_full/best.ckpt "
                "--generated-name pixelvar_main "
                f"--temperature {float(temperature)} "
                f"--top-k {int(top_k)} "
                f"--num-reference {benchmark_samples} "
                f"--num-generated {benchmark_samples} "
                f"--sample-batch-size 128 "
                f"--output-dir {_quote(main_dir)}",
                "python scripts/export_eval_images.py "
                "--config configs/train/sprites_hmar_v0_full.yaml "
                "--checkpoint checkpoints/hmar_sprites_v0_full/best.ckpt "
                "--model-kind hmar "
                "--generated-name hmar_steps1 "
                f"--temperature {float(temperature)} "
                f"--top-k {int(top_k)} "
                "--refinement-steps 1 "
                f"--num-reference {benchmark_samples} "
                f"--num-generated {benchmark_samples} "
                f"--sample-batch-size 128 "
                f"--output-dir {_quote(hmar_dir)}",
                "python scripts/evaluate_image_folders.py "
                f"--reference-dir {_quote(f'{main_dir}/reference')} "
                f"--generated-dir {_quote(f'pixelvar_main={main_dir}/pixelvar_main')} "
                f"--generated-dir {_quote(f'hmar_steps1={hmar_dir}/hmar_steps1')} "
                "--palette-json data/processed/sprites/palette.json "
                "--feature-space inception "
                f"--max-images {benchmark_samples} "
                "--batch-size 128 "
                "--kid-subsets 50 "
                f"--kid-subset-size {min(1000, benchmark_samples)} "
                "--msssim-pairs 1000 "
                f"--output-dir {_quote(eval_output)}",
            ],
        )

    if action == "benchmark-main-hmar-flat-known-metrics":
        benchmark_samples = 4096 if num_samples == 16 else int(num_samples)
        output_root = output if output != "outputs/samples/modal_sample_grid.png" else "outputs/eval_images"
        main_dir = f"{output_root.rstrip('/')}/pixelvar_main"
        hmar_dir = f"{output_root.rstrip('/')}/hmar_steps1"
        flat_ar_dir = f"{output_root.rstrip('/')}/flat_ar"
        flat_maskgit_dir = f"{output_root.rstrip('/')}/flat_maskgit"
        eval_output = "outputs/external_eval/main_hmar_flat_known_metrics"
        return (
            "gpu",
            [
                "python scripts/export_eval_images.py "
                "--config configs/train/sprites_v0_full.yaml "
                "--checkpoint checkpoints/var_sprites_v0_full/best.ckpt "
                "--generated-name pixelvar_main "
                f"--temperature {float(temperature)} "
                f"--top-k {int(top_k)} "
                f"--num-reference {benchmark_samples} "
                f"--num-generated {benchmark_samples} "
                "--sample-batch-size 128 "
                f"--output-dir {_quote(main_dir)}",
                "python scripts/export_eval_images.py "
                "--config configs/train/sprites_hmar_v0_full.yaml "
                "--checkpoint checkpoints/hmar_sprites_v0_full/best.ckpt "
                "--model-kind hmar "
                "--generated-name hmar_steps1 "
                f"--temperature {float(temperature)} "
                f"--top-k {int(top_k)} "
                "--refinement-steps 1 "
                f"--num-reference {benchmark_samples} "
                f"--num-generated {benchmark_samples} "
                "--sample-batch-size 128 "
                f"--output-dir {_quote(hmar_dir)}",
                "python scripts/export_eval_images.py "
                "--config configs/train/sprites_flat_ar_v0_full.yaml "
                "--checkpoint checkpoints/flat_ar_sprites_v0_full/best.ckpt "
                "--model-kind flat_ar "
                "--generated-name flat_ar "
                f"--temperature {float(temperature)} "
                f"--top-k {int(top_k)} "
                f"--num-reference {benchmark_samples} "
                f"--num-generated {benchmark_samples} "
                "--sample-batch-size 128 "
                f"--output-dir {_quote(flat_ar_dir)}",
                "python scripts/export_eval_images.py "
                "--config configs/train/sprites_flat_maskgit_v0_full.yaml "
                "--checkpoint checkpoints/flat_maskgit_sprites_v0_full/best.ckpt "
                "--model-kind flat_maskgit "
                "--generated-name flat_maskgit "
                f"--temperature {float(temperature)} "
                f"--top-k {int(top_k)} "
                "--refinement-steps 8 "
                f"--num-reference {benchmark_samples} "
                f"--num-generated {benchmark_samples} "
                "--sample-batch-size 128 "
                f"--output-dir {_quote(flat_maskgit_dir)}",
                "python scripts/evaluate_image_folders.py "
                f"--reference-dir {_quote(f'{main_dir}/reference')} "
                f"--generated-dir {_quote(f'pixelvar_main={main_dir}/pixelvar_main')} "
                f"--generated-dir {_quote(f'hmar_steps1={hmar_dir}/hmar_steps1')} "
                f"--generated-dir {_quote(f'flat_ar={flat_ar_dir}/flat_ar')} "
                f"--generated-dir {_quote(f'flat_maskgit={flat_maskgit_dir}/flat_maskgit')} "
                "--palette-json data/processed/sprites/palette.json "
                "--feature-space inception "
                f"--max-images {benchmark_samples} "
                "--batch-size 128 "
                "--kid-subsets 50 "
                f"--kid-subset-size {min(1000, benchmark_samples)} "
                "--msssim-pairs 1000 "
                f"--output-dir {_quote(eval_output)}",
            ],
        )

    if action == "audit-flat-ar-memorization":
        audit_samples = 4096 if num_samples == 16 else int(num_samples)
        generated_dir = "outputs/eval_images/flat_ar/flat_ar"
        audit_output = "outputs/memorization_audit/flat_ar"
        commands = []
        commands.append(
            "python scripts/export_eval_images.py "
            "--config configs/train/sprites_flat_ar_v0_full.yaml "
            "--checkpoint checkpoints/flat_ar_sprites_v0_full/best.ckpt "
            "--model-kind flat_ar "
            "--generated-name flat_ar "
            f"--temperature {float(temperature)} "
            f"--top-k {int(top_k)} "
            f"--num-reference {audit_samples} "
            f"--num-generated {audit_samples} "
            "--sample-batch-size 128 "
            "--output-dir outputs/eval_images/flat_ar"
        )
        commands.append(
            "python scripts/audit_memorization.py "
            "--processed-dir data/processed/sprites "
            f"--generated-dir {_quote(generated_dir)} "
            f"--output-dir {_quote(audit_output)} "
            f"--max-generated {audit_samples} "
            "--device cuda "
            "--gen-batch-size 128 "
            "--ref-batch-size 4096 "
            "--nearest-sheet-pairs 32"
        )
        return ("gpu", commands)

    if action == "audit-main-hmar-memorization":
        audit_samples = 4096 if num_samples == 16 else int(num_samples)
        return (
            "gpu",
            [
                "python scripts/export_eval_images.py "
                "--config configs/train/sprites_v0_full.yaml "
                "--checkpoint checkpoints/var_sprites_v0_full/best.ckpt "
                "--generated-name pixelvar_main "
                f"--temperature {float(temperature)} "
                f"--top-k {int(top_k)} "
                f"--num-reference {audit_samples} "
                f"--num-generated {audit_samples} "
                "--sample-batch-size 128 "
                "--output-dir outputs/eval_images/pixelvar_main",
                "python scripts/audit_memorization.py "
                "--processed-dir data/processed/sprites "
                "--generated-dir outputs/eval_images/pixelvar_main/pixelvar_main "
                "--output-dir outputs/memorization_audit/pixelvar_main "
                f"--max-generated {audit_samples} "
                "--device cuda "
                "--gen-batch-size 128 "
                "--ref-batch-size 4096 "
                "--nearest-sheet-pairs 32",
                "python scripts/export_eval_images.py "
                "--config configs/train/sprites_hmar_v0_full.yaml "
                "--checkpoint checkpoints/hmar_sprites_v0_full/best.ckpt "
                "--model-kind hmar "
                "--generated-name hmar_steps1 "
                f"--temperature {float(temperature)} "
                f"--top-k {int(top_k)} "
                "--refinement-steps 1 "
                f"--num-reference {audit_samples} "
                f"--num-generated {audit_samples} "
                "--sample-batch-size 128 "
                "--output-dir outputs/eval_images/hmar_steps1",
                "python scripts/audit_memorization.py "
                "--processed-dir data/processed/sprites "
                "--generated-dir outputs/eval_images/hmar_steps1/hmar_steps1 "
                "--output-dir outputs/memorization_audit/hmar_steps1 "
                f"--max-generated {audit_samples} "
                "--device cuda "
                "--gen-batch-size 128 "
                "--ref-batch-size 4096 "
                "--nearest-sheet-pairs 32",
            ],
        )

    if action == "build-four-way-sample-sheet":
        return (
            "cpu",
            [
                "python scripts/build_sample_sheet_from_folders.py "
                "--folder 'PixelVAR main=outputs/eval_images/pixelvar_main/pixelvar_main' "
                "--folder 'HMAR step 1=outputs/eval_images/hmar_steps1/hmar_steps1' "
                "--folder 'Flat AR memorizing=outputs/eval_images/flat_ar/flat_ar' "
                "--folder 'Flat MaskGIT failed=outputs/eval_images/flat_maskgit/flat_maskgit' "
                "--output outputs/final/four_way_sample_sheet.png "
                "--samples-per-method 16 "
                "--columns 16 "
                "--scale 4 "
                "--seed 42 "
                "--title 'Four-way generated sample comparison' "
                "--note 'Rows are independent random samples from the 4096-image benchmark folders. Flat AR is labeled memorizing because the audit found 3162 train, 365 validation, and 333 test exact matches.'"
            ],
        )

    if action in {
        "prepare-sd-pixl-baseline",
        "run-sd-pixl-smoke",
        "run-sd-pixl-batch",
        "normalize-sd-pixl-baseline",
        "build-sd-pixl-sample-sheet",
    }:
        def sd_pixl_prepare_cmd(config_name: str, seed: int, prompt_index: int) -> str:
            prompt_arg = (
                f"--prompt {_quote(sd_pixl_prompt)}"
                if sd_pixl_prompt
                else f"--prompt-file configs/external/sd_pixl_prompts.txt --prompt-index {int(prompt_index)}"
            )
            return (
                "python scripts/prepare_sd_pixl_baseline.py "
                "--repo-dir outputs/external_baselines/sd_pixl/repo "
                "--method-dir outputs/external_baselines/sd_pixl "
                "--palette-json data/processed/sprites/palette.json "
                f"--config-name {_quote(config_name)} "
                f"{prompt_arg} "
                f"--seed {int(seed)} "
                "--image-size 32 "
                f"--steps {int(sd_pixl_steps)} "
                "--save-steps 50 "
                f"--model-id {_quote(sd_pixl_model_id)} "
                "--num-references 1 "
                "--num-inference-steps 20 "
                "--controlnet-models canny_small "
                "--controlnet-scale 0.25"
            )

        def sd_pixl_run_cmd(config_name: str) -> str:
            return (
                "cd outputs/external_baselines/sd_pixl/repo && "
                "accelerate launch main.py "
                f"-c {_quote(config_name)} "
                "--download "
                "-respath ../workdir"
            )

        prepare_cmd = sd_pixl_prepare_cmd("pixelvar_sd_pixl.yaml", seed=0, prompt_index=sd_pixl_prompt_index)
        normalize_cmd = (
            "python scripts/normalize_external_images.py "
            "--input-dir outputs/external_baselines/sd_pixl/workdir "
            "--pattern final_argmax.png "
            "--output-dir outputs/external_baselines/sd_pixl/png32 "
            "--palette-json data/processed/sprites/palette.json "
            "--image-size 32 "
            "--prefix sd_pixl"
        )
        sheet_cmd = (
            "python scripts/build_sample_sheet_from_folders.py "
            "--folder 'SD-piXL=outputs/external_baselines/sd_pixl/png32' "
            "--output outputs/final/sd_pixl_sample_sheet.png "
            "--samples-per-method 16 "
            "--columns 16 "
            "--scale 4 "
            "--seed 42 "
            "--title 'SD-piXL external baseline samples' "
            "--note 'SD-piXL is prompt-conditioned score-distillation optimization, not an unconditional dataset-trained sampler. These images are normalized to the PixelVAR 32x32 palette protocol.'"
        )
        if action == "prepare-sd-pixl-baseline":
            return ("cpu", [prepare_cmd])
        if action == "normalize-sd-pixl-baseline":
            return ("cpu", [normalize_cmd])
        if action == "build-sd-pixl-sample-sheet":
            return ("cpu", [sheet_cmd])

        if action == "run-sd-pixl-batch":
            batch_count = 4 if num_samples == 16 else int(num_samples)
            commands = []
            for idx in range(batch_count):
                config_name = f"pixelvar_sd_pixl_{idx:03d}.yaml"
                prompt_index = int(sd_pixl_prompt_index) + idx
                commands.append(sd_pixl_prepare_cmd(config_name, seed=idx, prompt_index=prompt_index))
                commands.append(sd_pixl_run_cmd(config_name))
            commands.extend([normalize_cmd, sheet_cmd])
            return ("sd_pixl_gpu", commands)

        return ("sd_pixl_gpu", [prepare_cmd, sd_pixl_run_cmd("pixelvar_sd_pixl.yaml"), normalize_cmd, sheet_cmd])

    if action in {
        "run-practical-diffusion-smoke",
        "run-practical-diffusion-batch",
        "normalize-practical-diffusion-baseline",
        "build-practical-diffusion-sample-sheet",
    }:
        raw_dir = "outputs/external_baselines/practical_diffusion/raw"
        png32_dir = "outputs/external_baselines/practical_diffusion/png32"
        sheet_path = "outputs/final/practical_diffusion_sample_sheet.png"
        sample_count = int(num_samples)
        if num_samples == 16:
            sample_count = 4 if action == "run-practical-diffusion-smoke" else 64

        prompt_arg = (
            f"--prompt {_quote(diffusion_prompt)}"
            if diffusion_prompt
            else f"--prompt-file configs/external/practical_diffusion_prompts.txt --prompt-index {int(diffusion_prompt_index)}"
        )
        lora_arg = ""
        if diffusion_lora_id:
            lora_arg = f" --lora-id {_quote(diffusion_lora_id)} --lora-scale {float(diffusion_lora_scale)}"
            if diffusion_lora_weight_name:
                lora_arg += f" --lora-weight-name {_quote(diffusion_lora_weight_name)}"
        diffusion_note = (
            f"{diffusion_model_id} + LoRA {diffusion_lora_id}"
            if diffusion_lora_id
            else f"{diffusion_model_id} without LoRA"
        )

        generate_cmd = (
            "python scripts/run_practical_diffusion_baseline.py "
            f"--model-id {_quote(diffusion_model_id)} "
            f"--output-dir {_quote(raw_dir)} "
            f"{prompt_arg} "
            f"--num-images {sample_count} "
            f"--seed {int(diffusion_seed)} "
            f"--height {int(diffusion_height)} "
            f"--width {int(diffusion_width)} "
            f"--steps {int(diffusion_steps)} "
            f"--guidance-scale {float(diffusion_guidance_scale)} "
            f"--dtype {_quote(diffusion_dtype)} "
            f"--negative-prompt {_quote(diffusion_negative_prompt)}"
            f"{lora_arg}"
        )
        normalize_cmd = (
            "python scripts/normalize_external_images.py "
            f"--input-dir {_quote(raw_dir)} "
            f"--output-dir {_quote(png32_dir)} "
            "--palette-json data/processed/sprites/palette.json "
            "--image-size 32 "
            "--prefix practical_diffusion "
            "--transparent-from-corners "
            "--transparent-tolerance 18.0"
        )
        sheet_cmd = (
            "python scripts/build_sample_sheet_from_folders.py "
            f"--folder {_quote(f'Practical diffusion={png32_dir}')} "
            f"--output {_quote(sheet_path)} "
            "--samples-per-method 16 "
            "--columns 16 "
            "--scale 4 "
            "--seed 42 "
            "--title 'Practical diffusion external baseline samples' "
            f"--note {_quote(f'{diffusion_note}; generated raw at {int(diffusion_width)}x{int(diffusion_height)}, then normalized to the PixelVAR 32x32 palette protocol.')}"
        )

        if action == "normalize-practical-diffusion-baseline":
            return ("cpu", [normalize_cmd])
        if action == "build-practical-diffusion-sample-sheet":
            return ("cpu", [sheet_cmd])
        return ("sd_pixl_gpu", [generate_cmd, normalize_cmd, sheet_cmd])

    if action == "generate-sprites-selected":
        gen_config = config if config != "configs/train/overfit32.yaml" else "configs/train/sprites_v0_full.yaml"
        gen_checkpoint = checkpoint or "checkpoints/var_sprites_v0_full/best.ckpt"
        gen_samples = 8192 if num_samples == 16 else int(num_samples)
        gen_output = (
            output
            if output != "outputs/samples/modal_sample_grid.png"
            else f"outputs/generated/sprites_v0_full_t08_top8_{gen_samples}"
        )
        gen_cmd = (
            "python scripts/generate_option_a_set.py "
            f"--config {_quote(gen_config)} "
            f"--checkpoint {_quote(gen_checkpoint)} "
            f"--output-dir {_quote(gen_output)} "
            f"--num-samples {gen_samples} "
            "--batch-size 64 "
            f"--temperature {float(temperature)} "
            f"--top-k {int(top_k)} "
            "--grid-samples 64 "
            "--num-grids 16 "
            "--max-image-files 2048"
        )
        return ("gpu", [gen_cmd])

    if action == "cmd-cpu":
        if not cmd:
            raise ValueError("--cmd is required for --action cmd-cpu")
        return ("cpu", [cmd])

    if action == "cmd-gpu":
        if not cmd:
            raise ValueError("--cmd is required for --action cmd-gpu")
        return ("gpu", [cmd])

    raise ValueError(
        "Unknown action. Use one of: cuda-check, prepare-pokemon, check-pokemon, "
        "download-sprites-public, prepare-sprites, prepare-raw-sprites, prepare-opengameart, "
        "prepare-opengameart-public, prepare-sprites-generated-keep, prepare-sprites-mixed, "
        "prepare-sprites-mixed-opengameart, export-vqvae-sprites, prepare-patch-vq-sprites, smoke, "
        "train-overfit32, train-debug1k, train-v0-full, train-sprites-overfit32, "
        "train-sprites-debug1k, train-sprites-v0-full, train-sprites-generated-keep-overfit32, "
        "train-sprites-generated-keep-debug1k, train-sprites-generated-keep-v0-full, "
        "train-sprites-mixed-overfit32, train-sprites-mixed-debug1k, train-sprites-mixed-v0-full, "
        "train-sprites-mixed-oga-overfit32, train-sprites-mixed-oga-debug1k, "
        "train-sprites-mixed-oga-v0-full, "
        "train-vqvae-sprites-overfit32, train-vqvae-sprites-debug1k, train-vqvae-sprites-v0-full, "
        "train-sprites-vqvae16-overfit32, train-sprites-vqvae16-debug1k, train-sprites-vqvae16-v0-full, "
        "train-sprites-patchvq16-overfit32, train-sprites-patchvq16-debug1k, train-sprites-patchvq16-v0-full, "
        "train-sprites-hmar-overfit32, train-sprites-hmar-debug1k, train-sprites-hmar-v0-full, "
        "train-sprites-flat-ar-overfit32, train-sprites-flat-ar-debug1k, train-sprites-flat-ar-v0-full, "
        "train-sprites-flat-maskgit-overfit32, train-sprites-flat-maskgit-debug1k, train-sprites-flat-maskgit-v0-full, "
        "train, train-ladder, train-flat-ar-ladder, train-flat-maskgit-ladder, "
        "sample, sample-vq-var, sample-patch-vq-var, sample-hmar, "
        "eval-patch-vq-decoded, eval-hmar-sprites, eval-hmar-refinement-ablation, eval-sprites, "
        "benchmark-main-hmar-known-metrics, benchmark-main-hmar-flat-known-metrics, audit-flat-ar-memorization, "
        "audit-main-hmar-memorization, "
        "build-four-way-sample-sheet, prepare-sd-pixl-baseline, run-sd-pixl-smoke, run-sd-pixl-batch, "
        "normalize-sd-pixl-baseline, build-sd-pixl-sample-sheet, "
        "run-practical-diffusion-smoke, run-practical-diffusion-batch, "
        "normalize-practical-diffusion-baseline, build-practical-diffusion-sample-sheet, "
        "generate-sprites-selected, cmd-cpu, cmd-gpu."
    )


@app.local_entrypoint()
def main(
    action: str = "cuda-check",
    config: str = "configs/train/overfit32.yaml",
    cmd: str = "",
    resume: bool = False,
    checkpoint: str = "",
    num_samples: int = 16,
    temperature: float = 0.8,
    top_k: int = 8,
    refinement_steps: int = 4,
    output: str = "outputs/samples/modal_sample_grid.png",
    transparent_color: str = "",
    sheet_tile_size: int = 32,
    sprites_frames_per_group: int = 178,
    sd_pixl_steps: int = 250,
    sd_pixl_model_id: str = "ssd1b",
    sd_pixl_prompt_index: int = 0,
    sd_pixl_prompt: str = "",
    diffusion_model_id: str = "segmind/SSD-1B",
    diffusion_lora_id: str = "",
    diffusion_lora_weight_name: str = "",
    diffusion_lora_scale: float = 0.8,
    diffusion_prompt_index: int = 0,
    diffusion_prompt: str = "",
    diffusion_steps: int = 25,
    diffusion_guidance_scale: float = 7.0,
    diffusion_height: int = 512,
    diffusion_width: int = 512,
    diffusion_seed: int = 0,
    diffusion_dtype: str = "float16",
    diffusion_negative_prompt: str = (
        "realistic photo, 3d render, blurry, smooth shading, detailed background, "
        "text, watermark, logo, cropped, multiple characters, multiple sprites, "
        "sprite sheet, character sheet, grid, lineup, duplicate, variations, portrait close-up"
    ),
) -> None:
    runner, commands = commands_for_action(
        action=action,
        config=config,
        cmd=cmd,
        resume=resume,
        checkpoint=checkpoint,
        num_samples=num_samples,
        temperature=temperature,
        top_k=top_k,
        refinement_steps=refinement_steps,
        output=output,
        transparent_color=transparent_color,
        sheet_tile_size=sheet_tile_size,
        sprites_frames_per_group=sprites_frames_per_group,
        sd_pixl_steps=sd_pixl_steps,
        sd_pixl_model_id=sd_pixl_model_id,
        sd_pixl_prompt_index=sd_pixl_prompt_index,
        sd_pixl_prompt=sd_pixl_prompt,
        diffusion_model_id=diffusion_model_id,
        diffusion_lora_id=diffusion_lora_id,
        diffusion_lora_weight_name=diffusion_lora_weight_name,
        diffusion_lora_scale=diffusion_lora_scale,
        diffusion_prompt_index=diffusion_prompt_index,
        diffusion_prompt=diffusion_prompt,
        diffusion_steps=diffusion_steps,
        diffusion_guidance_scale=diffusion_guidance_scale,
        diffusion_height=diffusion_height,
        diffusion_width=diffusion_width,
        diffusion_seed=diffusion_seed,
        diffusion_dtype=diffusion_dtype,
        diffusion_negative_prompt=diffusion_negative_prompt,
    )
    print(f"[modal] action={action} runner={runner}")
    if runner == "gpu":
        run_b200.remote(commands)
    elif runner == "sd_pixl_gpu":
        run_sd_pixl_b200.remote(commands)
    else:
        run_cpu.remote(commands)
