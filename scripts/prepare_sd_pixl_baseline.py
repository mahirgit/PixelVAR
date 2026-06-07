#!/usr/bin/env python3
"""Prepare SD-piXL for PixelVAR external-baseline runs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml

from export_palette_hex import load_colors, write_hex


DEFAULT_REPO_URL = "https://github.com/AlexandreBinninger/SD-piXL.git"


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt
    if args.prompt_file:
        prompts = [
            line.strip()
            for line in args.prompt_file.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not prompts:
            raise ValueError(f"No prompts found in {args.prompt_file}")
        return prompts[args.prompt_index % len(prompts)]
    raise ValueError("Provide either --prompt or --prompt-file")


def ensure_repo(repo_dir: Path, repo_url: str) -> None:
    if (repo_dir / "main.py").exists() and (repo_dir / "config" / "config.yaml").exists():
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(repo_dir)], check=True)


def write_config(
    repo_dir: Path,
    config_name: str,
    method_dir: Path,
    palette_hex: Path,
    prompt: str,
    args: argparse.Namespace,
) -> Path:
    base_config = repo_dir / "config" / "config.yaml"
    config = yaml.safe_load(base_config.read_text())

    config["seed"] = int(args.seed)
    config["image"] = args.input_image.as_posix() if args.input_image else None
    config["automatic_caption"] = False
    config["saving_resize"] = int(args.image_size)
    config["prompt"] = prompt
    config["negative_prompt"] = args.negative_prompt

    config["generator"]["image_H"] = int(args.image_size)
    config["generator"]["image_W"] = int(args.image_size)
    config["generator"]["palette"] = palette_hex.resolve().as_posix()
    config["generator"]["initialize_renderer"] = bool(args.initialize_renderer)
    config["generator"]["initialization_method"] = args.initialization_method
    config["generator"]["smooth_softmax"] = bool(args.smooth_softmax)
    config["generator"]["gumbel"] = bool(args.gumbel)

    config["training"]["steps"] = int(args.steps)
    config["training"]["save_steps"] = int(args.save_steps)
    config["training"]["lr_warmup_steps"] = min(int(config["training"].get("lr_warmup_steps", 250)), int(args.steps))
    config["training"]["augmentation"]["random_tau"] = bool(args.random_tau)

    config["diffusion"]["model_id"] = args.model_id
    config["diffusion"]["vae_id"] = args.vae_id
    config["diffusion"]["lora_path"] = args.lora_path if args.lora_path else None
    config["diffusion"]["lora_scale"] = float(args.lora_scale)
    config["diffusion"]["num_references"] = int(args.num_references)
    config["diffusion"]["num_inference_steps"] = int(args.num_inference_steps)
    config["diffusion"]["guidance_scale"] = float(args.reference_guidance_scale)

    control_models = [item.strip() for item in args.controlnet_models.split(",") if item.strip()]
    if not control_models:
        raise ValueError("SD-piXL expects at least one ControlNet model id")
    config["controlnet"]["models_id"] = control_models
    config["controlnet"]["controlnet_conditioning_scale"] = [float(args.controlnet_scale)] * len(control_models)
    config["controlnet"]["use_controlnet"] = bool(args.use_controlnet)

    config["sd"]["guidance_scale"] = float(args.sd_guidance_scale)
    config["sd"]["t_max"] = float(args.t_max)
    config["sd"]["t_bound_max"] = float(args.t_bound_max)

    config_path = repo_dir / "config" / config_name
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    manifest = {
        "repo_url": args.repo_url,
        "repo_dir": repo_dir.as_posix(),
        "method_dir": method_dir.as_posix(),
        "config": config_path.as_posix(),
        "config_name": config_name,
        "palette_hex": palette_hex.as_posix(),
        "prompt": prompt,
        "image_size": int(args.image_size),
        "steps": int(args.steps),
        "model_id": args.model_id,
        "input_image": args.input_image.as_posix() if args.input_image else None,
        "raw_workdir": (method_dir / "workdir").as_posix(),
        "normalized_dir": (method_dir / "png32").as_posix(),
    }
    (method_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return config_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SD-piXL external baseline config")
    parser.add_argument("--repo-dir", type=Path, default=Path("outputs/external_baselines/sd_pixl/repo"))
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--method-dir", type=Path, default=Path("outputs/external_baselines/sd_pixl"))
    parser.add_argument("--palette-json", type=Path, default=Path("data/processed/sprites/palette.json"))
    parser.add_argument("--config-name", default="pixelvar_sd_pixl.yaml")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--input-image", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--model-id", choices=["sdxl", "ssd1b"], default="ssd1b")
    parser.add_argument("--vae-id", default="taesdxl")
    parser.add_argument("--lora-path", default="")
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--num-references", type=int, default=1)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--reference-guidance-scale", type=float, default=7.5)
    parser.add_argument("--sd-guidance-scale", type=float, default=30.0)
    parser.add_argument("--t-max", type=float, default=0.90)
    parser.add_argument("--t-bound-max", type=float, default=0.70)
    parser.add_argument("--controlnet-models", default="canny_small")
    parser.add_argument("--controlnet-scale", type=float, default=0.25)
    parser.add_argument("--no-controlnet", dest="use_controlnet", action="store_false")
    parser.add_argument("--no-initialize-renderer", dest="initialize_renderer", action="store_false")
    parser.add_argument("--initialization-method", default="palette-bilinear")
    parser.add_argument("--hard-renderer", dest="smooth_softmax", action="store_false")
    parser.add_argument("--no-gumbel", dest="gumbel", action="store_false")
    parser.add_argument("--fixed-tau", dest="random_tau", action="store_false")
    parser.set_defaults(use_controlnet=True, initialize_renderer=True, smooth_softmax=True, gumbel=True, random_tau=True)
    args = parser.parse_args()

    args.method_dir.mkdir(parents=True, exist_ok=True)
    ensure_repo(args.repo_dir, args.repo_url)

    colors = load_colors(args.palette_json)
    palette_hex = args.method_dir / "pixelvar_palette.hex"
    write_hex(colors, palette_hex)

    prompt = read_prompt(args)
    config_path = write_config(
        repo_dir=args.repo_dir,
        config_name=args.config_name,
        method_dir=args.method_dir,
        palette_hex=palette_hex,
        prompt=prompt,
        args=args,
    )
    print(f"Wrote SD-piXL config to {config_path}")
    print(f"Prompt: {prompt}")


if __name__ == "__main__":
    main()
