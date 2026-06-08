#!/usr/bin/env python3
"""Generate raw images for a practical SDXL-family diffusion baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_NEGATIVE_PROMPT = (
    "realistic photo, 3d render, blurry, smooth shading, detailed background, "
    "text, watermark, logo, cropped, multiple characters, multiple sprites, "
    "sprite sheet, character sheet, grid, lineup, duplicate, variations, portrait close-up"
)


def read_prompts(path: Path) -> list[str]:
    prompts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            prompts.append(line)
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def prompts_for_samples(prompt: str, prompt_file: Path | None, prompt_index: int, num_images: int) -> list[str]:
    if num_images <= 0:
        raise ValueError("--num-images must be positive")
    if prompt:
        return [prompt] * num_images
    if prompt_file is None:
        raise ValueError("Either --prompt or --prompt-file is required")

    prompts = read_prompts(prompt_file)
    start = int(prompt_index) % len(prompts)
    return [prompts[(start + offset) % len(prompts)] for offset in range(num_images)]


def torch_dtype(name: str):
    import torch

    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def load_pipeline(args):
    import torch
    from diffusers import AutoPipelineForText2Image

    kwargs = {
        "torch_dtype": torch_dtype(args.dtype),
        "use_safetensors": True,
    }
    if args.variant:
        kwargs["variant"] = args.variant

    pipe = AutoPipelineForText2Image.from_pretrained(args.model_id, **kwargs)
    if args.lora_id:
        lora_kwargs = {"adapter_name": "pixel_art"}
        if args.lora_weight_name:
            lora_kwargs["weight_name"] = args.lora_weight_name
        pipe.load_lora_weights(args.lora_id, **lora_kwargs)
        if hasattr(pipe, "set_adapters"):
            pipe.set_adapters(["pixel_art"], adapter_weights=[float(args.lora_scale)])

    pipe = pipe.to(args.device)
    pipe.set_progress_bar_config(disable=False)
    if args.attention_slicing and hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if args.cpu_offload and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    return pipe, torch


def save_manifest(args, prompts: list[str], rows: list[dict[str, object]]) -> None:
    manifest = {
        "method": args.method_name,
        "model_id": args.model_id,
        "lora_id": args.lora_id or None,
        "lora_weight_name": args.lora_weight_name or None,
        "lora_scale": float(args.lora_scale),
        "seed": int(args.seed),
        "num_images": len(rows),
        "height": int(args.height),
        "width": int(args.width),
        "steps": int(args.steps),
        "guidance_scale": float(args.guidance_scale),
        "negative_prompt": args.negative_prompt,
        "prompts": prompts,
        "images": rows,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a practical SDXL-family text-to-image baseline")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/external_baselines/practical_diffusion/raw"))
    parser.add_argument("--method-name", default="practical_diffusion")
    parser.add_argument("--filename-prefix", default="")
    parser.add_argument("--prompt-file", type=Path, default=Path("configs/external/practical_diffusion_prompts.txt"))
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--model-id", default="segmind/SSD-1B")
    parser.add_argument("--variant", default="")
    parser.add_argument("--lora-id", default="")
    parser.add_argument("--lora-weight-name", default="")
    parser.add_argument("--lora-scale", type=float, default=0.8)
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--attention-slicing", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    args = parser.parse_args()

    prompts = prompts_for_samples(
        prompt=args.prompt,
        prompt_file=args.prompt_file,
        prompt_index=args.prompt_index,
        num_images=args.num_images,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in args.output_dir.glob("*.png"):
        path.unlink()

    pipe, torch = load_pipeline(args)
    cross_attention_kwargs = {"scale": float(args.lora_scale)} if args.lora_id else None

    rows = []
    filename_prefix = args.filename_prefix or args.method_name
    for idx, prompt in enumerate(prompts):
        generator = torch.Generator(device=args.device).manual_seed(int(args.seed) + idx)
        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                num_inference_steps=int(args.steps),
                guidance_scale=float(args.guidance_scale),
                height=int(args.height),
                width=int(args.width),
                generator=generator,
                cross_attention_kwargs=cross_attention_kwargs,
            ).images[0]
        out_path = args.output_dir / f"{filename_prefix}_raw_{idx:06d}.png"
        image.save(out_path)
        rows.append(
            {
                "index": idx,
                "seed": int(args.seed) + idx,
                "prompt": prompt,
                "path": out_path.as_posix(),
            }
        )
        print(f"Wrote {out_path}", flush=True)

    save_manifest(args, prompts, rows)
    print(f"Wrote {len(rows)} {args.method_name} images to {args.output_dir}")


if __name__ == "__main__":
    main()
