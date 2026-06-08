import json

import numpy as np
from PIL import Image

from scripts.export_palette_hex import load_colors, write_hex
from scripts.normalize_external_images import load_palette, normalize_image
from scripts.run_practical_diffusion_baseline import prompts_for_samples, read_prompts


def test_export_palette_hex_roundtrip(tmp_path):
    palette_json = tmp_path / "palette.json"
    palette_json.write_text(
        json.dumps(
            {
                "palette_size": 2,
                "colors": [[1, 2, 3], [254, 128, 0]],
            }
        )
    )

    colors = load_colors(palette_json)
    output = tmp_path / "palette.hex"
    write_hex(colors, output)

    assert output.read_text().splitlines() == ["010203", "FE8000"]


def test_normalize_external_image_quantizes_to_palette(tmp_path):
    palette_json = tmp_path / "palette.json"
    palette_json.write_text(json.dumps({"colors": [[255, 0, 0], [0, 0, 255]]}))
    source = tmp_path / "source.png"
    Image.new("RGBA", (64, 48), (245, 12, 8, 255)).save(source)

    image = normalize_image(
        path=source,
        image_size=32,
        palette=load_palette(palette_json),
        alpha_threshold=128,
        crop_square=True,
        transparent_from_corners=False,
        transparent_tolerance=12.0,
    )

    assert image.shape == (32, 32, 4)
    assert np.all(image[:, :, :3] == np.array([255, 0, 0], dtype=np.uint8))
    assert np.all(image[:, :, 3] == 255)


def test_normalize_external_image_can_remove_corner_background(tmp_path):
    palette_json = tmp_path / "palette.json"
    palette_json.write_text(json.dumps({"colors": [[0, 0, 0]]}))
    source = tmp_path / "source.png"
    canvas = Image.new("RGBA", (64, 64), (255, 255, 255, 255))
    pixels = np.asarray(canvas, dtype=np.uint8).copy()
    pixels[20:44, 20:44, :3] = 0
    Image.fromarray(pixels, mode="RGBA").save(source)

    image = normalize_image(
        path=source,
        image_size=32,
        palette=load_palette(palette_json),
        alpha_threshold=128,
        crop_square=True,
        transparent_from_corners=True,
        transparent_tolerance=1.0,
    )

    assert image[0, 0, 3] == 0
    assert image[16, 16, 3] == 255
    assert np.all(image[16, 16, :3] == np.array([0, 0, 0], dtype=np.uint8))


def test_practical_diffusion_prompt_file_ignores_comments(tmp_path):
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text(
        "\n# comment\npixel art knight\n\npixel art mage\n",
        encoding="utf-8",
    )

    assert read_prompts(prompt_file) == ["pixel art knight", "pixel art mage"]


def test_practical_diffusion_prompts_cycle_from_index(tmp_path):
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("a\nb\nc\n", encoding="utf-8")

    prompts = prompts_for_samples(
        prompt="",
        prompt_file=prompt_file,
        prompt_index=2,
        num_images=4,
    )

    assert prompts == ["c", "a", "b", "c"]


def test_practical_diffusion_explicit_prompt_repeats():
    prompts = prompts_for_samples(
        prompt="single prompt",
        prompt_file=None,
        prompt_index=99,
        num_images=3,
    )

    assert prompts == ["single prompt", "single prompt", "single prompt"]
