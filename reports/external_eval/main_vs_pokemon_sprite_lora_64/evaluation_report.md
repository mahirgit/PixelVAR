# External Image-Folder Evaluation

Reference folder: `outputs/eval_images/pixelvar_main/reference`
Feature space: `inception`

Lower is better for FID, KID, mean MS-SSIM, and pixel nearest-neighbor distance.
Higher is better for precision, recall, density, coverage, and palette consistency.

| Method | Images | FID | KID mean | Precision | Recall | Density | Coverage | MS-SSIM | Palette consistency | Opaque ratio | Edge density |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pixelvar_main | 64 | 87.8128 | 0.007124 | 0.8125 | 0.9375 | 0.6375 | 0.9219 | 0.8535 | 1.0000 | 0.2259 | 0.1799 |
| pokemon_sprite_lora | 64 | 184.3101 | 0.130036 | 0.3125 | 0.2812 | 0.0844 | 0.1094 | 0.5171 | 1.0000 | 0.2768 | 0.1939 |
