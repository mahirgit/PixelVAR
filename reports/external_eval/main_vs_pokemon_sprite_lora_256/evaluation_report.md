# External Image-Folder Evaluation

Reference folder: `outputs/eval_images/pixelvar_main/reference`
Feature space: `inception`

Lower is better for FID, KID, mean MS-SSIM, and pixel nearest-neighbor distance.
Higher is better for precision, recall, density, coverage, and palette consistency.

| Method | Images | FID | KID mean | Precision | Recall | Density | Coverage | MS-SSIM | Palette consistency | Opaque ratio | Edge density |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pixelvar_main | 256 | 46.4794 | 0.007480 | 0.9102 | 0.8711 | 0.8852 | 0.8906 | 0.8368 | 1.0000 | 0.2259 | 0.1791 |
| pokemon_sprite_lora | 256 | 154.0150 | 0.136589 | 0.1172 | 0.1211 | 0.0312 | 0.0508 | 0.5421 | 1.0000 | 0.2762 | 0.1912 |
