# External Image-Folder Evaluation

Reference folder: `outputs/eval_images/pixelvar_main/reference`
Feature space: `inception`

Lower is better for FID, KID, mean MS-SSIM, and pixel nearest-neighbor distance.
Higher is better for precision, recall, density, coverage, and palette consistency.

| Method | Images | FID | KID mean | Precision | Recall | Density | Coverage | MS-SSIM | Palette consistency | Opaque ratio | Edge density |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pixelvar_main | 64 | 87.8128 | 0.007124 | 0.8125 | 0.9375 | 0.6375 | 0.9219 | 0.8535 | 1.0000 | 0.2259 | 0.1799 |
| practical_diffusion | 64 | 196.9628 | 0.092097 | 0.0312 | 0.6719 | 0.0094 | 0.0469 | 0.3935 | 1.0000 | 0.3520 | 0.2564 |
