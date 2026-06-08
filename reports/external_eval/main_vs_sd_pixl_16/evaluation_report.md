# External Image-Folder Evaluation

Reference folder: `outputs/eval_images/pixelvar_main/reference`
Feature space: `inception`

Lower is better for FID, KID, mean MS-SSIM, and pixel nearest-neighbor distance.
Higher is better for precision, recall, density, coverage, and palette consistency.

| Method | Images | FID | KID mean | Precision | Recall | Density | Coverage | MS-SSIM | Palette consistency | Opaque ratio | Edge density |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pixelvar_main | 16 | 151.0927 | 0.014713 | 1.0000 | 0.9375 | 1.3500 | 1.0000 | 0.8247 | 1.0000 | 0.2319 | 0.1888 |
| sd_pixl | 16 | 497.7065 | 0.517647 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.4926 | 1.0000 | 0.6720 | 0.4007 |
