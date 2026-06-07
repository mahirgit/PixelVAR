# External Image-Folder Evaluation

Reference folder: `outputs/eval_images/pixelvar_main/reference`
Feature space: `inception`

Lower is better for FID, KID, mean MS-SSIM, and pixel nearest-neighbor distance.
Higher is better for precision, recall, density, coverage, and palette consistency.

| Method | Images | FID | KID mean | Precision | Recall | Density | Coverage | MS-SSIM | Palette consistency | Opaque ratio | Edge density |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pixelvar_main | 4096 | 13.2516 | 0.006010 | 0.6406 | 0.9292 | 0.4093 | 0.6050 | 0.8323 | 1.0000 | 0.2276 | 0.1802 |
| hmar_steps1 | 4096 | 12.6555 | 0.005209 | 0.6494 | 0.9314 | 0.4136 | 0.6165 | 0.8309 | 1.0000 | 0.2269 | 0.1809 |
| flat_ar | 4096 | 9.3558 | 0.002975 | 0.9172 | 0.9541 | 0.7654 | 0.8518 | 0.8281 | 1.0000 | 0.2321 | 0.1790 |
| flat_maskgit | 4096 | 67.3908 | 0.061825 | 0.1545 | 0.0317 | 0.0455 | 0.0454 | 0.9426 | 1.0000 | 0.1942 | 0.1631 |
