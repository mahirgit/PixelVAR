# Memorization Audit

Generated folder: `outputs/eval_images/flat_ar/flat_ar`
Processed dataset: `data/processed/sprites`
Generated images: `4096`
Off-palette opaque pixels remapped: `0`

## Exact Matches

| Split | Generated matches | Match rate | Unique hashes |
| --- | ---: | ---: | ---: |
| train | 3162 | 0.771973 | 2994 |
| val | 365 | 0.089111 | 349 |
| test | 333 | 0.081299 | 314 |

## Nearest Neighbors

| Split | Mean Hamming | P05 Hamming | Min Hamming | Exact | <=1% | <=2% | <=5% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 0.002445 | 0.000000 | 0.000000 | 3162 | 3632 | 4015 | 4096 |
| val | 0.031253 | 0.000000 | 0.000000 | 365 | 633 | 1453 | 3271 |
| test | 0.031170 | 0.000000 | 0.000000 | 333 | 586 | 1305 | 3308 |

## Dataset Duplicates

- Cross-split duplicate hashes: `0`
- Cross-split duplicate samples: `0`

| Split | Samples | Unique | Duplicate groups | Duplicate samples | Max group |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 74664 | 64233 | 9333 | 10431 | 3 |
| val | 9360 | 8061 | 1170 | 1299 | 3 |
| test | 9288 | 7968 | 1161 | 1320 | 3 |

## Generated Duplicates

- Unique generated images: `3890`
- Duplicate groups: `196`
- Duplicate generated samples: `206`
- Max generated duplicate group size: `3`

## Interpretation

There are exact validation matches. Treat the flat AR metric win as suspicious until the matching images are reviewed.
