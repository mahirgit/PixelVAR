# Memorization Audit

Generated folder: `outputs/eval_images/pixelvar_main/pixelvar_main`
Processed dataset: `data/processed/sprites`
Generated images: `4096`
Off-palette opaque pixels remapped: `0`

## Exact Matches

| Split | Generated matches | Match rate | Unique hashes |
| --- | ---: | ---: | ---: |
| train | 157 | 0.038330 | 153 |
| val | 13 | 0.003174 | 13 |
| test | 11 | 0.002686 | 11 |

## Nearest Neighbors

| Split | Mean Hamming | P05 Hamming | Min Hamming | Exact | <=1% | <=2% | <=5% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 0.030085 | 0.000977 | 0.000000 | 157 | 1023 | 1784 | 3389 |
| val | 0.052147 | 0.014648 | 0.000000 | 13 | 102 | 392 | 2115 |
| test | 0.050875 | 0.013672 | 0.000000 | 11 | 121 | 416 | 2235 |

## Dataset Duplicates

- Cross-split duplicate hashes: `0`
- Cross-split duplicate samples: `0`

| Split | Samples | Unique | Duplicate groups | Duplicate samples | Max group |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 74664 | 64233 | 9333 | 10431 | 3 |
| val | 9360 | 8061 | 1170 | 1299 | 3 |
| test | 9288 | 7968 | 1161 | 1320 | 3 |

## Generated Duplicates

- Unique generated images: `4092`
- Duplicate groups: `4`
- Duplicate generated samples: `4`
- Max generated duplicate group size: `2`

## Interpretation

There are exact validation matches. Review the matching images before claiming this model is fully non-memorizing.
