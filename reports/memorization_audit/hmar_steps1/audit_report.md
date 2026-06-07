# Memorization Audit

Generated folder: `outputs/eval_images/hmar_steps1/hmar_steps1`
Processed dataset: `data/processed/sprites`
Generated images: `4096`
Off-palette opaque pixels remapped: `0`

## Exact Matches

| Split | Generated matches | Match rate | Unique hashes |
| --- | ---: | ---: | ---: |
| train | 154 | 0.037598 | 150 |
| val | 15 | 0.003662 | 15 |
| test | 22 | 0.005371 | 22 |

## Nearest Neighbors

| Split | Mean Hamming | P05 Hamming | Min Hamming | Exact | <=1% | <=2% | <=5% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 0.028018 | 0.000977 | 0.000000 | 154 | 990 | 1820 | 3514 |
| val | 0.051688 | 0.014648 | 0.000000 | 15 | 125 | 371 | 2067 |
| test | 0.049829 | 0.013672 | 0.000000 | 22 | 135 | 397 | 2220 |

## Dataset Duplicates

- Cross-split duplicate hashes: `0`
- Cross-split duplicate samples: `0`

| Split | Samples | Unique | Duplicate groups | Duplicate samples | Max group |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 74664 | 64233 | 9333 | 10431 | 3 |
| val | 9360 | 8061 | 1170 | 1299 | 3 |
| test | 9288 | 7968 | 1161 | 1320 | 3 |

## Generated Duplicates

- Unique generated images: `4091`
- Duplicate groups: `5`
- Duplicate generated samples: `5`
- Max generated duplicate group size: `2`

## Interpretation

There are exact validation matches. Review the matching images before claiming this model is fully non-memorizing.
