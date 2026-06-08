# External Baseline Comparison

## Reading Rules

All external outputs in this table were normalized into the same `32x32` RGBA
PNG protocol with the PixelVAR palette before evaluation.

Do not compare all rows as if they had the same statistical strength:

- SD-piXL was evaluated on `16` images because the method is slow and visually
  failed under our protocol.
- SSD-1B practical diffusion was first evaluated on `64` images, then scaled to
  a stronger `256`-image run for the matched practical-generator comparison.
- Pokemon sprite LoRA was first evaluated on `64` images, then scaled to the
  stronger `256`-image run now used as the primary practical external-generator
  result.
- Each external method should be compared primarily against the matching
  PixelVAR row from the same evaluation report and sample count.

Palette consistency is expected to be `1.0000` after normalization. It confirms
protocol compliance, not that an external model natively learned our palette.

## Metric Summary

Lower is better for FID, KID, and MS-SSIM. Higher is better for precision,
recall, density, and coverage.

| External baseline | Images | Matching PixelVAR FID | External FID | Matching PixelVAR KID | External KID | Matching PixelVAR precision | External precision | Matching PixelVAR recall | External recall | Matching PixelVAR coverage | External coverage | External MS-SSIM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SD-piXL | 16 | 151.0927 | 497.7065 | 0.014713 | 0.517647 | 1.0000 | 0.0000 | 0.9375 | 0.0000 | 1.0000 | 0.0000 | 0.4926 |
| SSD-1B practical diffusion smoke | 64 | 87.8128 | 196.9628 | 0.007124 | 0.092097 | 0.8125 | 0.0312 | 0.9375 | 0.6719 | 0.9219 | 0.0469 | 0.3935 |
| SSD-1B practical diffusion primary | 256 | 46.4794 | 158.5819 | 0.007480 | 0.102159 | 0.9102 | 0.0547 | 0.8711 | 0.4922 | 0.8906 | 0.0352 | 0.4107 |
| Pokemon sprite SDXL LoRA smoke | 64 | 87.8128 | 184.3101 | 0.007124 | 0.130036 | 0.8125 | 0.3125 | 0.9375 | 0.2812 | 0.9219 | 0.1094 | 0.5171 |
| Pokemon sprite SDXL LoRA primary | 256 | 46.4794 | 154.0150 | 0.007480 | 0.136589 | 0.9102 | 0.1172 | 0.8711 | 0.1211 | 0.8906 | 0.0508 | 0.5421 |

## Visual and Protocol Summary

| Baseline | Why it was included | Visual result | Main caveat | Current decision |
| --- | --- | --- | --- | --- |
| SD-piXL | Most research-targeted public low-resolution/color-limited baseline we could run. | Poor: noisy blocks and non-sprite outputs. | Prompt/image-conditioned optimization, not a trained unconditional sprite sampler; slow enough that scaling is not worthwhile after the failed 16-image batch. | Keep as a serious attempted research baseline, not a competitive result. |
| SSD-1B practical diffusion | Generic SDXL-family practical baseline people expect. | Weak/moderate: some recognizable sprites, many lineup/crop/multi-character failures. | Prompt-sensitive text-to-image model; not sprite-dataset-trained. | Keep as generic practical comparison; 256-image run completed and still below PixelVAR. |
| Pokemon sprite SDXL LoRA | Strongest accessible practical sprite-specific generator found so far. | Moderate: many centered sprites, but side fragments, frames, paired characters, and style mismatch remain. | Trained for Pokemon trainer-style sprites, not MSD Sprites; text-to-image protocol is still not architecture-equivalent to PixelVAR. | Primary practical external generator baseline; 256-image run completed and still below PixelVAR. |

## Artifact Links

| Artifact | File |
| --- | --- |
| SD-piXL metrics | `reports/external_eval/main_vs_sd_pixl_16/metrics.csv` |
| SD-piXL report | `reports/external_eval/main_vs_sd_pixl_16/evaluation_report.md` |
| SD-piXL sample sheet | `reports/final/sd_pixl_sample_sheet.png` |
| SSD-1B practical diffusion metrics | `reports/external_eval/main_vs_practical_diffusion_64/metrics.csv` |
| SSD-1B practical diffusion report | `reports/external_eval/main_vs_practical_diffusion_64/evaluation_report.md` |
| SSD-1B practical diffusion 256 metrics | `reports/external_eval/main_vs_practical_diffusion_256/metrics.csv` |
| SSD-1B practical diffusion 256 report | `reports/external_eval/main_vs_practical_diffusion_256/evaluation_report.md` |
| SSD-1B practical diffusion sample sheet | `reports/final/practical_diffusion_sample_sheet.png` |
| Pokemon sprite LoRA metrics | `reports/external_eval/main_vs_pokemon_sprite_lora_64/metrics.csv` |
| Pokemon sprite LoRA report | `reports/external_eval/main_vs_pokemon_sprite_lora_64/evaluation_report.md` |
| Pokemon sprite LoRA 256 metrics | `reports/external_eval/main_vs_pokemon_sprite_lora_256/metrics.csv` |
| Pokemon sprite LoRA 256 report | `reports/external_eval/main_vs_pokemon_sprite_lora_256/evaluation_report.md` |
| Pokemon sprite LoRA sample sheet | `reports/final/pokemon_sprite_lora_sample_sheet.png` |

## Bottom Line

PixelVAR remains ahead of every external baseline we have actually run under the
shared normalized-image protocol. The Pokemon sprite SDXL LoRA is the best
practical external visual baseline, but the 256-image runs confirm that neither
practical generator closes the gap. SSD-1B practical diffusion reaches FID
`158.5819` and coverage `0.0352`; Pokemon sprite LoRA reaches FID `154.0150`
and coverage `0.0508`; PixelVAR's matching 256-image row is FID `46.4794` and
coverage `0.8906`.
