# Final Model Decision Table

Lower feature scores are better. Rows marked comparable use the shared real Sprites validation palette-token evaluator. Patch-VQ uses a decoded RGBA evaluator, and generated-keep uses its own generated validation reference, so those scores are not direct winner comparisons.

This table ranks internal project branches. External baselines are tracked in
`reports/final/external_baseline_comparison.md`.

| Rank | Branch | Best score | Setting | Comparable | Decision |
| ---: | --- | ---: | --- | --- | --- |
| 1 | Real-only VAR | `0.00147` | temp=0.8, top_k=8 | yes | WINNER / main result |
| 2 | HMAR masked refinement | `0.00189` | temp=0.8, top_k=8, steps=1 | yes | Do not promote |
| 3 | Generated-keep VAR | `0.00157` | temp=0.8, top_k=8 | no | Do not promote |
| 4 | Real + generated mixed VAR | `0.00755` | temp=0.8, top_k=8 | yes | Do not promote |
| 5 | OpenGameArt-mixed VAR | `0.00778` | temp=0.8, top_k=8 | yes | Do not promote |
| 6 | Patch-VQ VAR | `0.04689` | temp=1, top_k=16 | no | Do not promote |

## Notes

- **Real-only VAR**: Best shared real-validation score; keep as primary checkpoint. Metrics: `reports/eval/sprites_v0_full/metrics.csv`. Grid: `reports/eval/sprites_v0_full/temp_0.8_topk_8_grid.png`.
- **HMAR masked refinement**: Technically successful; 1 refinement step is best, but still behind VAR. Metrics: `reports/eval/hmar_sprites_refinement_ablation/steps_1/metrics.csv`. Grid: `reports/eval/hmar_sprites_refinement_ablation/steps_1/temp_0.8_topk_8_grid.png`.
- **Generated-keep VAR**: Good self-reference score, but not the same real-validation comparison. Metrics: `reports/eval/sprites_generated_keep_v0_full/metrics.csv`. Grid: `reports/eval/sprites_generated_keep_v0_full/temp_0.8_topk_8_grid.png`.
- **Real + generated mixed VAR**: Trains cleanly, but real-validation score is worse than real-only VAR. Metrics: `reports/eval/sprites_mixed_v0_full_realval/metrics.csv`. Grid: `reports/eval/sprites_mixed_v0_full_realval/temp_0.8_topk_8_grid.png`.
- **OpenGameArt-mixed VAR**: Public OGA pipeline works, but the added data does not improve quality. Metrics: `reports/eval/sprites_mixed_oga_v0_full_realval/metrics.csv`. Grid: `reports/eval/sprites_mixed_oga_v0_full_realval/temp_0.8_topk_8_grid.png`.
- **Patch-VQ VAR**: Learned-token pipeline works, but samples are blockier and metric is separate. Metrics: `reports/eval/sprites_patchvq16_decoded/metrics.csv`. Grid: `reports/eval/sprites_patchvq16_decoded/temp_1_topk_16_grid.png`.

## Decision

`var_sprites_v0_full` remains the main result. HMAR is the closest ablation on the same real-validation metric, but still does not beat the real-only VAR baseline.
