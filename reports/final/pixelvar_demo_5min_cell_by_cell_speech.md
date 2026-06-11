# PixelVAR Notebook Demo Speech

Target length: about 5 minutes.

Use with: `notebooks/02_pixelvar_final_demo.ipynb`

## Cell 1 - Title

This notebook shows the final PixelVAR system more concretely than the slides. The story is simple: a sprite becomes palette tokens, PixelVAR generates in that token space, and the metrics plus audit explain why we selected the main model.

## Cell 2 - Environment Setup

This setup cell prepares the environment. In Colab, it clones the correct branch, installs the requirements, and moves into the repository. Locally, it confirms the current environment is ready.

## Cell 3 - Imports And Helpers

This cell loads the plotting, table, path, and image helpers. Nothing scientific happens here; it just makes the notebook reproducible from repo artifacts.

## Cell 4 - Representation Section

The first real idea is representation. PixelVAR does not generate arbitrary RGB pixels. Each sprite is a transparent palette-token map: token 0 is background, and tokens 1 through 16 are fixed palette colors.

## Cell 5 - Token Pyramid Output

Here we see that representation visually: original sprite, transparency mask, tokenized palette map, and the 16 color tokens.

The bottom row is the key part. The sprite becomes a pyramid: 1x1, 2x2, 4x4, 8x8, 16x16, and 32x32. Together, these make a 1,365-token sequence.

The first few scales can be all token 0. That is not a bug: the sprite occupies a small part of a transparent 32x32 canvas, so coarse mode pooling may choose transparency. Visible structure appears at finer scales.

## Cell 6 - Selected Generation Section

Now we move from representation to generation. The selected model is the real-only PixelVAR branch, `var_sprites_v0_full`, sampled at temperature 0.8 and top-k 8.

## Cell 7 - Selected PixelVAR Output

This output shows the selected PixelVAR generation setting. If the checkpoint is available, it samples live; otherwise it shows the saved grid from the same configuration.

Either way, the point is the same: the sprites are compact, transparent, and palette-safe. The model generates valid sprite-like assets directly in token space instead of RGB images that need cleanup.

## Cell 8 - Sampling Control Section

Next, sampling settings. Even with the same checkpoint, temperature and top-k control how conservative or varied the samples become.

## Cell 9 - Sampling Sweep Output

Here we compare three settings. Lower temperature is conservative. The selected setting, temperature 0.8 with top-k 8, balances structure and variation. Higher temperature adds variety, but also weaker forms.

This is why the final decision uses both metrics and sample inspection. Training is not the whole story; sampling changes visible output.

## Cell 10 - Internal Branches Section

Now we compare PixelVAR against the other implemented branches. The main model was not selected in isolation.

## Cell 11 - Internal Branch Outputs

The table ranks the branches. Real-only VAR is selected with the best comparable sprite-feature score. HMAR is close at one refinement step, but more refinement did not help. Generated-keep and mixed-data branches trained, but did not improve real-validation quality. Patch-VQ worked as a learned-token ablation, but outputs were blockier.

The grids make this visible. PixelVAR and HMAR are the strongest non-memorizing candidates; other branches shift distribution, repeat forms, or lose sharpness.

## Cell 12 - Metric Trap Section

This is the key evaluation point. Raw image metrics alone can mislead on tiny sprites, because memorization can score well.

## Cell 13 - Metrics And Audit Output

The table and plots show the problem. Flat AR has the best raw FID, so FID alone would make it look like the winner.

But the exact-match audit changes the conclusion. Flat AR reproduced thousands of training samples and hundreds of validation and test samples, so it is not a clean generative winner. PixelVAR has slightly worse FID, but much lower exact-match counts, so it becomes the selected audit-aware result.

## Cell 14 - External Baselines Section

The external baselines are practical pressure tests. They are not perfect state-of-the-art comparisons, but they ask a useful question: do accessible diffusion-style generators naturally solve this normalized 32x32 transparent sprite task?

## Cell 15 - External Baseline Outputs

The answer is mostly no. PixelVAR has much better coverage and lower FID. Pokemon LoRA is the strongest external visual row, but still leaves frames, fragments, and mismatch artifacts. SSD-1B sometimes produces recognizable characters, but precision and coverage are low. SD-piXL fails this adapted protocol badly.

So the conclusion is narrow: not that diffusion is bad in general, but that for this transparent, palette-constrained 32x32 target, the palette-token model fits better.

## Cell 16 - Scale Section

The final section checks scale: whether the sampler stays stable beyond a small cherry-picked batch.

## Cell 17 - Scale Output And Closing

The 170,000-sample pass shows stable proposal-scale sampling. Most samples passed the automatic gate, some went to review, and only a tiny reject bucket remained.

This is not a human preference score; it is an automatic stability check. The final takeaway is that PixelVAR is a bounded but defensible 32x32 transparent-sprite generator because its representation matches the domain, and because evaluation includes memorization-aware auditing.
