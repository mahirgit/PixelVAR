# PixelVAR 10-Minute Exact Speech

Target: about 9:40 to 10:00 at a rehearsed but natural pace.

Split:

- Speaker 1: slides 1-4 and 17, about 3:25.
- Speaker 2: slides 5-8, about 2:30.
- Speaker 3: slides 9-16, about 3:55. This is the largest section because it carries the metrics, audit, and baselines.

## Speaker 1

### Slide 1

Hi everyone. We are presenting PixelVAR, our project on coarse-to-fine pixel-art sprite generation using palette tokens. The short version is this: instead of generating RGB images and cleaning them afterward, we generate sprites directly as transparent palette-token maps.

Our final system focuses on 32 by 32 sprites with one transparency token and sixteen color tokens. We evaluate it with samples, baselines, ablations, external comparisons, and a memorization audit.

### Slide 2

The reason this matters is that pixel art is not just a small image. In normal image generation, a tiny color change or a little blur may not matter much. In pixel art, one wrong pixel can break an outline, change a pose, or make a character unreadable.

Sprites also need transparency. The background is part of the asset format, and the color palette often has to stay small and controlled.

So our framing was: do not generate continuous RGB and then repair it. Make the model generate the discrete thing we actually care about: transparent palette-index tokens.

### Slide 3

This puts PixelVAR between two areas. On one side, there is pixel-art-specific work, like GAN-based or conditional sprite systems. Those are useful, but often they are pose-based or conditional.

On the other side, there are strong modern image generators. But diffusion is continuous and prompt-first, so it often needs cleanup to become a valid sprite. VAR and HMAR give us the right coarse-to-fine idea, but not for tiny transparent palette-constrained sprites.

Our niche is exactly there: next-scale prediction for small sprites where transparency and palette membership are native to the representation.

### Slide 4

Here is the honest proposal contract. We completed the 32 by 32 dataset pipeline, the main VAR option, HMAR, internal baselines, known metrics, the memorization audit, and external comparison rows.

The biggest change was the tokenizer. The proposal expected a learned multi-scale VQ-VAE-style tokenizer. We implemented and tried that direction, but reconstructions were too soft or ghosted. For pixel art, crisp exact pixels matter more than a more elegant learned latent space.

So the final main path is a deterministic palette-token pyramid. That is a pivot, but it is a defensible one: it protects the exact constraints that make pixel art usable.

Speaker 1 to Speaker 2: So the key design question became: how do we represent sprites so the model never has to repair palette or transparency afterward?

## Speaker 2

### Slide 5

The method is a six-scale pyramid: 1 by 1, 2 by 2, 4 by 4, 8 by 8, 16 by 16, and finally 32 by 32. That gives 1,365 total tokens.

PixelVAR predicts each next scale conditioned on coarser scales, so it learns the rough silhouette first and then fills in detail. HMAR uses the same ordering, but masks and refines tokens within each scale.

The token contract is important: token 0 is transparent background, tokens 1 through 16 are palette colors, and HMAR has a separate mask token. That means missing information is never confused with real transparency.

### Slide 6

For evaluation, we used a fixed 32 by 32 transparent RGBA protocol. The final curated MSD replacement dataset has 93,312 sprites, with grouped train, validation, and test splits.

We evaluated in three layers: a sprite-specific feature score, known image metrics like FID and KID, and an exact-match audit. The audit matters because tiny sprites are easy to memorize, and raw image metrics can be misleading.

We also compared against internal baselines and accessible external diffusion-style baselines.

### Slide 7

This is the selected PixelVAR checkpoint. The samples are compact, readable, and palette-safe. The best sampling setting was temperature 0.8 with top-k 8.

The sprite-feature score is 0.00147, the best completed comparable branch in our internal decision table. Palette consistency is 1.0, but we should not oversell that. It is expected because the model can only output transparent plus palette tokens.

The more meaningful signs are the low sprite-feature score and the close edge-density match: generated edge density is 0.1808, while real validation is 0.1811.

### Slide 8

We did not choose the model from one pretty sample sheet. We compared several implemented branches.

HMAR step 1 was close, with score 0.00189, but did not beat the main VAR. Generated-keep training looked reasonable, but larger synthetic data did not improve real-validation score. OpenGameArt mixing worked as a pipeline, but the distribution did not match. Patch-VQ worked as a learned-token ablation, but it was blockier.

So the selected result is not just "the model we liked visually." It is the best supported branch among the completed comparable options.

Speaker 2 to Speaker 3: The main model worked, but the important question is whether it beat meaningful alternatives without just memorizing.

## Speaker 3

### Slide 9

The HMAR ablation is useful because it shows a negative result. The hypothesis was reasonable: maybe iterative refinement would help outlines. But at this resolution, more refinement did not help.

HMAR step 1 was close, but steps 2, 4, and 8 got worse on the sprite-feature evaluator. This tells us that more computation or more refinement is not automatically better when the output is only 32 by 32 and heavily constrained.

### Slide 10

On the pixel-art domain metrics, palette consistency is perfect for all token models, so it is not the main quality signal. Opacity, edge density, nearest-neighbor distance, and exact-match rate carry more weight.

PixelVAR and HMAR both match the reference edge density closely. Flat MaskGIT under-fills the sprites and loses coverage. Flat AR looks strong on several numbers, but the exact-match rate is already a warning sign.

### Slide 11

The known image metrics make that warning clearer. If we only looked at raw Inception-style metrics, Flat AR would look like the winner: best FID, KID, precision, recall, and coverage.

But this is exactly why tiny-sprite generation needs an audit. A model can score well by reproducing the training distribution too literally. So these metrics are useful, but they are not enough by themselves.

### Slide 12

This is the centerpiece result. Raw FID would pick Flat AR. The exact-match audit changes the conclusion.

Flat AR exactly reproduced 3,162 training samples, plus 365 validation samples and 333 test samples. So it cannot be treated as a clean winner, even though its FID is best.

PixelVAR has a slightly worse FID, but only 13 validation exact matches and 11 test exact matches. HMAR is similar. So our conclusion is audit-aware: Flat AR wins raw metrics, but PixelVAR is selected because it is strong without the same memorization failure.

### Slide 13

The expanded scoreboard combines the story. Internally, PixelVAR has the best sprite-feature score, HMAR is close, Flat AR is disqualified as a clean winner by the audit, and Flat MaskGIT has weak coverage.

For external rows, PixelVAR is also far better under our normalized 32 by 32 evaluation. But the claim is careful: these are practical pressure tests under the same output normalization, not proof of broad state of the art.

### Slide 14

This slide shows that pressure test visually and numerically. PixelVAR's 256-sample row has much lower FID and much higher coverage than Pokemon LoRA and SSD-1B. SD-piXL failed the adapted protocol especially badly.

The important point is not that diffusion is bad in general. The point is that accessible diffusion-style generators did not naturally satisfy this very specific target: small, transparent, palette-constrained sprites after normalization.

### Slide 15

The visual comparison explains why we need both samples and metrics. PixelVAR and HMAR produce coherent rows. Flat AR can look plausible, but that plausibility is partly memorization. Flat MaskGIT repeats forms and loses coverage.

So the final decision is not based on one metric or one sample sheet. It comes from the combination of visual quality, domain metrics, known image metrics, and the exact-match audit.

### Slide 16

We also tested scale. The 170,000 generation pass stayed stable: 161,479 samples passed the automatic gate, about 8,500 went into review, and only 21 were rejected.

This is not a human approval result. It means the sampler remained stable at proposal-scale generation volume, far beyond a small cherry-picked batch.

Speaker 3 to Speaker 1: So the final claim is intentionally narrow: strong 32 by 32 palette-token generation, with honest limits.

## Speaker 1

### Slide 17

The final claim is that PixelVAR works because the representation matches the domain. Pixel art is discrete, transparent, and palette-constrained, so direct palette-token generation fits better than RGB generation plus repair.

Our strongest result is not a broad state-of-the-art claim. It is bounded: a 32 by 32, 16-color-plus-alpha sprite generator, selected through ablations, external pressure tests, and memorization-aware evaluation.

What remains is also clear. We still need to complete the 8-color, 32-color, and 4-scale ablation metrics, inspect those sample sheets carefully, and then revisit 64 by 64 generation with a realistic compute budget.

So our takeaway is: coarse-to-fine palette-token generation is a defensible approach for transparent 32 by 32 sprite generation, but it has to be evaluated with audits, not just raw image metrics.
