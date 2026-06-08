# PixelVAR Speaker Guide

Use `reports/final/pixelvar_final_presentation_all_main_17slides.pptx` for the live talk.
Use `reports/final/pixelvar_final_presentation_master_evidence.pptx` only as the full evidence archive.

Do not present `main` as the final code state. The final working branch is `codex/pixelvar-external-baselines`.

This version has no backup section. All 17 slides are main slides and can be presented. The added comparison/metric slides should be kept short verbally, not read line by line.

## Story Spine

Pixel art is discrete. Continuous generators miss palette and transparency constraints. PixelVAR generates palette tokens directly. The proposed neural-tokenizer path was implemented/tried but was too soft for crisp pixel art, so the main result pivoted to deterministic palette tokens. VAR, HMAR, internal baselines, external baselines, metrics, and audits were implemented. The most important result is the audit: raw FID would reward Flat AR memorization. Limitations are disclosed, not hidden.

## Main Deck Timing

| Slide | Speaker | Time | Purpose |
|---:|---|---:|---|
| 1 | P1 | 0:20 | Thesis: sprites as palette tokens, not repaired RGB |
| 2 | P1 | 0:45 | Pixel art constraints: palette, transparency, exact pixels |
| 3 | P1 | 0:45 | Related work gap in four rows |
| 4 | P1 | 0:45 | Proposal contract and tokenizer pivot |
| 5 | P2 | 1:05 | Core methodology: pyramid, token contract, HMAR mask |
| 6 | P2 | 0:45 | Dataset, split, metric layers, baselines |
| 7 | P2 | 0:55 | Main result: samples and selected checkpoint |
| 8 | P2 | 0:45 | Internal sample comparison across implemented branches |
| 9 | P3 | 0:45 | HMAR ablation: close, but more refinement did not help |
| 10 | P3 | 0:35 | Domain metrics: palette, opacity, edge, exact-match caveat |
| 11 | P3 | 0:45 | Known image metrics full table |
| 12 | P3 | 1:00 | Centerpiece audit: raw FID vs memorization |
| 13 | P3 | 0:45 | Expanded metric scoreboard across internal and external rows |
| 14 | P3 | 0:45 | External baselines: normalized protocol plus failure visuals |
| 15 | P3 | 0:45 | Baseline sample comparison |
| 16 | P3 | 0:25 | 170K generation scale; automatic gate, not user study |
| 17 | P1 | 0:45 | Limitations, next proof, bounded final claim |

Total planned time: about 12-13 minutes if spoken comfortably. For a strict 10-minute slot, compress slides 10-11 and 13 to one-sentence reads.

## Presenter Split

P1 covers motivation, gap, proposal contract, pivot, and final limitations: slides 1-4 and 17.

P2 covers method, setup, main result, and internal sample comparison: slides 5-8.

P3 covers ablations, metrics, audit, external baselines, sample comparisons, and generation scale: slides 9-16.

Transition from P1 to P2: "So the key design question became: how do we represent sprites so the model never has to repair palette or transparency afterward?"

Transition from P2 to P3: "The main model worked, but the important question is whether it beat meaningful alternatives without just memorizing."

Transition from P3 to P1: "So the final claim is intentionally narrow: strong 32x32 palette-token generation, with honest limits."

## Safe Claims

- PixelVAR is the strongest completed comparable proposal branch on our sprite-feature evaluator.
- Generating directly in palette-token space gives native transparency and palette compliance.
- HMAR was implemented and close, but did not beat single-pass VAR in this 32x32 low-color setting.
- Raw FID alone would promote Flat AR, but exact-match auditing shows it is memorizing.
- Accessible diffusion-style baselines did not beat PixelVAR under our normalized 32x32 protocol, but we do not claim broad SOTA.
- The VQ-VAE path was implemented/tried but not promoted because crisp pixel reconstructions mattered more than latent elegance.

## Claims To Avoid

- Do not say "we achieved SOTA."
- Do not say palette consistency proves quality.
- Do not call the 170K keep rate a human approval rate.
- Do not say Flat AR is simply worse than PixelVAR without explaining that raw FID prefers Flat AR but the audit disqualifies it as a clean winner.
- Do not say all proposal items are complete.
- Do not say external baselines are perfectly fair apples-to-apples comparisons.
- Do not say the user study was replaced by metrics. Say it remains future work and quantitative/audit evaluation was strengthened.

## Key Lines

Slide 1: "PixelVAR generates pixel-art sprites directly as transparent palette-token maps, not RGB images repaired afterward."

Slide 4: "We did not hide the VQ-VAE issue. We implemented/tried the neural-token direction, but crisp pixel art made deterministic palette tokens the defensible main path."

Slide 5: "Token 0 is transparency, tokens 1 through 16 are palette colors, and HMAR uses a separate mask token so background is not confused with missing information."

Slide 7: "Palette consistency is 1.0 by construction. The meaningful parts are sprite-feature score and structure matching."

Slide 9: "If we used only raw FID, Flat AR would look like the winner. The exact-match audit changes the scientific conclusion."

Slide 11: "This is not a human preference study. It shows that the sampler stayed stable at proposal-scale generation volume."

Slide 12: "Our main claim is not broad SOTA. It is that coarse-to-fine palette-token generation works for 32x32 transparent sprites and needs audit-aware evaluation."

## Likely Q&A

Q: Why did you move away from the proposed VQ-VAE tokenizer?
A: We implemented/tried the neural-token direction, but reconstructions were too soft or ghosted for pixel art. Pixel-art quality depends on crisp exact pixels, so deterministic palette tokens better matched the domain. Patch-VQ remains a learned-token ablation, not the main result.

Q: Why is palette consistency always 1.0?
A: Because the output space is constrained to transparent plus palette tokens. It is a representation guarantee, not a standalone quality metric.

Q: Why not use only FID?
A: FID rewarded Flat AR, but Flat AR reproduced 3162/4096 training samples and hundreds of held-out samples. For tiny sprites, exact-match auditing is essential.

Q: Is PixelVAR better than HMAR?
A: On our sprite-feature evaluator, yes: 0.00147 for PixelVAR versus 0.00189 for HMAR step 1. HMAR is close and slightly better on some Inception-style metrics, so the conservative conclusion is that both are strong non-memorizing candidates, with PixelVAR selected as the main model.

Q: What is the biggest missing result?
A: A user study and completed 8/32 palette plus 4-scale ablations. The runners exist, but we are not claiming results until metrics and sample sheets are inspected.

Q: Why is MDIGAN not in the numeric table?
A: MDIGAN is conditional paired-pose imputation. PixelVAR is unconditional generation. Putting MDIGAN in the same numeric table would give it reference-pose information PixelVAR does not get, so we cite it as related work instead.
