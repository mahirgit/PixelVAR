# MDIGAN Decision

Decision date: 2026-06-08

## Decision

MDIGAN should be cited as related work, not used as a main numeric baseline for
the current PixelVAR comparison.

This is a task-fit decision, not a dismissal of the paper. MDIGAN is highly
related to pixel-art character sprites, but it solves a different problem: given
one or more poses of the same character, generate the missing pose. PixelVAR's
reported result is unconditional 32x32 sprite generation from learned palette
tokens.

## Why It Is Not A Main Baseline

1. Task mismatch

   MDIGAN is a conditional missing-pose imputation model. Its generator receives
   available pose images for a character plus a target pose label, then outputs
   the missing pose. PixelVAR samples new sprites without input reference poses.
   A direct FID/KID row beside PixelVAR would therefore compare conditional
   completion against unconditional generation.

2. Data mismatch

   The official MDIGAN setup expects paired domain folders such as `0-back`,
   `1-left`, `2-front`, and `3-right`, with matching image indices for each
   character. The current PixelVAR final protocol uses single 32x32 generated
   sprites and a validation distribution, not a four-direction paired-pose test
   set. Our MSD replacement curation groups frames by static character fields for
   split safety, but it does not provide the clean semantic four-pose domain
   layout required for an honest MDIGAN run.

3. Metric mismatch

   MDIGAN's evaluation is naturally paired: the generated missing pose is
   compared to the held-out target pose of the same character. Our external
   evaluator compares folders of independent generated samples to a validation
   distribution. Feeding validation poses into MDIGAN and scoring the imputed
   targets as if they were unconditional samples would give MDIGAN privileged
   character-specific context that PixelVAR does not receive.

4. Engineering cost without fair payoff

   The public repository is TensorFlow 2.10 / Python 3.9 oriented and expects a
   custom paired-domain dataset integration. Building that integration is
   feasible, but it would create a new conditional-pose-completion experiment,
   not a fair replacement for the current unconditional baseline table.

## How To Mention It

Use this framing in the report:

> MDIGAN is a strong related-work reference for character sprite pose
> imputation. We did not include it in the main numeric comparison because its
> input/output protocol is conditional and paired, while PixelVAR is evaluated as
> an unconditional generator.

## If We Later Want To Run It

Run MDIGAN only as a separate conditional experiment:

1. Build or acquire a four-direction paired dataset with reliable `back`,
   `left`, `front`, and `right` domains.
2. Split by character identity, not by frame.
3. Train MDIGAN on the paired training characters.
4. Evaluate missing-pose reconstruction on held-out characters with paired
   metrics and sample sheets.
5. Label the result as "conditional pose imputation", separate from the
   unconditional PixelVAR table.

That future experiment would be useful, but it would answer a different
question from the proposal's main PixelVAR result.

## Sources Checked

- MDIGAN paper: https://arxiv.org/abs/2409.10721
- Official repository: https://github.com/fegemo/mdigan-characters
- Official project page: https://fegemo.github.io/mdigan-characters
