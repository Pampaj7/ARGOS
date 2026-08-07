# ARGOS v2 hybrid temporal-memory oracle audit

Dataset-2 validation froze `CS1, CS2, CS4, CS8, CF1, CF2` before dataset 7 was opened. On dataset 7, raw EPE is 0.54773; the one-step raw, short-corrected, raw-anchor-bank and full-hybrid oracle gains are 0.05236, 0.09372, 0.13952 and 0.14442 EPE. The full hybrid exceeds the better short family by 0.05070 EPE and is positive on every seen backbone and sequence.

Raw anchors explain 96.6% of the full oracle gain. Corrected candidates add 0.00489 EPE after all raw anchors, and uniquely rescue 0.85% of pixels; far raw anchors rescue 9.08% where short corrected memory fails. The next implementation should therefore be raw-anchor-first and non-recurrent, with CF1/CF2 only as a controlled short-TTL extension.

The train-only Isolation Forest reaches AUROC/AP 0.812/0.683 for temporal-memory failure and 0.828/0.606 for harmful corrected updates on dataset 2. Median/MAD is informative but weaker. These are diagnostic validation results, not an authorization policy.

**Stage-1 verdict: GO for a causal multi-anchor architecture; conditional GO for a full provenance-typed hybrid because corrected memory adds only a small residual ceiling.** Dataset 7 is not a pristine project-wide holdout. No unseen backbone or external/OOD dataset was evaluated.
