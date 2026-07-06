# D4D temporal-consistency findings

6 configs × 6 full consecutive clips (2/specimen, ≤120 frames), causal inference, RAFT
motion-compensated diagnostics (fwd-bwd occlusion). **No dense temporal GT** — all temporal
metrics are prediction-space / motion-compensated diagnostics; lower is NOT automatically
better (tissue is non-rigid).

## Primary question
> Does SCARED temporal pretraining give a temporal-stability advantage on D4D, even when
> calibration-only collapses to identity and scratch wins sparse MAE?

**Answer: No.** No configuration meaningfully changes temporal consistency vs raw S2M2-S.

## Temporal vs geometric (aggregate)

| config | anchor MAE | MC-inconsist. | HF energy | depth MC (mm) | boundary MC | modified % | \|applied\| |
|--------|-----------:|--------------:|----------:|--------------:|------------:|-----------:|-----------:|
| raw        | 3.50 | **0.390** | **0.507** | **0.691** | 2.802 | — | — |
| zero-shot  | 4.72 | 0.397 | 0.525 | 0.717 | 2.698 | 24.7 | 0.346 |
| calib-2s   | 3.66 | 0.400 | 0.525 | 0.722 | 2.760 | 20.2 | 0.212 |
| calib-8s   | 3.50 | 0.400 | 0.525 | 0.723 | 2.758 | 17.0 | 0.195 |
| full-4s    | 4.09 | 0.394 | 0.521 | 0.706 | 2.704 | 19.4 | 0.241 |
| scratch-4s | 3.47 | 0.390 | 0.507 | 0.691 | 2.802 | 0.0 | 0.002 |

## Reading
- **Temporal axis is flat**: MC-inconsistency 0.390–0.400 (2.5% spread), HF energy 0.507–0.525.
  The refiners' corrections are small relative to real inter-frame tissue/camera motion, which
  dominates the motion-compensated residual — so they barely move the temporal metrics.
- **Refiners slightly WORSEN pure temporal jitter**: zero-shot/calib/full raise HF energy
  (0.507→0.52) and depth MC (0.691→0.72). They add per-frame correction noise, not temporal
  smoothing. Only boundary MC drops marginally (2.80→2.70).
- **Identity collapse confirmed**: calibration halves correction footprint (modified 24.7→17%,
  \|applied\| 0.35→0.20) and its temporal metrics converge to raw. scratch-4s is literal
  identity (modified 0%, \|applied\| 0.002) → temporally identical to raw.
- **Pretrained-full vs scratch**: full-4s modifies 19% of pixels but is NOT more temporally
  stable than scratch's identity (mc 0.394 vs 0.390; hf 0.521 vs 0.507). Pretraining's
  corrections buy no temporal advantage — if anything a touch more jitter.

## Conclusion
On D4D, "EGBM-v3-CARE-S" does not behave as a temporal stabiliser: its effect is per-frame
spatial correction whose temporal footprint is in the noise of real non-rigid motion.
Adaptation tunes correction MAGNITUDE (geometric safety), not temporal stability. Combined
with the few-shot pilot: the models become geometrically **safe**, never temporally
**beneficial**, on this domain. A temporal-consistency contribution is **not** supported by D4D.
