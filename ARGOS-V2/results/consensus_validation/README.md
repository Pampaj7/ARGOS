# Cross-Memory Consensus Correction (CMC) validation

## Decision

**NO-GO.** The predeclared stage-1 gate failed on the train-split sweep;
stages 2 (held-out) and 3 (unseen backbone) were therefore not run, per the
protocol fixed in `model_design/CONSENSUS_AUDIT.md` before any run started.

CMC is an ARGOS-original zero-parameter trick: treat tight agreement among the
four BiDA-aligned memories (t-1/2/4/8) as a statistical witness that the
current raw disparity is the outlier, and move raw toward the memory median,
bounded at 3 px, only where consensus is tight and disagreement large.

## The scientific result: memory errors are correlated

The sweep falsifies the core hypothesis, and the way it fails is the useful
finding. On 3 train sequences x 300 frames x 3 seen backbones
(2,568 frame-evaluations, 30.07M pixel observations at coverage 0.50):

| Config regime | Gain (px) | Update ratio | False-update rate |
|---|---:|---:|---:|
| Best safe (`n4_s0.25_d0.5_k1`) | +0.0013 | 0.21% | 22.8% |
| Aggressive (`n3_s1_d0.5_k0`) | -0.0116 | 2.44% | **85.8%** |
| Predeclared gate | >= +0.005 | — | <= 20% |

When all four independently-warped past estimates agree tightly with each
other yet disagree with raw, they are wrong together in 23-86% of cases
depending on how hard the gate is pushed. Multi-witness consensus is not an
independent-witness test in this domain: persistent artifacts (static
specular patches, repeated occlusion boundaries) contaminate every memory age
in the same image region, exactly the correlated-error failure mode
predeclared in the audit. It dominates rather than being the exception.

The mechanism ceiling confirms this is intrinsic, not a thresholding issue:
applying the bounded median move *only where GT says it helps* yields just
0.0297 px — 30% of the committed multi-memory oracle (0.098 px). The
remaining 70% of the oracle requires per-pixel *selection among individual
ages*, not any consensus statistic; and per-age selection is precisely the
decision measured to be near-unlearnable (selector AUROC ~0.55) in
`results/learned_t1_refiner/`.

Combined with the four committed studies, this closes the loop on the
temporal-memory question: the multi-memory oracle gain is real but resides in
pixels where exactly one age happens to be right — not where memories agree
(CMC, this study), not in latent state (EndoStreamDepth study), not in
learned per-age scores (PPM study), and not in frozen semantic features
(DINOv3 study).

## What was run

- Stage 1 sweep: predeclared 36-config grid (`n_min x tau_s x tau_d x kappa`),
  train-split sequences only (`dataset_1_keyframe_2`, `dataset_2_keyframe_4`,
  `dataset_6_keyframe_4`), 3 seen backbones, ages (1,2,4,8), SEA-RAFT flow,
  canonical BiDA alignment, namespace `cache-grid-from-cached-predictions`,
  pixel-count-weighted, common mask = GT coverage & raw-valid &
  aligned-t1-valid (t-1 warp support folded into aligned validity).
- Stage 2/3: not authorized by the gate; no held-out or unseen data was
  touched by CMC beyond what previous committed studies already used.
- Safety of the best safe config was actually excellent (clean1 degradation
  0.04%, new-Bad3 0.002%, vs 30%/28% for the learned t-1 refiner) — the gate
  failed on gain and on false-update rate, not on clean-pixel damage.

## Files

- Design + predeclared gates: `model_design/CONSENSUS_AUDIT.md`
- Component: `model_design/external_components/temporal_consensus.py`
- Tests: `model_design/tests/test_temporal_consensus.py` (6 passed)
- Runner: `scripts/run_consensus_validation.py`
- Sweep outputs: `sweep_summary.json`, `sweep_summary_by_backbone.csv`,
  `sweep_frame_metrics.csv`, `sweep_config.json`, `split_manifest.json`,
  `run.log`

## Reproduction

```bash
cd /dtu/p1/leopam/ARGOS/ARGOS-V2
source /dtu/p1/leopam/ARGOS/.miniconda/etc/profile.d/conda.sh && conda activate argos
ESUB_BYPASS=1 ESUB_QUIET=1 bsub -q p1i -app h100app -gpu "num=1:mode=shared" -n 4 -I \
  python3 scripts/run_consensus_validation.py --stage sweep --frames 300
# unit tests
python3 -m pytest -q model_design/tests/test_temporal_consensus.py
```
