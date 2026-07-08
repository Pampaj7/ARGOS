# ARGOS v2 Causal BiDA Temporal Diagnostic Report

## Executive Verdict
`TEMPORAL_SMOOTHING_WITHOUT_STATE_USE`.

FaithfulCausalBiDA improves geometry and some temporal metrics over raw on the diagnostic validation subset, but persistent hidden state does not beat reset/zero-state. The useful signal is mostly aligned previous disparity/local evidence, not long-lived propagated memory.

Diagnostic scope: all 4 validation sequences, capped to the first 128 frames per sequence for this state audit. The full one-seed ladder remains the reference for full validation MAE.

## Checkpoints And Configs
- aligned: `results/03_temporal_refinement/argos_v2/one_seed_ladder/aligned_local_faithful_seed0/checkpoints/best.pt`
- faithful: `results/03_temporal_refinement/argos_v2/one_seed_ladder/faithful_causal_bida_seed0/checkpoints/best.pt`
- safe: `results/03_temporal_refinement/argos_v2/one_seed_ladder/safe_causal_bida_seed0/checkpoints/best.pt`
- manifest: `causal_bida_diagnostic_manifest.json`

## Temporal Metrics Table
| Config | MAE | Delta | TGM | MC inconsistency | HF temporal error | New-Bad3 | Modified |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 3.4599 | 0.0000 | 1.0296 | 2.9569 | 1.4957 | 0.0000 | 0.0000 |
| aligned local | 3.4173 | 0.0427 | 1.0139 | 2.8808 | 1.4690 | 0.7452 | 0.3851 |
| faithful full | 3.3826 | 0.0773 | 1.0269 | 2.9481 | 1.4895 | 1.1910 | 0.9935 |
| faithful reset | 3.3800 | 0.0799 | 1.0269 | 2.9482 | 1.4896 | 1.2550 | 0.9954 |
| shuffled existing | 3.3844 | 0.0755 | 1.0279 | 2.9497 | 1.4913 | 1.1652 | 0.9893 |
| shuffled corrected | 3.3819 | 0.0781 | 1.0277 | 2.9496 | 1.4911 | 1.2335 | 0.9919 |
| safe | 3.4599 | 0.0000 | 1.0296 | 2.9569 | 1.4957 | 0.0000 | 0.0000 |

## Full vs Reset vs Shuffled
State-reset means hidden state reset every frame while previous aligned disparity remains available. It is aligned local history without persistent propagation, not current-only.

- Full persistent MAE: 3.3826
- Reset MAE: 3.3800
- Existing shuffled MAE: 3.3844
- Corrected shuffled MAE: 3.3819

Full does not beat reset. Corrected shuffled remains close. Persistent memory is not confirmed.

## State Sensitivity
From `causal_bida_state_usage.json`:
- full vs reset mean output difference: 0.0119 px
- full vs zero-state mean output difference: 0.0119 px
- full vs existing shuffled mean output difference: 0.0198 px

These are tiny relative to the correction field and validation MAE. The state materially exists, but it barely changes the output.

## Memory Horizon
`causal_bida_memory_horizon.csv` shows reset-every-1/2/4/8 and persistent are nearly tied. The best MAE in this audit is reset-every-1, not persistent. Long horizon is not providing useful signal.

## Gradient/Block Usage
The diagnostic backward pass gives finite gradients in all intended Faithful blocks. Propagation/local mean grad norm ratio is 2.12. So the propagation block can train, but ablation shows the trained checkpoint does not rely on persistent state at inference.

## Shuffled-History Validity
The original shuffled-history mode is partially invalid: it shuffles previous raw disparity but keeps hidden state chronological. This audit adds `faithful_shuffled_corrected`, which resets/corrupts state consistently with shuffled local evidence and never uses future frames.

## Training Semantics Diagnosis
Training samples random 8-frame clips, starts state from zero for each clip, carries no state across clips, and selects checkpoints by validation MAE. This setup encourages short-window local alignment, not full-sequence persistent memory.

## Safe Model Collapse
SafeCausalBiDA is identity: modified ratio 0, no MAE change. Cause is closed gate bias `-4.0` plus zero residual init, reinforced by safe/sparse losses and MAE-only checkpoint selection.

## Final Classification
- FaithfulCausalBiDA: `TEMPORAL_SMOOTHING_WITHOUT_STATE_USE`
- SafeCausalBiDA: `WARM_START_RECOMMENDED`

## Exact Next Experiment
Do not run three seeds yet. Run one minimal safe warm-start from FaithfulCausalBiDA: copy faithful core, initialize gate open, start safety/sparsity at zero, ramp them gradually, and log hidden-state contribution during validation. If state still ties reset, promote aligned-local as the actual confirmed ARGOS v2 mechanism.
