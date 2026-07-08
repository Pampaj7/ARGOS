# ARGOS v2 One-Seed Ladder Readiness

## Status

Ready for a small one-seed SCARED ladder after this certification. Do not launch from this task.

## Configs

1. `raw`
2. `current_only`
3. `aligned_local_faithful`
4. `aligned_local_safe`
5. `faithful_causal_bida`
6. `faithful_causal_bida_state_reset`
7. `faithful_causal_bida_shuffled_history`
8. `safe_causal_bida`

## Reference Eval Commands

Smoke command shape:

```bash
.miniconda/envs/argos/bin/python scripts/temporal_refinement/eval_scripts/evaluate_argos_v2_streaming.py   --model faithful_causal_bida   --mode full   --max-frames 32   --device cpu
```

For training/evaluation ladder, use the same model names and output under:

```text
results/03_temporal_refinement/argos_v2/one_seed_ladder/<config>/
```

## Expected Runtime

Current CPU smoke is intentionally slow but tiny. Full ladder should use one H100, but only after adding the training wrapper and compact streaming metric writer.

## Remaining Blockers

- No training wrapper for the ladder yet.
- NVDS-lite still needs an adapter if it must be compared through the same `step()` streaming evaluator.
- D4D/SERV-CT remain out of scope until SCARED one-seed results justify promotion.

## Decision Rule

Proceed only if aligned-local-only beats current-only/unaligned baselines and causal BiDA beats aligned-local-only without safety regression.
