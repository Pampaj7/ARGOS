# ARGOS v2 Streaming Evaluator Implementation Report

## Implemented File

- `scripts/temporal_refinement/eval_scripts/evaluate_argos_v2_streaming.py`

## Semantics

```text
for each sequence:
    state = model.init_state(...)
    previous_frame = None
    for frame in chronological order:
        reliability = current_valid * warp(previous_valid, flow_t_to_prev) * inbounds * not_occluded
        refined, state, diagnostics = model.step(current, previous, flow_t_to_prev, reliability, state)
```

State resets only at sequence start or explicit `mode=state_reset`.

## Supported Models

- `raw`
- `current_only`
- `aligned_local_faithful`
- `aligned_local_safe`
- `faithful_causal_bida`
- `safe_causal_bida`

## Supported Ablations

- `full`
- `current_only`
- `shuffled_history`
- `state_reset`

## Metrics

The compact smoke evaluator reports raw/refined MAE, Bad-3, New-Bad3, modified-pixel ratio, frame count, and temporal-pair count. Full experiment scripts can extend this without changing state semantics.

## What Was Skipped

No batched optimized evaluator was added. The reference evaluator is intentionally simple; add batching only after proving numerical equivalence to this file.
