# ARGOS v2 Streaming Evaluator Audit

## Summary

The old NVDS-lite evaluator is not a true streaming evaluator. It evaluates non-overlapping `clip_len` windows and resets model history at each window boundary. This is fine for quick diagnostics but invalid for ARGOS v2 temporal conclusions.

## Relevant Files

- `scripts/temporal_refinement/nvds_lite_causal/train_nvds_lite.py::eval_sequences`
  - Bug/limitation: loops `for st in range(0, T, clen)` and calls the model on each window independently.
  - Effect: temporal history is reset every 8 frames by default; temporal pairs at window boundaries are not evaluated as continuous history.
  - Status: left unchanged for historical compatibility.

- `scripts/temporal_refinement/causal_bida/model.py`
  - `FaithfulCausalBiDA` and `SafeCausalBiDA` already expose `init_state`, `detach_state`, `step`, and `forward_sequence`.
  - State resets only when caller requests it.

- `scripts/temporal_refinement/eval_scripts/evaluate_argos_v2_streaming.py`
  - New reference evaluator.
  - Batch-size-1 streaming semantics.
  - Resets only at sequence start or explicit `mode=state_reset`.

## Sequence Handling

The new evaluator loads one sequence at a time and processes frames in chronological array order from the existing sequence shards. State is isolated per sequence. No dataloader chunk, window, or batch boundary can reset state because the reference evaluator does not batch multiple sequence chunks.

## History Modes

- `full`: previous adjacent frame and persistent model state.
- `current_only`: no previous-frame evidence.
- `state_reset`: state is explicitly reset every frame for ablation.
- `shuffled_history`: chooses only from past frames; never future frames. For non-adjacent past samples it uses zero flow rather than inventing unavailable long-range flow.

## Mask and Flow Semantics

- Uses target-to-source flow `flow(t -> t-1)`.
- Warps previous validity into current coordinates before intersection.
- Reliability mask includes current validity, warped previous validity, in-bounds support, finite support, and occlusion if supplied.

## Remaining Caveat

NVDS-lite itself still has a clip-based model API. The new evaluator is ready for the unified `step()` models and raw/current-only adapters. NVDS-lite needs a small adapter if it must be evaluated through this exact streaming API.
