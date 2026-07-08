# ARGOS v2 Causal BiDA Implementation Report

## New Source Files

- `scripts/temporal_refinement/causal_bida/__init__.py`
- `scripts/temporal_refinement/causal_bida/official_blocks.py`
- `scripts/temporal_refinement/causal_bida/model.py`
- `scripts/temporal_refinement/causal_bida/tests/test_shapes.py`

## Implemented Models

### `FaithfulCausalBiDA`

- official channel count: `mid_channels=48`;
- official residual block structure;
- official local 3-channel disparity feature extractor adapted to `[prev_warp, current, current]`;
- forward-only hidden propagation;
- true `step()` streaming API;
- `forward_sequence()` wrapper that calls `step()` sequentially;
- optional residual bound.

Trainable params: `489,185`.

### `SafeCausalBiDA`

- inherits faithful causal core;
- adds RGB/reliability-aware gate;
- bounded residual default `3.0` px;
- identity-safe gate bias default `-4.0` and zero residual output init;
- final correction `D_raw + gate * bounded_delta`.

Trainable params: `522,082`.

## Reused Blocks

- `ResidualBlockNoBN` from official `models/core/bidastabilizer.py`.
- `ResidualBlocksWithInputConv` from official `models/core/bidastabilizer.py`.
- `flow_warp` convention from official `models/core/bidastabilizer.py`.

## Adapted Blocks

- official forward propagation block is made persistent/stateful per frame;
- official local feature stack loses future disparity and duplicates current disparity for the third channel;
- official output head is made identity-safe for ARGOS smoke tests;
- safe model adds bounded gate absent from official code.

## Skipped Deliberately

- embedded SEA-RAFT, because ARGOS already has validated cached target-to-source flow;
- Lightning/training infrastructure;
- official datasets/evaluators;
- D4D/SERV-CT evaluation.
