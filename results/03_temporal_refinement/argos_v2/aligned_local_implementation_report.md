# ARGOS v2 Aligned-Local-Only Implementation Report

## Implemented Classes

- `AlignedLocalOnlyFaithful`
- `AlignedLocalOnlySafe`

Location:

- `scripts/temporal_refinement/causal_bida/model.py`

Exports updated in:

- `scripts/temporal_refinement/causal_bida/__init__.py`

## Interface

Both classes implement the same streaming contract:

```python
init_state(...) -> None
step(current_rgb, current_raw_disparity, previous_rgb, previous_raw_disparity,
     flow_target_to_previous, reliability_mask, state)
    -> refined_disparity, None, diagnostics
```

They are stateless by design. No fake persistent state is created.

## Diagnostics

- `delta`
- `prev_warp`
- `gate` for safe variant
- `proposal_delta` for safe variant

## Tests

Added:

- `scripts/temporal_refinement/causal_bida/tests/test_aligned_local.py`

The tests cover shapes, no hidden state, identity init, residual bound, gate range, target-to-source translation support, gradient coverage, and numerical finiteness.
