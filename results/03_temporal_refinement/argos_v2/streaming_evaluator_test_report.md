# ARGOS v2 Streaming Evaluator Test Report

## Tests Run

- `py_compile` for causal BiDA and streaming evaluator files: passed.
- `test_shapes.py`: passed.
- `test_aligned_local.py`: passed.
- `evaluate_argos_v2_streaming.py --self-test`: passed.
- Tiny real SCARED smoke on `dataset_1_keyframe_1`: passed.

## Key Results

Synthetic evaluator validation:

- state is not reset at frame 8;
- full stream equals chunked processing when state is externally preserved;
- intentional reset every 8 frames differs from true streaming;
- future perturbation does not change outputs at or before `t`;
- target-to-source flow direction passes translation check;
- warped-validity mask is nontrivial;
- outputs are finite;
- frame count 16 and temporal pair count 15 are correct.

Tiny SCARED smoke:

- raw 16-frame MAE: `1.0110`;
- FaithfulCausalBiDA full 16-frame MAE: `1.0110` at identity init;
- FaithfulCausalBiDA reset 16-frame MAE: `1.0110` at identity init;
- current-only 8-frame MAE: `1.0696`;
- AlignedLocalOnlySafe 8-frame backward smoke: finite, all gradients present.

The identical MAE for random/identity-initialized models is expected and desirable for this certification step.
