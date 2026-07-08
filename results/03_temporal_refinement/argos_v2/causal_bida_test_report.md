# ARGOS v2 Causal BiDA Test Report

## Commands Run

```bash
.miniconda/envs/argos/bin/python -m py_compile   scripts/temporal_refinement/causal_bida/official_blocks.py   scripts/temporal_refinement/causal_bida/model.py   scripts/temporal_refinement/causal_bida/tests/test_shapes.py

.miniconda/envs/argos/bin/python scripts/temporal_refinement/causal_bida/tests/test_shapes.py
```

Synthetic result:

```text
FaithfulCausalBiDA params 489185
SafeCausalBiDA params 522082
causal_bida_synthetic_tests=PASS
```

Tiny SCARED smoke:

```text
seq: dataset_1_keyframe_1
frames: 8
out_shape: (1, 8, 1, 256, 320)
loss: 1.069632649421692
missing_grad: []
finite: True
```

## Covered

- shape correctness;
- parameter counts;
- identity initialisation;
- future perturbation causal leakage on synthetic sequence;
- state reset/init path;
- state-effect path by perturbing hidden state;
- target-to-source `flow_warp` use in sequence wrapper;
- finite outputs;
- residual bound for safe model;
- gate range by sigmoid construction;
- all trainable parameters receive finite gradients in synthetic and SCARED tiny smoke.

## Not Covered Yet

- official checkpoint equivalence;
- long streaming evaluation;
- training stability;
- D4D/SERV-CT;
- offline bidirectional official upper bound.
