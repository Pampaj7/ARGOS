# ARGOS v2 Aligned-Local-Only Test Report

## Command

```bash
.miniconda/envs/argos/bin/python scripts/temporal_refinement/causal_bida/tests/test_aligned_local.py
```

## Result

```text
AlignedLocalOnlyFaithful params 239393
AlignedLocalOnlySafe params 272290
aligned_local_tests=PASS
```

## Tiny SCARED Smoke

`AlignedLocalOnlySafe` on `dataset_1_keyframe_1`, 8 frames:

```text
loss: 1.069632649421692
finite: True
missing_grad: []
params: 272290
```

No training was run.
