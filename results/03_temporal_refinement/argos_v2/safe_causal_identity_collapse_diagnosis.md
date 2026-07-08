# ARGOS v2 Safe Causal BiDA Identity Collapse Diagnosis

identity collapse: modified_pixel_ratio remained 0.0 at every validation checkpoint after step 200.

- Gate bias: `-4.0` in `SafeCausalBiDA`.
- Residual head: zero-initialized through the faithful base.
- Safe losses enabled: safe=0.2, sparse=0.02.
- Final val MAE: 5.9741; modified ratio: 0.0000.

Primary cause: closed gate plus zero residual initialization, reinforced by safe/sparse losses and validation selection on MAE. This is a training/init problem, not evidence that safety gating is impossible.
