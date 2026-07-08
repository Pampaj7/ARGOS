# ARGOS v2 Training State Semantics Audit

- Training samples random clips from training sequences.
- Clip length: 8.
- State starts from zero at every sampled clip.
- State is not carried across clips.
- `detach_state=False` inside the clip, so BPTT horizon is the clip length.
- The model is never trained with persistent state beyond 8 frames.
- Validation selection uses refined MAE, not a temporal consistency objective.

Diagnosis: training encourages short-window aligned local use more than long-horizon persistent memory.
