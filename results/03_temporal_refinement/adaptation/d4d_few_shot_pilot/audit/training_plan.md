# Training plan + frozen-eval protocol

**Frozen eval**: ALL runs select on `session_disjoint/validation` (22 anchors) and report on
`session_disjoint/test` (30 anchors) — identical across sizes/modes/models.

**Train data**: sessions from the existing `few_shot/<k>session_seed<s>/train.csv` MINUS any
session in the frozen val/test (drops in `split_inventory.csv`). Anchor counts after filtering
are SMALL (1-session: 0–2; 2-session: 1–2; 4-session: 3–12; 8-session: 28–31) — supervision is
pixel-level (~40k valid px/anchor) so calibration-scale training remains feasible; reported
honestly rather than replaced.

**Matrix**: models {v3.2c, EGBM-v3-CARE-S} × modes {calibration_only, head_only, full} ×
splits {1,2,4,8}session × seeds {0,1,2} + scratch at {4,8}session × seeds + zero-shot baselines.
1session_seed1 has 0 usable anchors → recorded skip.

**Decision gate** (Phase 4): EGBM-v3-CARE-S on 4session_seed1 (11 anchors; substitutes the
starved 2session_seed0 = 1 anchor — documented). Gate passes if ≥1 pretrained mode reduces
raw-good damage, retains >6px gain, and beats zero-shot combined score on the frozen val.
