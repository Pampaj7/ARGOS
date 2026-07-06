# Code audit — D4D few-shot pilot

- **Trainer**: `scripts/temporal_refinement/adaptation/train_d4d_few_shot_adapter.py` (canonical,
  one path for both models and all modes; CLI `--model --adaptation-mode --split --seed --epochs
  --lr --max-updates --dry-run`).
- **Reuse**: shards + index from the zero-shot pipeline (`d4d_s2m2_zero_shot/{shards,d4d_index.csv}`),
  `FullFrameDataset` 16-ch features, `model_registry` checkpoints (unchanged selection),
  `frame_metrics` for eval. No new architecture.
- **Checkpoints**: v3.2c `tiny_refiner_v3_2c…/best.pt` (threshold 0.7, soft p_bad·residual during
  training — its original convention; hard threshold at inference). EGBM-v3-CARE-S
  `egbm_v3_care_streaming/best.pt` (internal gate, differentiable end-to-end).
- **Loss** (predefined): `L1(refined,gt|valid) + 1.0·|applied| on raw-good(<1px) + 0.05·|applied|`
  — targets the false-activation mechanism (raw-good preservation + identity preference) while
  keeping geometric fit. Selection score (predefined, val only):
  `MAE + 0.02·newBad3% + 0.5·harmful_rate`.
- **Grad isolation**: verified bitwise in-run (`frozen_param_drift` recorded per run; smoke = []).
- **Budget**: ≤50 epochs, patience 8 (combined score), AdamW, lr 1e-3/3e-4/3e-5/3e-4 for
  calib/head/full/scratch, clip 1.0, batch 4 full frames, deterministic seeds. fp32 (data tiny;
  AMP unnecessary at this scale — noted).
