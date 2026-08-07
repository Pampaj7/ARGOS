# ARGOS v2 — scientific/protocol review A

Reviewed before full training on 2026-08-05.

CRITICAL ISSUES: 0

- Frozen geometry_v1 remains unchanged: 60,739 parameters, 17 ordered evidence channels, direct current-to-anchor SEA-RAFT, raw CS1/2/4/8, no recurrence/writeback/critic/gate.
- The only scientific variable is total optimization budget: 10/30/60 epochs and 39,730/119,190/238,380 canonical optimizer steps.
- Train IDs are 1/3/6; validation ID is 2; the runtime guard rejects ID 7 and `dataset_7` paths.
- S2M2-S, RAFT-Stereo, and StereoAnywhere are the fixed training backbones. The canonical unbalanced-by-design group sampler is preserved exactly.
- Effective batch size 12, AdamW (lr 0.002, weight decay 0.0001), loss code, AMP, clip norm 5, crop generation, validation cadence, and checkpoint-selection grid match the validated runner.
- CosineAnnealingLR is stepped per optimizer update; T_max is stretched to the complete paired budget. There is no warmup.
- Scratch initialization reproduces the canonical post-dependency-construction RNG state per seed; the canonical trained checkpoint is not used to initialize runs.
- Full runs assert batch size 12, flow batch 32, 3,973 steps/epoch, registered output directory, and full non-truncated data.
- Budget selection is preregistered and uses D2 only. D7 is unavailable until a frozen recipe and explicit unlock exist.
- Validation bank omission of unused H4/GT-memory tensors does not alter training losses or the canonical per-run selection metric; exact initialization states separately preserve construction-time RNG provenance.

Result: PASS for full-launch protocol.
