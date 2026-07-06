# Methodology

**Loss** (per 8-frame clip, all valid frames):
  L = L1(refined,gt|valid) + 0.5·|residual| on raw-good(<1px)   [spatial + safety]
  + 1.0·L_tgm   (tgm mode only),  L_tgm = mean_t |(Dref_t−Dref_{t−1})−(Dgt_t−Dgt_{t−1})|.
TGM weight 1.0 predefined. Residual bounded (3·tanh). AdamW lr 3e-4, batch 6, 1500 steps,
mixed precision, grad-clip 1.0, seed 0. Checkpoint selected on frozen val combined score
(MAE + 0.02·newBad3 + 0.5·harmful). Reported on frozen SCARED test.

**Ablations**: current_frame (no temporal memory) and shuffled (broken temporal order) share
the identical architecture/budget. If tgm ≈ current_frame ≈ shuffled, the model is not using time.

**SCARED temporal metrics (with GT)**: tgm_error (|Δrefined−Δgt|), temporal error jitter
(|e_t−e_{t−1}|), HF error energy (2nd temporal diff of error), boundary tgm.

**D4D transfer**: zero-shot; sparse Zivid-anchor MAE + prediction-space motion-compensated
temporal diagnostics (RAFT), labelled diagnostic (no dense temporal GT). SERV-CT: static
geometric safety only (no training there).

**Decision gate**: continue only if tgm beats spatial-only on SCARED temporal AND beats
current_frame/shuffled AND no major spatial/safety loss AND ≥1 reproducible D4D temporal
improvement over raw AND does not collapse to identity.
