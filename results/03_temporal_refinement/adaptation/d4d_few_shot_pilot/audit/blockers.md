# Blockers / risks

1. **Anchor starvation after frozen-overlap drop**: 1/2-session splits hold 0–2 train anchors.
   Runs execute (pixel-level supervision) but variance will be high; 1session_seed1 skipped (0).
2. **MPC/CPV excluded** (per task; also blocked at odd D4D grid).
3. **Frozen-test zero-shot baseline** differs numerically from the full-set zero-shot report
   (30-anchor test subset vs 137) — both reported, not mixed.
4. Combined-score weights (0.02, 0.5) predefined before test evaluation; not tuned afterwards.
