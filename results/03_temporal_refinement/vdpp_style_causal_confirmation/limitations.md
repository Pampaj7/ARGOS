# Limitations

- **n=3 seeds per cell.** Enough to reveal the overlap problem, not enough to rule out a real but
  small effect. A robust confirmation would need more seeds (≥8-10) and/or variance-reduction
  (e.g. paired seeds, matched initialization across configs) to shrink the CI below the observed
  seed-to-seed gap (~0.05 in tgm_error, comparable to the between-config gap).
- **shuffled_history under-trained relative to the rest.** Two of its three seeds ran 700-800
  steps instead of 1200 (walltime pressure on p1i's interactive queue), because windowed
  evaluation of this mode is the most expensive (O(clen²) per window). This works against the
  hypothesis being tested (less-trained shuffled model should look worse, not better) but is a
  confound worth flagging — a fully step-matched shuffled run was not achieved in this pass.
- **D4D has no dense temporal ground truth.** All D4D temporal numbers (mc_inconsistency, HF
  energy, boundary MC, etc.) are prediction-space / motion-compensated diagnostics via RAFT optical
  flow, not measurements against real temporal GT. They indicate self-consistency, not accuracy.
- **D4D bootstrap is small-n.** 4 clips/seed with valid RAFT flow (of `--clips-per-specimen 3` × 3
  specimens = 9 requested, some dropped for missing flow/occlusion validity) — bootstrap CIs at
  this n are wide and the "sign stability" check is correspondingly weak evidence, not proof of
  absence of a temporal effect.
- **Only clip length 8 tested**, per the task's explicit scope (clip-length-16 deferred). A
  longer-horizon causal-history advantage, if any, could only appear at longer clips.
- **TGM-weight sweep only covers full_history.** λ was not swept for the ablations, so we can't
  rule out that current_frame_only or shuffled_history would also improve their (non-temporal)
  metrics at lower λ, which would further weaken the "TGM implies temporal usage" argument.
- **Single architecture / single frozen backbone (S2M2-S).** Findings are specific to this 752k
  ConvGRU causal refiner; do not generalize to other temporal architectures without re-running the
  same decoupled ablation.
