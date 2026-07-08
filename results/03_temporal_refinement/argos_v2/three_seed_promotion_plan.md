# ARGOS v2 Three-Seed Promotion Plan

## Promote Now
- `aligned_local_faithful`: yes, as the clean alignment baseline. It improves over raw/current-only with the smallest model.
- `faithful_causal_bida`: conditional. It improves aggregate MAE, but persistent state is not confirmed because state-reset is tied/slightly better.

## Do Not Promote Yet
- `current_only`: identity baseline only.
- `safe_causal_bida`: identity collapse under current settings.
- `faithful_causal_bida_state_reset` and `faithful_causal_bida_shuffled_history`: evaluation modes, not separately trained models.

## Minimal Next Action
Before a three-seed matrix, inspect why FaithfulCausalBiDA's persistent state does not beat state reset. The next run should include hidden-state diagnostics and possibly a small ablation where the local aligned path is held constant while state contribution is isolated.

## Three-Seed Candidate Set
If proceeding without architecture changes, run only:
1. raw S2M2 eval baseline
2. aligned_local_faithful
3. faithful_causal_bida + state-reset/shuffled eval modes

SafeCausalBiDA should wait for a targeted identity-collapse fix.
