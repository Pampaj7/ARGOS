# ARGOS v2 bounded-memory validation

The canonical CODD-style BiDA fusion remains positive only as a bounded-horizon adapter. Continuous streaming was already shown to collapse. This study trained one canonical run each for no-recurrence and no-learned-stereo-evidence, selected reset/hard policies on dataset 2 only, and opened dataset 7 once after freezing.

The validation-selected adaptive hybrid H=8 improved validation EPE but did not satisfy the frozen safety targets on dataset 7. Fixed H=4 remains the defensible operational policy; it is a short-window refiner, not indefinite recurrent memory. Hard endpoint output is not promoted because it increases harmful and clean-pixel updates. The no-learned-stereo-evidence ablation unexpectedly exceeds the full reference in this canonical run, so learned ResNet matching cues are not established as essential.

See `ablation_summary.csv`, `fixed_horizon_summary.csv`, `adaptive_reset_validation.csv`, `drift_by_age.csv`, and `aggregate_summary.json`.
