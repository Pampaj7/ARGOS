# D4D few-shot adaptation pilot

Does the SCARED-trained refiner's correction skill transfer to D4D with light target
calibration? Tests calibration-only / head-only / full / from-scratch at 1/2/4/8 sessions,
against the zero-shot 0%-target baseline. **No backbone change; S2M2 frozen throughout.**

## Result (EGBM-v3-CARE-S, frozen session_disjoint test, raw MAE 3.50)
- **Calibration-only (10.9k params, 0.25%) fixes the domain-shift harm** with 1–2 sessions:
  raw-good damage Δ(<1px) −1.44 → −0.05, new-Bad3 23.5 → ~0, selectivity −0.58 → +0.09.
- **But no mode clearly beats raw MAE** at ≤8 sessions — adaptation makes the refiner **safe**,
  not yet **beneficial**. Large-error skill is suppressed by calibration; only **full**
  fine-tuning retains it (+0.58 gain, selectivity +0.35) but overfits at ~11 anchors.
- v3.2c is near-identity; adaptation just confirms safe abstention.

Full analysis: `reports/findings.md`, `reports/zero_shot_vs_adapted.md`,
`reports/decision_gate.json` (PASSED), `reports/methodology.md`.

## Layout
`audit/` (code_audit, split_inventory, model_parameter_groups, training_plan, blockers),
`runs/<run_id>/` (config.json, best_combined.pt, test_metrics.csv, raw_error_bin, train_log),
`aggregate/` (aggregate/safety/parameter_efficiency/sample_efficiency/raw_error_bin/runtime),
`figures/` (6 plots vs sessions), `run_manifest.csv`, `reports/`.

## Reproduce
```bash
python scripts/temporal_refinement/adaptation/run_d4d_few_shot_matrix.py    # resumable
python scripts/temporal_refinement/adaptation/summarize_d4d_few_shot.py     # tables + plots
# single run:
python scripts/temporal_refinement/adaptation/train_d4d_few_shot_adapter.py \
   --model EGBM-v3-CARE-S --adaptation-mode calibration_only --split 4session_seed1 --seed 0
```

## Recommendation for the paper sweep
EGBM-v3-CARE-S, calibration-only vs full, 8/16/all sessions, with a selectivity-targeted
objective (up-weight >6px correction) to convert safety into net benefit over raw.
