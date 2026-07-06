# D4D zero-shot S2M2 + refiner baseline

First model evaluation on the D4D sparse-keyframe benchmark. Raw S2M2-S + SCARED-trained
refiners applied **zero-shot** (no training, no D4D tuning). Establishes the **0%-target-data
baseline** for the few-shot adaptation study.

## Headline
Raw S2M2-S: **MAE 4.09 px, Bad-3 33 %** on D4D keyframes. **Every refiner harms it**
zero-shot (ΔMAE −0.37…−0.92; 9–15 % anchors improved; harmful rate 0.51–0.73), because they
**false-activate on raw-good pixels** and only help the large-error (>6 px) regime they were
trained on. Consistent across all 3 specimens. See `reports/zero_shot_findings.md`.

## Final report (12 points)
1. **Anchors**: 166 usable → 156 evaluated (10 skipped: missing right stereo pair), 137 with
   valid GT overlap for stats. `skipped_anchors.csv`, `context_manifest.csv`.
2. **Raw S2M2-S**: MAE 4.09 px, Bad-1 67.7 %, Bad-3 33.2 %.
3. **Zero-shot refiners**: all degrade — v3.2c 4.46, EGBM-v3-CARE-S 4.66, EGBM-v2-CARE 4.82,
   EGBM-v1 4.81, EGBM-v2 5.01 px. MPC/CPV blocked at D4D odd resolution.
4. **Improve/harm**: all harmful; new-Bad3 14–30 %; only 9–15 % of anchors improve.
5. **Per-specimen**: harm on all three (specimen_1 −0.59, specimen_2 −0.09, specimen_3 −0.85
   ΔMAE). `metrics/specimen_metrics.csv`.
6. **Raw-good preservation**: poor — on raw <1 px pixels ΔMAE −0.71 (damages good regions).
   `metrics/raw_error_bin_metrics.csv`.
7. **Large-error regions**: refiners DO help (>12 px: +1.64; 6–12 px: +0.41) — the only regime
   they pay off, but it is a minority of D4D pixels.
8. **Boundary/SNR**: boundary MAE worse than interior for all refiners (`metrics/`). SNR-strat
   deferred (see limitations).
9. **Runtime/VRAM**: refiners ~6–16 ms/frame; S2M2 context build peak 414 MB VRAM.
10. **Supports adaptation hypothesis**: YES — zero-shot transfer fails consistently → target
    adaptation is necessary.
11. **Next experiment**: few-shot adaptation on D4D session-disjoint splits (already generated),
    curve from this 0 % point; start with EGBM-v3-CARE-S (lowest zero-shot harm) and v3.2c.
12. **Blockers/risks**: MPC/CPV need even-dim handling; 19 zero-overlap anchors excluded;
    D4D is sparse keyframe (not dense temporal). `reports/limitations.md`.

## Reproduce
```bash
# context + S2M2 raw (GPU, p1i):
bsub -I -q p1i -gpu "num=1:mode=shared" -n 4 bash -lc '... run_d4d_context_shards.py --out <OUT>'
# zero-shot eval (GPU):
bsub -I -q p1i -gpu "num=1:mode=shared" -n 4 bash -lc '... evaluate_d4d_zero_shot.py'
```

## Layout
`code_audit.md`, `checkpoint_inventory.csv`, `inference_assumptions.md`, `d4d_index.csv`,
`skipped_anchors.csv`, `context_manifest.csv`, `metrics/` (anchor/aggregate/specimen/session/
quality/convention/raw_error_bin/correction_safety/blocked), `model_comparison.json`,
`aggregate_summary.json`, `reports/{zero_shot_findings,limitations}.md`,
`environment_summary.txt`. Raw prediction arrays (`raw/`, `shards/`) are gitignored
(regenerable).
