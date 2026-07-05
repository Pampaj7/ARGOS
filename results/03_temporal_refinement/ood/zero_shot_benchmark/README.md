# Zero-shot OOD benchmark — ARGOS temporal stereo refiners

**Scientific question:** do the ARGOS refiners learn *general* stereo-failure correction,
or do they overfit the appearance and error distribution of the primary SCARED-derived
surgical dataset?

**Answer (SERV-CT, zero-shot): they overfit.** Every refiner *improves* disparity slightly
in-domain but *degrades* it out-of-distribution, and the large-correction models (MPC, CPV)
degrade it catastrophically.

## Headline (SERV-CT `honest_test` / Experiment_2, 8 frames, dense GT)

| model | in-domain ΔMAE | OOD ΔMAE | OOD new-Bad3 (raw-good→bad) | OOD harmful rate |
|-------|---------------:|---------:|----------------------------:|-----------------:|
| raw S2M2-S | — (1.28px) | — (1.28px) | 0.0% | 0.00 |
| EGBM-v1 | **+0.085** | **−0.53** | 8.5% | 0.70 |
| EGBM-v2-CARE | +0.123 | −0.60 | 8.5% | 0.68 |
| EGBM-v3-CARE-S | (streaming) | −0.79 | 10.9% | 0.76 |
| EGBM-v2 | +0.091 | −0.75 | 11.2% | 0.74 |
| v3.2c | +0.019 | −0.54 | 13.3% | 0.47 |
| **MPC** | +0.033 | **−5.35** | **44.2%** | 0.83 |
| **CPV** | +0.096 | **−5.06** | **39.3%** | 0.82 |

ΔMAE = raw MAE − refined MAE (positive = improvement). In-domain figures are the models'
own selected SCARED-val metrics (from their checkpoints); OOD figures are this benchmark.

## Why this happens

- **In-domain, S2M2-S raw MAE ≈ 5.2px** (SCARED is hard for the stereo backbone). The
  refiners were trained/selected to make small, safe corrections in that ~5px error regime.
- **OOD, S2M2-S raw MAE ≈ 1.1–1.3px** — on SERV-CT's cleaner rectified pairs the *raw*
  stereo is already excellent. There is almost no error to fix.
- The refiners' detector/gate fired for a 5px world; on SERV-CT it **false-fires on
  already-good pixels** and injects corrections that only add error. MPC/CPV carry a
  large-magnitude proposal head (residual scale 32px) tuned to recover big SCARED errors;
  OOD it **over-authorizes** large corrections (p99 correction ≈ 12.8px) and destroys the
  disparity (new-Bad3 ~44%).

This is direct evidence that the refiners encode the *primary dataset's error statistics*
rather than a general "is this pixel wrong?" prior — and it motivates exactly the
safe-fraction / over-authorization control that Agent A is developing.

## Files

- `protocol.md` — exact zero-shot protocol, fairness rules, feature construction, per-model policy.
- `servct/servct_model_comparison.csv` — full metric battery per model (dataset-level).
- `servct/servct_sequence_metrics.csv` — per-experiment (test vs train).
- `servct/servct_frame_metrics.csv` — per-frame (~60 metrics/frame).
- `servct/safety_metrics.csv` — safety subset (new-Bad, harmful/beneficial, correction magnitude, damage concentration, trust/large-proposal utilisation).
- `servct/in_domain_vs_ood.csv` — the crux comparison.
- `servct/checkpoint_manifest.json` — exact checkpoint + policy per model.
- `servct/environment_summary.txt` — hardware/software/runtime.
- `servct/aggregate_summary.json` — machine-readable summary.
- `servct/final_comparison_table.tex` — paper-ready table.
- `servct/diagnostics/*.png` — comparison plots.
- `servct/qualitative/*.png` — per-frame panels (raw | refined | GT | raw err | refined err | applied residual | beneficial | harmful).

## Reproduce

```bash
# 1. adapt SERV-CT -> sequence layout + manifest
python scripts/temporal_refinement/ood/adapters/servct_adapter.py
# 2. generate pretrained S2M2-S raw disparity (H100)   [B1]
bsub -I -q p1i -gpu "num=1:mode=shared" -n 8 bash -lc '... predict_s2m2_long_sequences.py \
    --sequences-root .../prepared/servct/sequences --out-root .../prepared/servct/s2m2_s512 \
    --variant S --width 512'
# 3. build training-format shards (144x180)
python scripts/temporal_refinement/ood/eval/build_ood_shards.py
# 4. run the benchmark
python scripts/temporal_refinement/ood/eval/evaluate_ood_refiners.py --dataset servct --device cuda
```

## Scope / honesty

- **SERV-CT only** carries dense per-frame disparity GT and is the defensible zero-shot
  dense-disparity target. **D4D is not included** here: its only GT is sparse Zivid
  structured-light (~2 scans/session), with no dense per-frame disparity GT — see
  `../dataset_discovery/missing_requirements.md` (blocker B2). No D4D disparity numbers are
  fabricated.
- SERV-CT temporal continuity is weak/sparse, so streaming/window refiners run in
  causal-replay mode; this is reported, not tuned around.
- **No OOD tuning** of any kind: every model uses its already-selected primary checkpoint,
  threshold, and proposal scale. Only physical unit conversions are applied to the data.
