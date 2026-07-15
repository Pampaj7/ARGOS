# ARGOS v2 frozen OOD generalization

This is an evaluation-only study. The promoted balanced detector, A2 bounded
proposal, SEA-RAFT alignment, temperature, and thresholds were frozen before
any OOD result was read. `frozen_manifest.json` records and verifies all seven
artifact/source SHA256 hashes. No optimizer or training path exists in the
runner.

## Headline result

ARGOS v2 generalizes to the second unseen stereo backbone on held-out SCARED-C,
but it does **not** generalize safely across every surgical dataset. SERV-CT and
D4D expose detector calibration failures. StereoMIS updates are sparse and
bounded, but do not improve aggregate no-reference temporal consistency.

| Dataset / protocol | Raw EPE | Refined EPE | Gain `raw-refined` | False update | Clean degradation | Intervention precision | Temporal delta | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| CREStereo, cache grid, 320 transitions | 1.1003 | 1.0625 | +0.0378 | 2.08% | 1.00% | 81.67% | -0.02450 | **GO** |
| SERV-CT, cache grid, 14 weak-sparse transitions | 1.0454 | 1.1015 | -0.0561 | 38.41% | 22.69% | 48.84% | -0.15243* | **NO-GO** |
| SCARED structured-light, native grid, 45 static anchors | 2.9026 | 2.9132 | -0.0106 | 0.71% | 0.68% | 41.63% | n/a | **GO, preservation only** |
| D4D anchors, native grid, 135/156 with paired support | 3.7498 | 3.7004 | +0.0494 | 29.15% | 17.95% | 61.39% | n/a | **NO-GO** |
| D4D temporal, cache grid, 468 transitions | n/a | n/a | n/a | n/a | n/a | n/a | -0.10682 | **NO-GO with anchor safety** |
| StereoMIS, cache grid, 38,238 transitions, no GT | n/a | n/a | n/a | n/a | n/a | n/a | +0.00149 | **NO-GO** |

`Temporal delta = refined motion-compensated temporal error - raw`; negative is
better. *SERV-CT continuity is `weak_sparse`, so its temporal delta is not a
dense-video claim. EPE magnitudes across cache and native grids are not directly
comparable.

## Dataset conclusions

### CREStereo — GO

Both held-out sequences improve. EPE drops 3.43%, Bad3 goes 7.11% to 6.92%,
boundary EPE goes 2.7890 to 2.6756, temporal error improves, intervention
precision is 81.67%, and only 1.25% of frames worsen. This is strong evidence
for backbone transfer within the SCARED-C domain.

### SERV-CT — NO-GO

Both eight-frame experiments fail. EPE rises 5.37%, Bad3 rises 4.72% to 5.11%,
false-update reaches 38.41%, and the worst frame degrades by 0.2163 px. The
temporal metric improves slightly, but sparse frame spacing and worse geometry
show that this cannot be interpreted as safe refinement. It is nevertheless far
less destructive than the historical ARGOS v1 SERV-CT result (~6.3–6.6 px).

### SCARED structured-light — GO for preservation only

These 45 direct-GT keyframes have no reliable synchronized predecessor. The
honest protocol is therefore `past=current`. Native-grid EPE worsens by 0.0106
px (0.37%), false-update and clean degradation stay below 1%, and there is no
update beyond the cache-grid structural bound after scaling. The output is
approximately preserved, but not exactly: 62.22% of frame means worsen by tiny
amounts, the worst frame is +0.0918 px, and the maximum local native-grid update
is 15.58 px. This is not evidence of temporal gain.

### D4D anchors and temporal windows — NO-GO

Mean native-grid EPE improves 1.32%, but safety fails: Bad3 rises 23.58% to
24.21%, new-Bad3 is 1.90%, false-update is 29.15%, clean degradation is 17.95%,
and the worst frame degrades by 2.55 px. The gain is concentrated in specimen_1;
specimen_2 and specimen_3 worsen. The four-frame temporal windows reduce
motion-compensated disagreement by 0.1068 px, but the anchor GT demonstrates
that this temporal smoothing is not a safe geometric transfer.

### StereoMIS — NO-GO for temporal improvement

Only no-reference metrics are valid. All 38,238 transitions were evaluated:

| Sequence | Transitions | Intervention | Raw temporal | Refined temporal | Delta |
|---|---:|---:|---:|---:|---:|
| P1 | 18,419 | 2.20% | 0.14140 | 0.14089 | -0.00051 |
| P2_8 | 9,155 | 2.41% | 0.02277 | 0.02508 | +0.00231 |
| P3 | 10,664 | 3.28% | 0.05304 | 0.05728 | +0.00424 |
| Aggregate | 38,238 | 2.55% | 0.08835 | 0.08984 | +0.00149 |

Updates remain sparse and below 3 cache pixels, but two of three sequences and
the aggregate worsen. Contact sheets in `stereomis/P3/diagnostics/` are ordered
RGB | raw disparity | refined disparity | absolute update; they are diagnostics,
not decision evidence.

## Failure modes

1. **Cross-dataset calibration shift.** The frozen detector authorizes too much
   on SERV-CT and D4D despite acceptable warp support.
2. **Temporal consistency is not geometric correctness.** SERV-CT and D4D can
   become temporally smoother while clean-pixel safety or Bad3 worsens.
3. **Clean-region false positives.** D4D and SERV-CT reproduce the core failure
   mode at lower severity than ARGOS v1.
4. **Static preservation is not exact.** Structured-light static repeats reveal
   rare but large local corrections even though aggregate clean safety is good.
5. **Cross-sequence temporal transfer is inconsistent.** StereoMIS P1 improves
   marginally, while P2_8 and P3 worsen.

## Runtime and memory

Observed synchronized per-frame medians under shared-GPU contention are 18.8 ms
for CREStereo ARGOS-only, 27.0 ms for D4D temporal ARGOS-only, and on StereoMIS
69.9 ms for S2M2-S plus 20.3 ms for ARGOS v2. CUDA process-local peak allocation
is 94.0 MiB for the frozen refiner path and 429.4 MiB for S2M2-S plus refiner.
See `runtime_summary.json`; these are not dedicated-H100 throughput claims.

## Interpretation and paper readiness

The completed experiment is **ready to report in a paper** because it supplies
controlled positive and negative OOD evidence with frozen hashes and no tuning.
The architecture is **not ready for a broad OOD-safety or deployment claim**.
The defensible claim is narrower: ARGOS v2 transfers across a second unseen
backbone within SCARED-C and can approximately preserve static structured-light
predictions, but cross-dataset authorization remains unsolved.

## Reproduction

```bash
PY=/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python
cd /dtu/p1/leopam/ARGOS/ARGOS-V2

CUDA_VISIBLE_DEVICES=0 $PY scripts/run_ood_generalization.py --dataset crestereo --output results/ood_generalization --device cuda:0
CUDA_VISIBLE_DEVICES=0 $PY scripts/run_ood_generalization.py --dataset servct --output results/ood_generalization --device cuda:0
CUDA_VISIBLE_DEVICES=0 $PY scripts/run_ood_generalization.py --dataset scared_sl --output results/ood_generalization --device cuda:0
CUDA_VISIBLE_DEVICES=0 $PY scripts/run_ood_generalization.py --dataset d4d --output results/ood_generalization --device cuda:0
CUDA_VISIBLE_DEVICES=0 $PY scripts/run_ood_generalization.py --dataset stereomis --output results/ood_generalization --device cuda:0 --stereomis-sequences P1 P2_8 P3
CUDA_VISIBLE_DEVICES=0 $PY scripts/run_ood_generalization.py --dataset aggregate --output results/ood_generalization

$PY -m pytest -q model_design/tests/test_abstention.py model_design/tests/test_raw_error_detector.py model_design/tests/test_bidavideo.py
```

## Files

- `aggregate_summary.json`: machine-readable full aggregate.
- `verdicts.json`: per-dataset and overall decisions.
- `protocol_manifest.json`: exact sample counts and grids.
- `frozen_manifest.json`: checkpoint, threshold, and source hashes.
- `runtime_summary.json`: latency quantiles and GPU allocation.
- Each dataset directory contains `frame_metrics.csv`, `sequence_metrics.csv`,
  `summary.json`, and `README.md`.
- Per-run logs are preserved as `run_*.log`.
