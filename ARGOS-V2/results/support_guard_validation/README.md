# ARGOS v2 Support Guard Validation

Frozen D1 validation of `a_final = a_error AND a_support`. No neural module was trained and no OOD dataset participated in fitting or threshold selection.

## Frozen policy

- representation: 24-channel Raw Error Detector penultimate feature;
- method: `knn`;
- granularity: `frame`;
- calibration acceptance quantile: `0.9`;
- threshold: `0.6312034726142884`;
- G2/G3 correlation: `0.7550181188568272`; G4: `skipped by YAGNI: G2/G3 were not materially complementary`.

## Verdict

**NO-GO**

The guard detects the gross SERV-CT and D4D shifts, but it is not stable enough
within SCARED-C and does not preserve the successful unseen-backbone transfer.
It therefore passes cross-domain fail-closed safety only by rejecting all
interventions on SERV-CT and D4D.

| Dataset | Raw EPE | No-guard EPE | Guard EPE | Support accept | Authorization retained | A2 gain retained | False update | Clean degradation | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SCARED-C test | 0.961604 | 0.934655 | 0.956002 | 65.98% | 36.40% | 20.79% | 1.51% | 0.65% | NO-GO |
| Fast-FoundationStereo | 0.975739 | 0.952306 | 0.969825 | 66.30% | 34.13% | 25.24% | 1.40% | 0.54% | NO-GO |
| CREStereo | 1.100314 | 1.062523 | 1.093518 | 65.67% | 30.35% | 17.98% | 1.53% | 0.58% | NO-GO |
| SERV-CT | 1.045386 | 1.101520 | 1.045386 | 0.00% | 0.00% | n/a | 0.00% | 0.00% | GO (fail closed) |
| D4D | 2.872306 | 2.820551 | 2.872306 | 0.00% | 0.00% | n/a | 0.00% | 0.00% | GO (fail closed) |
| SCARED structured-light | 0.348767 | 0.350326 | 0.350272 | 95.55% | 90.22% | n/a | 0.52% | 0.43% | GO (identity tolerance) |
| StereoMIS | n/a | n/a | n/a | 91.46% | 54.22% | n/a | n/a | n/a | diagnostic only |

On the SCARED-C calibration sequences the same frozen operating point accepted
98.01% of support, retained 98.07% of authorizations and 98.75% of A2 EPE gain.
On the untouched SCARED-C test sequences support acceptance fell to 65.98% and
gain retention to 20.79%. The shift is especially concentrated in
`dataset_7_keyframe_3`; keyframe 4 is fully accepted. This sequence sensitivity
is the decisive failure mode.

SERV-CT and D4D become bit-exact raw, eliminating their earlier false-update
and clean-degradation failures. That is useful evidence that feature support
contains a gross domain-shift signal, but it is not a promoted solution because
zero coverage is explicitly disallowed and Fast-FoundationStereo/CREStereo
gain retention is far below the 60--70% criterion. StereoMIS has no GT; its
motion-compensated temporal error changes from 0.064015 raw to 0.070684 guarded,
so no geometric or temporal benefit is claimed.

Exact frame, sequence, backbone and method results are in `frame_metrics.csv`,
`sequence_metrics.csv`, `per_backbone.csv`, `per_dataset.csv` and
`method_comparison.csv`. `verdicts.json` contains the predeclared decisions.

## Leakage and freeze checks

- support statistics: SCARED-C training only, 32,370 balanced vectors;
- threshold/method selection: `dataset_7_keyframe_1/2` only;
- frozen evaluation: `dataset_7_keyframe_3/4`, then unseen/OOD datasets;
- reference: 24 detector-penultimate channels, 4,096-vector bank, k=5;
- no OOD dataset or unseen backbone entered fitting or selection;
- A2, detector, abstention policy, SEA-RAFT and BiDA hashes match the frozen artifacts;
- no training or parameter update was run.

## Reproduction

```bash
/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python -m pytest -q model_design/tests/test_support_guard.py
/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python scripts/run_support_guard_validation.py --smoke --output results/support_guard_smoke --device cuda:0
rm -rf results/support_guard_smoke
/dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python scripts/run_support_guard_validation.py --output results/support_guard_validation --device cuda:0 --max-train-pairs 32 --max-calibration-pairs 160 --max-test-pairs 160 --fit-pixels-per-status-frame 16 --fit-pixels-per-group 512 --calibration-score-pixels-frame 64 --calibration-feature-pixels-frame 16 --bank-size 4096 --knn-k 5 --d4d-windows 156 --stereomis-samples-per-sequence 128
```

## Runtime

Median isolated support-score latency: 0.5768 ms; compact reference memory: 395712 bytes.

All rejected pixels use `torch.where` and are bit-exact raw. StereoMIS results are no-reference diagnostics only.
