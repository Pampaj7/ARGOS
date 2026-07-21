# ARGOS v2 — Multi-domain Raw Error Detector

## Verdict

**NO-GO for the stated hypothesis.** Adding one supervised surgical domain to
SCARED-C does not produce a detector that is both useful and safe on a third
domain. The architecture was unchanged: the 1,107-parameter S1 detector was the
only trainable component; SEA-RAFT, BiDA, A2 and disparity caches remained
frozen.

M1 (`SCARED-C + D4D -> SERV-CT`) is the decisive failure. The frozen detector
worsens SERV-CT EPE by 0.3678 px, with 49.08% false updates and 40.53% clean
degradation. It also retains only 46.4% of the SCARED-C-only authorized gain.

M2 (`SCARED-C + SERV-CT -> D4D`) provides a narrower positive diagnostic: it
reduces D4D false updates from 28.49% to 1.80% and keeps a small +0.0054 px EPE
gain, but intervenes on only 3.31% of pixels. No M2 calibration point satisfied
the predeclared gates, and even the held-out SERV-CT experiment is 0.0014 px
worse than raw. This is conservative abstention, not robust useful domain
generalization. The D4D aggregate is also not repeatable by specimen: gains are
+0.00765 px on specimen 1, -0.00036 on specimen 2 and -0.01269 on specimen 3.

## Cache-grid geometry and safety at coverage 0.50

| Fold / dataset | Raw EPE | Output EPE | Gain | False update | Clean degradation | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 SCARED-C test | 0.961605 | 0.949089 | +0.012516 | 0.88% | 0.31% | 2.33% | 81.77% |
| M1 Fast-FoundationStereo | 0.975741 | 0.965615 | +0.010126 | 0.79% | 0.19% | 2.19% | 85.24% |
| M1 CREStereo | 1.100315 | 1.086211 | +0.014105 | 0.89% | 0.23% | 2.52% | 86.34% |
| M1 D4D specimen 3 | 7.300630 | 7.335185 | -0.034555 | 0.00% | 0.00% | 3.07% | 6.13% |
| **M1 SERV-CT unseen** | **1.046284** | **1.414039** | **-0.367755** | **49.08%** | **40.53%** | **51.02%** | **24.30%** |
| M2 SCARED-C test | 0.961605 | 0.942083 | +0.019522 | 0.25% | 0.20% | 1.97% | 84.20% |
| M2 Fast-FoundationStereo | 0.975741 | 0.958053 | +0.017688 | 0.19% | 0.14% | 1.97% | 86.90% |
| M2 CREStereo | 1.100315 | 1.075199 | +0.025116 | 0.20% | 0.15% | 2.08% | 91.21% |
| M2 SERV-CT seen/calibration | 1.015624 | 1.017059 | -0.001435 | 1.63% | 1.23% | 2.35% | 40.96% |
| **M2 D4D unseen** | **2.905514** | **2.900132** | **+0.005382** | **1.80%** | **1.61%** | **3.31%** | **56.55%** |

Full Bad1, Bad3, boundary EPE, new-Bad3 and frame-tail statistics are in each
fold's `per_dataset.csv`, `per_dataset_backbone.csv`, `sequence_metrics.csv`
`per_specimen.csv` and `safety_summary.json`.

## Genuine supervision and folds

- SCARED-C: existing corrected temporal pseudo-GT; original train,
  calibration keyframes 1/2 and final keyframes 3/4 preserved.
- D4D: curated Zivid structured-light GT only at the current anchor. M1 uses
  specimen 1 train (72), specimen 2 calibration (33), specimen 3 final (51).
  Context frames carry evidence only; GT is never temporally propagated.
- SERV-CT: two CT-derived weak-replay experiments, seven causal pairs each.
  M1 never loads either experiment before freeze. M2 uses Experiment 1 for
  training and Experiment 2 for calibration; this is not dense temporal GT.
- `stereo_depth`, IGEV++ and all stereo predictions are explicitly rejected as
  labels. Fast-FoundationStereo and CREStereo are final-only.

Both ratio ladders selected D1 (75% SCARED-C, 25% added domain); neither D1 nor
D2 produced an eligible calibration point. D1/D2 selection used no fully
unseen-domain or unseen-backbone outcomes.

## Runtime

The detector costs 0.10–0.13 ms/frame. End-to-end cached-stereo ARGOS latency,
including SEA-RAFT/evidence/A2/detector, is 3.34–4.02 ms/frame on H100. Peak
allocated GPU memory during final evaluation is 604 MB; training peak is 646
MB. The trainable parameter count remains 1,107.

## Interpretation and YAGNI stop

M3 and the capacity control were not run. M1 already falsifies the primary
third-domain hypothesis; M2 shows that the attainable benefit is mostly
low-coverage abstention. Scaling capacity would confound domain diversity, and
M3 has no held-out geometry domain. SCARED structured-light and StereoMIS were
not rerun because they do not decide this supervised leave-one-domain-out
question; their validated frozen-pipeline diagnostics remain preserved.

See `model_design/MULTIDOMAIN_TRAINING_AUDIT.md` for the audit and each fold's
README for exact commands and method-level tables.
