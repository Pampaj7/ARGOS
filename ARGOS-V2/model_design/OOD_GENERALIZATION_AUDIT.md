# ARGOS v2 frozen OOD generalization audit

## Immutable artifacts

This study is evaluation-only. It loads, without parameter updates:

- detector `results/raw_error_abstention/full/checkpoints/best_validation.pt`
  (SHA256 `78b1bb6cf809dc76448222e41e3bcfafb754bc9b7b6629edcdfa2e1a33444e67`);
- A2 proposal `results/learned_t1_refiner/ablations/A2/checkpoints/best_validation.pt`
  (SHA256 `6cd29277397001333ef3ce630b2f3bc04ec393cdc72e65aa5eb087afd3b389ea`);
- temperature and operating modes
  `results/raw_error_abstention/full/operating_modes.json`
  (SHA256 `791f27d21e3f9fa63fe267d5742c4fb85226f49e6027b285aeb90754fbe10b69`).

The runner also refuses to execute if the canonical BiDA, A2, detector, or
abstention source hashes differ from the audited versions. Their hashes are
recorded in `results/ood_generalization/frozen_manifest.json`.

The frozen balanced mode is temperature 0.7403659225, error probability >=0.50,
expected raw error >=0.25 cache pixels, uncertainty <=2.0 cache pixels, valid
aligned support, and an A2 proposal bounded by 3 cache pixels. Rejection is exact
raw identity. No OOD result is permitted to change these values.

## Reused infrastructure

- BiDA flow, warp, FB and photometric evidence:
  `model_design/external_components/bidavideo.py`.
- A2 and detector: their canonical files under `model_design/models/`.
- CREStereo cache/GT/RGB loading: `TemporalPairDataset`.
- SERV-CT manifest/shards: validated `servct_adapter.py` and
  `build_ood_shards.py` outputs.
- SCARED direct GT: validated `strong_keyframes_rectified_manifest.csv` and
  S2M2-S inference recipe used by `eval_strong_keyframes_s2m2.py`.
- D4D: validated 156 four-frame causal shards produced by
  `run_d4d_context_shards.py`, with curated-pose Zivid GT at the anchor only.
- StereoMIS: validated rectified sequences produced by `scripts/stereomis/convert.py`;
  no dense GT exists.

## Honest protocol boundaries

- **CREStereo:** second unseen backbone on the same frozen SCARED-C keyframes
  3/4 used for the primary-unseen report. This isolates backbone shift.
- **SERV-CT:** 16 ordered CT-reference frames in two eight-frame experiments.
  The validated adapter labels continuity `weak_sparse`; temporal metrics are
  causal-replay diagnostics, not dense-video claims.
- **SCARED structured light:** the 45 anchor images are not frame-exact
  synchronized to the separate temporal video. The only honest frozen-model
  protocol is static repeat (`past=current`), which evaluates direct-geometry
  preservation and abstention but cannot demonstrate temporal improvement.
- **D4D anchors:** geometry is evaluated only where curated Zivid GT is valid.
  Temporal metrics use the three causal transitions inside each validated
  four-frame window. They are window-level diagnostics, not a reconstruction of
  every frame of all 98 curated clips.
- **StereoMIS:** no disparity/depth metrics are reported. Only flow-compensated
  temporal disagreement, intervention/update statistics, support and qualitative
  diagnostics are valid.

All disparity inputs are converted to the frozen model's 144x180 cache grid.
Resizing from another grid scales disparity magnitude by `180/source_width`.
GT uses coverage-normalized resizing of `disparity*valid` divided by resized
valid coverage, with primary coverage 0.50. Raw and refined geometry always use
the exact same paired mask.

For SCARED structured-light and D4D, a secondary report also preserves the
validated dataset evaluation grid: the frozen 144x180 correction is resized
with disparity scaling and added to the untouched raw prediction on that grid.
It is labelled `native-grid from cache-grid correction`, not native ARGOS
inference. This reporting view does not alter authorization or any threshold.

## Predeclared interpretation

Geometry datasets are GO when mean EPE does not worsen materially, catastrophic
tails remain bounded by the structural 3-pixel update, and clean preservation is
substantially safer than historical ARGOS v1. A no-reference temporal dataset is
GO only for safe deployment diagnostics if interventions remain sparse/bounded
and motion-compensated temporal error does not worsen materially; it cannot by
itself establish geometric accuracy.
