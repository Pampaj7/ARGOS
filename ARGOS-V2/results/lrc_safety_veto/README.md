# ARGOS v2 — frozen raw-LRC safety-veto calibration

This is a validation-only negative result.  No model was trained and no
final-seen, unseen-backbone, or OOD dataset was loaded.

The policy was exactly `frozen_selector_authorization AND
frame_relative_raw_LRC_at_or_above_quantile`.  It can only close a raw versus
causal-t-1-memory replacement; rejected pixels remain raw bit-exactly and
accepted pixels remain the frozen aligned-memory value bit-exactly.

Calibration used only SCARED-C `dataset_7_keyframe_1/2`, pooled over S2M2-S,
RAFT-Stereo and StereoAnywhere, on the strict common GT/raw/BiDA/LRC support
at coverage `.50`.  The same preregistered frame-relative quantiles `.50`,
`.75`, `.90`, `.95` were evaluated for three frozen selector seeds.

At `.95`, the gain retained was 72.8%, 53.1%, and 19.7% for seeds 0, 1, and 2
respectively.  Thus the apparent seed-0 safety trade-off did not reproduce;
there is no seed-robust policy meeting the predeclared 70%-gain and safety
criteria.  The branch is **NO-GO**.  No test or unseen results are reported
because opening them would violate the frozen-validation protocol.

`seed_{0,1,2}/calibrate_metrics.csv` contains per-backbone, per-sequence and
aggregate metrics for every quantile; `operating_point.json` documents the
per-seed rule outcome and `calibrate_frozen_manifest.json` records hashes.

Reproduction (GPU 1 exposed as `cuda:0`):

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/dtu/p1/leopam/ARGOS/ARGOS-V2:/dtu/p1/leopam/ARGOS/ARGOS-V2/scripts \
  /dtu/p1/leopam/ARGOS/.miniconda/envs/argos/bin/python scripts/run_lrc_safety_veto.py \
  --mode calibrate --selector-output results/utility_memory_selector/seed_0 \
  --output results/lrc_safety_veto/seed_0 --workers 32 --batch-size 8 --device cuda:0
```
