# OOD Dataset Discovery — Phase 1 (Agent B)

Zero-shot out-of-distribution evaluation of ARGOS temporal stereo refiners.
Scientific question: **do the refiners learn general stereo-failure correction, or
overfit the appearance/error distribution of the primary SCARED-derived surgical
data?**

This directory is the Phase-1 discovery report. It is **read-only** w.r.t. the raw
datasets; nothing here modifies source data, models, or checkpoints.

## Regenerate

```bash
.miniconda/envs/argos/bin/python \
  scripts/temporal_refinement/ood/adapters/discover_ood_datasets.py
```

## Files

| file | content |
|------|---------|
| `discovered_datasets.json` | machine-readable summary of both datasets + headline verdicts |
| `servct_inventory.csv` | per-frame SERV-CT inventory (calib, GT ranges, valid %) |
| `d4d_inventory.csv` | per-session D4D inventory (frame counts, GT type, calib) |
| `candidate_sequence_manifest.csv` | standardized sequence list + zero-shot usability flags |
| `format_notes.md` | exact formats, units, conventions, alignment per dataset |
| `missing_requirements.md` | precise blockers; what is absent / must be produced |

## Headline findings

**Refiner upstream = pretrained S2M2-S @ 512.** The refiners are trained to correct
S2M2-S raw disparity (original image disparity coordinates, float16). A fair zero-shot
OOD benchmark therefore needs S2M2-S raw disparity on each OOD dataset produced with
the *same pretrained* checkpoint (no OOD finetuning). **This raw disparity does not yet
exist for either OOD dataset** and is the first hard dependency (GPU inference).

### SERV-CT — usable after raw-disp generation
- 2 experiments, 8 frames each (16 total): `honest_test/Experiment_2` (009–016),
  `honest_train/Experiment_1` (001–008). Already converted to ARGOS format under
  `dataset/SERVCT/argos/servct_argos/`.
- **Dense GT**: `disp_gt.npy` (px), `depth_gt_mm.npy` (mm), `valid_mask.npy`, per-frame
  `calib.json`. Disparity is **positive px, left-reference, rectified** — matches the
  ARGOS/S2M2 convention. No conversion of GT convention needed.
- **Temporal continuity is WEAK/SPARSE**: each Experiment is an ordered set of stereo
  pairs along an endoscopic path, not a smooth video. Streaming/window refiners have no
  real 5-frame temporal support here → they must run in *causal-replay* (frame repeated)
  mode, which is a degenerate temporal case. Single-frame refiners are unaffected.
- **Verdict**: viable zero-shot dense-disparity OOD benchmark. Use `honest_test`
  (Experiment_2) as the held-out zero-shot set; treat `honest_train` as an additional
  OOD holdout (never used for tuning).

### D4D — NOT ready for a rigorous dense-disparity benchmark
- specimen_1 fully extracted: 16 sessions, ~12.7k stereo frames, 43 clips (true temporal
  video). specimen_2 partially extracted; specimen_3/4/5 still `.tar.gz` (not extracted).
- **No dense per-frame GT disparity.** GT is Zivid structured-light (`depth_images/`,
  `pointcloud/`) captured at ~2 scan timepoints per session — sparse in time, and
  requires pointcloud→depth reprojection + per-frame camera pose to become per-frame
  disparity GT. Raw left/right images are unrectified (rectification params present but
  not applied).
- **Verdict**: excellent for temporal continuity but blocked on GT. A dense zero-shot
  disparity benchmark on D4D needs a substantial, physically-careful Zivid-reprojection
  conversion (documented in `missing_requirements.md`). Not fabricated here.

## Consequence for the benchmark

The immediately-defensible zero-shot dense-disparity OOD benchmark is **SERV-CT**.
D4D is retained as a temporal-video OOD target pending GT conversion; its adapter is
built to be non-destructive and its blockers are documented rather than papered over.
No OOD-specific tuning, thresholds, clipping, or calibration is applied anywhere.
