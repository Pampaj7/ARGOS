# Missing requirements / blockers (precise, not fabricated)

Ordered by impact on the zero-shot OOD benchmark. Nothing below is invented — each item
is what the on-disk data does *not* currently provide.

## B1 — S2M2-S raw disparity absent for BOTH OOD datasets  (hard dependency, tractable)

The refiners correct **pretrained S2M2-S @ 512** raw disparity. No such raw disparity
`.npy` exists for SERV-CT or D4D. A search for cached S2M2 disparities found only the
SERV-CT *finetuned* checkpoint (`results/01_frame_stereo/SERVCT/servct_s2m2_honest_finetune_gpu/finetune_refiners_250/s2m2_servct_finetuned.pth`)
and montage PNGs — **not** usable (finetuning contaminates the zero-shot upstream, and no
per-frame arrays were saved).

- **Required**: run pretrained S2M2-S on every OOD left/right pair, save per-frame float16
  disparity in original image coordinates (mirroring the training cache).
- **Reuse**: `scripts/s2m2/eval_servct_s2m2.py` already loads S2M2 (`s2m2.core.model.s2m2`)
  with the `S` config and does crop/pad; adapt its inference (not its metrics) to dump
  disparity arrays. GPU: one H100, 16 SERV-CT frames is seconds; D4D is ~12.7k frames.
- **Status**: this is *upstream inference*, allowed and not tuning. It is the first thing
  Phase 2 must produce.

## B2 — D4D has NO dense per-frame GT disparity  (major, blocks D4D dense benchmark)

D4D GT is Zivid structured-light at ~2 scan timepoints per session (`depth_images/`,
`pointcloud/`), not per-frame. Turning it into per-frame disparity GT requires:

1. Load Zivid pointcloud (metric, in scanner frame).
2. Transform to each left-camera frame using `tf/` + `curated_camera_pose_{start,end}`.
3. Project to the rectified left image plane (`P_left`), z-buffer, get sparse depth.
4. `disp = fx·baseline / depth`; build validity mask (projected + non-occluded).
5. Handle non-rigid scene motion between the scan timepoint and each video frame — Zivid
   scans are static captures; intervening frames deform. GT is only trustworthy at/near
   the scan instants.

Consequence: a *dense per-frame* zero-shot disparity benchmark on D4D is **not honestly
available**. Options, in order of integrity:
- (a) Restrict D4D GT eval to frames at/adjacent to Zivid scan instants (few frames/session).
- (b) Use D4D only for **temporal-stability / self-consistency** OOD diagnostics (no GT):
  flicker, temporal MAE vs raw, correction magnitude distribution shift.
- (c) Defer D4D dense GT to a dedicated conversion effort.

This report does **not** manufacture D4D disparity GT. The D4D adapter is built
non-destructively and stops at the documented blocker.

## B3 — SERV-CT temporal continuity is weak/sparse  (scoping, not blocking)

Streaming and 4-frame-window refiners were designed for smooth video. SERV-CT gives 8
ordered-but-sparse pairs per experiment. For a fair zero-shot run these models fall back
to **causal replay** (current frame repeated across the window / streaming state reset per
frame). This is reported explicitly as a degenerate-temporal condition; it is *not* an
excuse to switch checkpoints or thresholds. Single-frame-capable refiners are unaffected.

## B4 — D4D specimens 3–5 not extracted; specimen_2 partial

`specimen_3/4/5` remain `.tar.gz` under `dataset/D4D/raw/source/`; `specimen_2` extracted
to a non-standard inner layout. Only `specimen_1` (16 sessions, ~12.7k frames) is fully
usable. Extraction is a prerequisite before any D4D scale-up (independent of B2).

## B5 — Agent A's final safe-fraction model not yet available

The benchmark's model registry reserves a slot for Agent A's final safe-fraction
checkpoint. It is intentionally decoupled: the registry loads all *currently available*
refiners now, and Agent A's model can be added later without rewriting the harness. Not a
blocker for the zero-shot run on existing models.

## Summary

| id | blocker | blocks | tractable now? |
|----|---------|--------|----------------|
| B1 | S2M2-S raw disparity not generated OOD | everything | yes (GPU inference) |
| B2 | D4D no dense per-frame GT disparity | D4D dense metrics | no (needs conversion effort) |
| B3 | SERV-CT weak temporal continuity | streaming fidelity | scoping only |
| B4 | D4D specimens 3–5 not extracted | D4D scale-up | yes (extraction) |
| B5 | Agent A model pending | one registry slot | non-blocking |

**Path of least fabrication**: build SERV-CT into a defensible dense zero-shot disparity
benchmark (after B1); use D4D for GT-free temporal OOD diagnostics until B2 is resolved.
