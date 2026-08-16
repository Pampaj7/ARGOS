# External comparison methods — acquisition status

Checked 2026-08-16 during the overnight run. Ordered from closest to our
setting (causal, black-box, frozen stereo) to furthest.

## 1. StereoDiffusion (MICCAI 2024) — NOT RUNNABLE, verified
The only published method in our exact category: causal, black-box, frozen
stereo, surgical, evaluated on SCARED.
Repo github.com/xuhaozheng/StereoDiff cloned: ONE commit ("Initial commit"),
7.43 KiB of git objects, contents are .gitignore + LICENSE + README.md (12
bytes). No code, no weights, no other branch (`git ls-remote --heads` shows
only main).
=> Citable as: the only comparable method has not released an implementation.

## 2. Neural Disparity Refinement (3DV 2021 / TPAMI 2024) — WEIGHTS BLOCKED
Same input contract as ours (RGB + black-box disparity), but spatial-only.
Code cloned fine. Weights are on Google Drive. Re-checked 2026-08-16 against every mirror we could
reach; the files are gone, not firewalled:
  - `drive.usercontent.google.com/download?id=...&confirm=t` -> HTTP 404, and the
    plain `drive.google.com/file/d/<id>/view` landing page also -> HTTP 404, for
    BOTH released IDs (1mkc1PDE6yk1q-_DqwHRGlRvTBadJ5-En,
    1NAMC4uNSPwUegyCchQ0DYidBOSaCpk9W). A restricted-but-present file answers 403
    or an interstitial; 404 on the view page means deleted.
  - `docs.google.com/uc?export=download&id=...` -> HTTP 404.
  - Hugging Face model search for "neural disparity refinement" and
    "disparity refinement" -> zero hits.
  - GitHub releases API for CVLAB-Unibo/neural-disparity-refinement -> empty.
Our network is not the obstacle: the same curl reaches huggingface.co and
archive.org in the same session.
=> Unrunnable: code present, weights no longer exist anywhere reachable.

## 3. Lai et al., Blind Video Temporal Consistency (ECCV 2018) — WEIGHTS DEAD
Model-agnostic, causal at inference, the natural task-agnostic baseline.
Code cloned fine. `pretrained_models/download_models.sh` points at
http://vllab.ucmerced.edu/wlai24/video_consistency/models/ — host answers 200
but every model file returns 404. Re-checked 2026-08-16:
  - Wayback CDX has captures of the project page, its CSS and the supplementary
    PDF, but ZERO captures of `models/ECCV18_blind_consistency.pth` or
    `models/ECCV18_blind_consistency_opts.pth` (crawlers skipped the binaries).
  - Hugging Face search for "blind video consistency" /
    "fast_blind_video_consistency" -> zero hits.
  - GitHub releases API for phoenix104104/fast_blind_video_consistency -> empty.
=> Unrunnable: hosting rotted and nothing archived the weights.

## 4. TC-Stereo (ECCV 2024) — LOADS, BUT STRUCTURALLY INAPPLICABLE
Causal but INTEGRATED: it replaces and retrains the stereo network and uses
its hidden state, so it is not a plug-in rival. Useful as the integrated
upper-bound reference row.
Weights downloaded from Dropbox (603 MB): sceneflow.pth, tartanair.pth,
kitti_raw.pth.
Working load configuration, found by search (0 missing / 0 unexpected keys):
    hidden_dims=[128]*3, corr_implementation="reg", corr_levels=4,
    corr_radius=4, n_downsample=2, slow_fast_gru=False, n_gru_layers=3,
    mixed_precision=False, init_thres=0.5,
    context_norm="instance", shared_backbone=True
Checkpoint is nested under the "model" key. 16.7M parameters.
Imports cleanly in the argos env despite requirements pinning torch 2.0.1.

Its temporal propagation (core/tc_stereo.py:119-137) requires `params` carrying
per-frame camera poses `T` and `previous_T`, intrinsics `K` and `baseline`,
and warps the previous disparity by
`relative_T = cal_relative_transformation(previous_T, T)` — a rigid 6-DoF
reprojection, not optical flow.

CORRECTION (2026-08-16, later the same day). We first recorded that SCARED-C
could not supply those poses and that deformable tissue broke the rigid-scene
assumption. BOTH CLAIMS WERE WRONG for this dataset, and the error was ours to
catch since we curated the data:

  - SCARED-C EXISTS BECAUSE of the pose problem. It replaces SCARED's
    kinematics-propagated poses (inaccurate: the da Vinci is cable-driven)
    with COLMAP-re-estimated ones, metric-scaled against the structured-light
    keyframe. Verified locally: dataset/SCARED-C/raw/dataset_2/keyframe_2/
    ships data/frame_data.tar.gz with 1033 per-frame 4x4 camera-pose JSONs —
    exactly the frames in our frozen boundary — plus intrinsics_colmap.yaml
    (K, 1280x1024) and endoscope_calibration.yaml (stereo -> baseline). Our
    own DATASET_CARD.md documented this the whole time.
  - SCARED is EX-VIVO. The tissue is static and the endoscope moves, so the
    rigid-scene assumption is approximately correct here. The deformability
    objection applies to DRENDS dynamic scenarios and to D4D, not to SCARED.

So TC-Stereo's temporal path IS runnable on SCARED-C with data already on
disk. What survives, and is the better argument, is a difference in required
inputs: TC-Stereo consumes per-frame 6-DoF pose, intrinsics and baseline;
TETHER consumes RGB, disparity and validity. In a deployment where pose is
unreliable — which is the very reason SCARED-C had to be built — that is a
real structural advantage, not a convenience.
=> Runnable as a temporal method on SCARED-C, but only with privileged pose
   input. The paper states the input-requirement difference rather than an
   impossibility claim, and lists the temporal comparison as available work.

RESOLUTION (2026-08-16). We run it the only way the data permits and report it as
what it is. `scripts/run_tcstereo_reference.py` drives TC-Stereo with `params=None`
on every frame --- its single-frame stereo path --- over the SAME frozen NPZ boundary
the BiDAStabilizer comparison consumed, so the frames, the crops and the ground truth
are bit-identical across methods. The resulting row is an INTEGRATED-ARCHITECTURE
reference (16.7M parameters, trained end-to-end, evaluated zero-shot), not a temporal
rival, and the manifest records the disabled temporal path and the reason. It answers
the reviewer question our own ablations cannot: whether one should simply use a newer
integrated network instead of refining a frozen one.

RESULT (2026-08-16). Pooled over the same 4249 frames and 18.0M pixels as the BiDA
table: TC-Stereo EPE 0.5242 / Bad1 9.05% / Bad3 1.68% / RMSE 1.4059, against raw
RAFT-Stereo robust 0.3029 / 2.087% / 0.186% / 0.4484. It is a functioning stereo
network here, not a broken one (91% of pixels within 1px), just a much less accurate
one than the in-domain frozen backbone.

CAVEAT, and it is large: the stored boundary is 144x180 with median GT disparity
8.04px (range 5.57-15.53), while TC-Stereo was trained at full resolution on ranges an
order of magnitude larger. A from-scratch cost-volume matcher two octaves below its
training regime is not being tested fairly. A refiner is far less exposed, since it
corrects an existing disparity rather than building a correlation pyramid. We kept the
boundary anyway because running TC-Stereo at full resolution on frames the other
methods never saw would buy fairness to one method by destroying the property that
makes the table meaningful. The row supports "swapping in a different integrated
architecture zero-shot is not a drop-in substitute for refining a strong in-domain
frozen model"; it does NOT support "TETHER outperforms TC-Stereo", and we do not
claim that.

Note on support: the planned two supports (GT alone; GT AND raw validity) coincide
exactly on this boundary because raw_valid is true everywhere. Both select the same
17,999,475 pixels.

## 5. Deep Video Prior (NeurIPS 2020) — weights on Google Drive, expect blocked
Non-causal anyway: per-video test-time training over the whole clip.
