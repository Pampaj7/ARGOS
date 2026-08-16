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

BLOCKER, and it is a scientific one rather than an engineering one. Its
temporal propagation (core/tc_stereo.py:119-137) requires `params` carrying
per-frame camera poses `T` and `previous_T`, intrinsics `K` and `baseline`,
and warps the previous disparity by
`relative_T = cal_relative_transformation(previous_T, T)` — a rigid 6-DoF
reprojection, not optical flow. That assumes (a) known ego-motion per frame,
which SCARED-C does not provide, and (b) a RIGID scene, which deformable
surgical tissue violates by construction. Running it with `params=None` on
every frame is possible but degenerates it to a single-frame stereo network,
which is not the method.
=> Not a fair or even feasible temporal comparison on surgical data. This is
   a real structural difference worth stating in the paper: rigid-motion
   reprojection versus optical-flow alignment.

RESOLUTION (2026-08-16). We run it the only way the data permits and report it as
what it is. `scripts/run_tcstereo_reference.py` drives TC-Stereo with `params=None`
on every frame --- its single-frame stereo path --- over the SAME frozen NPZ boundary
the BiDAStabilizer comparison consumed, so the frames, the crops and the ground truth
are bit-identical across methods. The resulting row is an INTEGRATED-ARCHITECTURE
reference (16.7M parameters, trained end-to-end, evaluated zero-shot), not a temporal
rival, and the manifest records the disabled temporal path and the reason. It answers
the reviewer question our own ablations cannot: whether one should simply use a newer
integrated network instead of refining a frozen one.

## 5. Deep Video Prior (NeurIPS 2020) — weights on Google Drive, expect blocked
Non-causal anyway: per-video test-time training over the whole clip.
