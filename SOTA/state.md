# ARGOS SOTA State

Validated: 2026-07-09.

This is the canonical SOTA state for this folder. It merges the operational notes, venue/dataset
verification, ARGOS-v2 positioning, and source links into one markdown file.

## Sources Merged

- `ARGOS (5).pdf`: internal ARGOS positioning, July 2026.
- `Match_Stereo_Videos_via_Bidirectional_Alignment.pdf`: BiDAStereo/BiDAStabilizer, TPAMI 2026.
- `NeurIPS-2025-ppmstereo-pick-and-play-memory-construction-for-consistent-dynamic-stereo-matching-Paper-Conference.pdf`: PPMStereo, NeurIPS 2025.
- `endostream.pdf`: EndoStreamDepth, MIDL 2026/PMLR 315.
- `agent.md`: operational notes extracted from the PDFs.
- `sota.md`: venue and dataset verification ledger.
- `a.txt`: ARGOS-v2 positioning note.

## Main ARGOS Claim

ARGOS should target a causal, online, plug-and-play temporal stereo refiner for surgical video:

```text
aligned local evidence -> forward propagated state -> bounded metric residual
```

The defensible novelty claim is narrow:

```text
A causal, safe, backbone-agnostic temporal stereo refiner trained across
heterogeneous frozen stereo predictors, using reliability-aware selective memory,
and evaluated for unseen-backbone and surgical cross-domain generalization.
```

Do not overclaim "first temporal stereo" or "first surgical depth"; those are false.

## Closest Direct Baseline

BiDAStabilizer is the closest direct baseline.

- Published as part of `Match Stereo Videos via Bidirectional Alignment`, TPAMI 2026.
- Post-hoc plugin on top of frozen per-frame stereo, not a full new stereo backbone.
- Uses aligned neighboring disparities, local feature extraction, forward/backward propagation,
  and a residual disparity output.
- Lightweight relative to BiDAStereo: about 0.7M trainable parameters, 8k iterations,
  about 0.5 day on A100 in the paper.
- Not causal: it uses future frames and a backward temporal pass.

ARGOS should adapt the principle, not copy the offline formulation.

## Architecture Decision

Start with the smallest causal BiDAStabilizer-style model, then add selective memory only if the
aligned-state baseline plateaus.

Universal evidence encoder inputs should be matcher-agnostic:

- left/right RGB;
- current raw disparity;
- stereo photometric residual;
- raw disparity gradient;
- aligned previous refined disparity;
- alignment residual;
- previous reliability;
- valid/occlusion/finite/warp-support masks.

The refiner should maintain causal state:

- previous hidden state;
- target-to-source optical flow;
- aligned previous evidence;
- persistent forward state.

The output should be identity-preserving by construction:

```text
D_ref = D_raw + g * c * tau * tanh(delta)
```

Required behavior:

- reset state at sequence boundaries;
- expose state reset for ablation;
- preserve raw-good pixels;
- limit correction magnitude;
- report harmful and beneficial corrections separately.
- initialize near identity: gate near 0, residual near 0, refined approximately raw.

## Design Signals From The PDFs

- Alignment matters. Removing BiDAStabilizer alignment hurts temporal consistency.
- Propagation matters. Removing propagation also hurts temporal consistency.
- Generic 3D conv/attention alternatives underperform BiDA-style propagation in the reported
  ablations.
- Local evidence should precede global/state propagation. Global aggregation alone can degrade.
- Temporal smoothness is not enough; spatial geometry and harmful corrections must be measured.

## Causal Selective Memory

PPMStereo is useful evidence for reliability-aware long-range memory, not the first ARGOS build.

- Published in NeurIPS 2025.
- Uses Pick-and-Play Memory: pick relevant frames by confidence/redundancy/similarity, then
  play/weight selected memory before read-out.
- Reported SOTA gains are strong, but the method is heavier: 8x A100 training, 180k iterations,
  320x512 crops, 20-frame inference.
- It is cost-volume/backbone-specific and not a simple plugin over arbitrary disparities.
- Its presented memory is sequence-level, not a pure causal streaming formulation.
- The useful ARGOS version is causal and past-only:

```text
memory = {t-1, t-2, ..., t-M}
score = quality + similarity - redundancy + warp_support - age_penalty
aggregate top-K, likely K=3 or K=5
```

Use PPMStereo-like memory only after the small causal aligned-state model plateaus.

## EndoStreamDepth Position

EndoStreamDepth is surgical and streaming, but monocular depth rather than stereo disparity.

- Published in PMLR 315, MIDL 2026.
- Processes frames sequentially with temporal Mamba state.
- Uses endoscopy-specific augmentations: rotations/flips, blur/defocus, brightness/contrast,
  gamma, fog/smoke-like effects.
- Uses multi-level temporal modules, multi-scale supervision, and self-supervised temporal
  regularization.
- Reported C3VD split-1 numbers in the local PDF: AbsRel 0.085, RMSE 2.739 mm, L1 1.780 mm,
  average 24 FPS.
- Its temporal regularizer is simple normalized frame-to-frame depth smoothing, without motion
  compensation; this can over-reward static predictions under real motion/deformation.
- Its useful ARGOS lesson is hierarchical streaming state, not "use Mamba" as a novelty claim.

Use it for domain adaptation and streaming-loss ideas, not as a direct stereo-refiner baseline.

## What Is Not Novel

Do not claim these as standalone contributions:

- selective temporal memory for video stereo: PPMStereo already does this;
- hierarchical Mamba for surgical streaming depth: EndoStreamDepth already does this;
- plugin stereo stabilization over a frozen matcher: BiDAStabilizer already does this.

ARGOS novelty must be the intersection:

- stereo plugin;
- causal streaming;
- universal signals rather than matcher-private features/cost volume;
- multi-backbone training;
- unseen-backbone testing;
- surgical OOD evaluation;
- explicit identity/raw-good safety.

## Venue State

High-confidence verified methods:

| Method | State | ARGOS relevance |
|---|---|---|
| BiDAStabilizer / BiDAStereo | TPAMI 2026; journal extension of ECCV 2024 BiDAStereo | closest stereo plugin baseline |
| PPMStereo | NeurIPS 2025 | long-range reliability-aware memory |
| EndoStreamDepth | MIDL 2026 / PMLR 315 | surgical streaming depth context |
| FlashDepth | ICCV 2025 Highlight | streaming Mamba depth baseline, natural domain |
| DepthSync | ICCV 2025 | video depth consistency, diffusion guidance |
| Video Depth Anything | CVPR 2025 Highlight | strong monocular video prior |
| ChronoDepth | CVPR 2025 | temporally consistent monocular depth |
| TC-Stereo | ECCV 2024 | causal/end-to-end temporal stereo, pose-dependent |
| Stereo Any Video | ICCV 2025 | temporally consistent stereo without pose/flow |
| StereoDiffusion | MICCAI 2024 | surgical stereo/depth context |
| EndoDAV | MICCAI 2025 | surgical depth context |
| StableDPT | preprint only | do not cite as accepted |

## Dataset Roles

| Dataset | Use in ARGOS | Notes |
|---|---|---|
| SCARED | source-domain development, keyframe geometry, internal processed temporal supervision | official GT is structured-light keyframes only |
| SCARED-C | processed temporal supervision only after per-sequence quality gate | not official native dense temporal GT |
| D4D | main OOD temporal and sparse/anchor geometry check; never train/tune | >300k frames, 369 point clouds, 98 clips; structured-light anchors |
| SERV-CT | static OOD geometry/safety | 16 CT-derived stereo pairs, no temporal signal |
| Hamlyn | optional pretraining/diversity | pseudo-GT from stereo/SfM; not clean metric GT |
| EndoSLAM | optional pretraining/diversity | local copy is not benchmark-ready unless full data is acquired |
| StereoMIS | qualitative/prediction-space only | stereo video + kinematics, no dense depth GT |
| C3VD | monocular colonoscopy context | not stereo; do not use as stereo benchmark |

Backbone protocol:

- train across heterogeneous frozen stereo predictors;
- keep at least one strong predictor fully unseen for testing;
- report whether the refiner helps or harms already-good OOD predictions.

Rules:

- Label propagated/densified SCARED targets as internal processed supervision.
- Keep D4D OOD: no training, no tuning, no threshold selection on it.
- Use SERV-CT only for static geometry and false-activation checks.
- Do not claim StereoMIS has dense disparity GT.

## Minimum Metrics

Spatial:

- EPE/MAE;
- Bad-1, Bad-3, Bad-5;
- boundary error;
- raw-good preservation;
- new-Bad and false activation on raw-good pixels.

Temporal:

- TGM or TEPE where processed GT exists;
- motion-compensated disparity inconsistency;
- temporal jitter/high-frequency error;
- correction flicker;
- state-reset degradation.

Safety:

- harmful vs beneficial correction rates;
- catastrophic correction count;
- correction magnitude percentiles;
- modified-pixel ratio;
- OOD sign stability;
- per-seed variance.

## Decision Gates

Do not advance a larger model unless these hold:

1. aligned local beats unaligned concatenation;
2. forward propagation beats aligned-local-only;
3. full causal model beats current-only;
4. shuffled history breaks the gain;
5. persistent state beats state reset;
6. corrections are nonzero but bounded;
7. raw-good preservation remains acceptable;
8. SCARED geometry does not materially degrade;
9. D4D temporal improvement has stable sign;
10. D4D sparse/anchor geometry remains safe;
11. SERV-CT shows no major false activation;
12. unseen-backbone performance remains safe;
13. runtime remains online-feasible, target under 100 ms/frame at eval resolution.

## Practical Next Build

Build the small causal aligned-state refiner first. Add causal Pick-and-Play memory only after the
minimal model has clean ablations showing memory is the bottleneck. Skip diffusion and large
attention.

## Validation Notes

- `state.md` was empty before this merge.
- The merged venue/dataset claims were cross-checked against local PDFs and the existing
  `sota.md` evidence ledger.
- `a.txt` was incorporated as the primary ARGOS-v2 positioning note.
- Spot online validation on 2026-07-09 confirmed the central external claims for BiDAVideo,
  PPMStereo, EndoStreamDepth, and D4D.

## Discrepancies And Flags

| Item | Verified state | Flag |
|---|---|---|
| BiDAStabilizer | TPAMI 2026, pub. 31 Mar 2026, DOI `10.1109/tpami.2026.3679033` | no issue |
| GemDepth | ICML 2026 poster | arXiv did not state venue directly |
| StableDPT | preprint only | do not cite as accepted |
| Video Depth Anything | CVPR 2025 Highlight | cite Highlight if status matters |
| SCARED keyframes | original challenge has 7 train + 2 test datasets with 5-10 keyframes each | redistributed subsets differ |
| StereoMIS | no dense depth GT | SLAM/qualitative only for ARGOS |
| D4D | arXiv preprint, data available | journal paper not published as of July 2026 |
| EndoDAV | MICCAI 2025 only | no arXiv found |
| StereoDiffusion | MICCAI 2024 only | no arXiv found |
| C3VD | monocular colonoscopy | not a stereo benchmark |

## Remaining Uncertainties

- SCARED exact public frame/keyframe counts vary by distribution.
- Hamlyn pseudo-GT quality varies by sequence and is not external-sensor metric GT.
- StableDPT may later receive a venue; no acceptance evidence was found in the current ledger.
- EndoSLAM local copy is only the lightweight source/sample structure, not a ready ARGOS benchmark.

## Source Links

Methods:

- BiDAStabilizer / BiDAStereo: https://doi.org/10.1109/tpami.2026.3679033
- GemDepth ICML poster: https://icml.cc/virtual/2026/poster/65792
- PPMStereo NeurIPS: https://proceedings.neurips.cc/paper_files/paper/2025/hash/8b2fc235787852ead92da2268cd9e90c-Abstract-Conference.html
- PPMStereo OpenReview: https://openreview.net/forum?id=MaglIUQKVX
- EndoStreamDepth PMLR: https://proceedings.mlr.press/v315/li26f.html
- EndoStreamDepth OpenReview: https://openreview.net/forum?id=I7lgdDdcij
- FlashDepth project: https://eyeline-labs.github.io/FlashDepth/
- FlashDepth CVF: https://openaccess.thecvf.com/content/ICCV2025/html/Chou_FlashDepth_Real-time_Streaming_Video_Depth_Estimation_at_2K_Resolution_ICCV_2025_paper.html
- DepthSync arXiv: https://arxiv.org/abs/2507.01603
- DepthSync CVF: https://openaccess.thecvf.com/content/ICCV2025/html/Dong_DepthSync_Diffusion_Guidance-Based_Depth_Synchronization_for_Scale-_and_Geometry-Consistent_Video_ICCV_2025_paper.html
- StableDPT arXiv: https://arxiv.org/abs/2601.02793
- Video Depth Anything CVF: https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Video_Depth_Anything_Consistent_Depth_Estimation_for_Super-Long_Videos_CVPR_2025_paper.html
- Video Depth Anything GitHub: https://github.com/DepthAnything/Video-Depth-Anything
- ChronoDepth CVF: https://openaccess.thecvf.com/content/CVPR2025/html/Shao_Learning_Temporally_Consistent_Video_Depth_from_Video_Diffusion_Priors_CVPR_2025_paper.html
- ChronoDepth project: https://xdimlab.github.io/ChronoDepth/
- TC-Stereo ECVA: https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/4579_ECCV_2024_paper.php
- BiDAStereo ECVA PDF: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07780.pdf
- Stereo Any Video CVF: https://openaccess.thecvf.com/content/ICCV2025/html/Jing_Stereo_Any_Video_Temporally_Consistent_Stereo_Matching_ICCV_2025_paper.html
- StereoDiffusion MICCAI: https://papers.miccai.org/miccai-2024/733-Paper0240.html
- EndoDAV MICCAI: https://papers.miccai.org/miccai-2025/0288-Paper1355.html

Datasets:

- SCARED challenge: https://endovissub2019-scared.grand-challenge.org/
- SCARED paper: https://arxiv.org/abs/2101.01133
- SERV-CT paper: https://doi.org/10.1016/j.media.2021.102302
- SERV-CT data: https://doi.org/10.5522/04/26352199
- SERV-CT toolkit: https://github.com/surgical-vision/servcttk
- D4D arXiv: https://arxiv.org/abs/2603.02985
- D4D data DOI: https://doi.org/10.25532/opara-1033
- D4D GitHub: https://github.com/reubendocea/d4d
- D4D project: https://reubendocea.github.io/d4d/
- Hamlyn Centre: http://hamlyn.doc.ic.ac.uk/vision
- Endo-Depth-and-Motion: https://davidrecasens.github.io/EndoDepthAndMotion/
- Hamlyn Rectified mirror: https://huggingface.co/datasets/vslamlab/Hamlyn_Rectified_Dataset
- EndoSLAM GitHub: https://github.com/CapsuleEndoscope/EndoSLAM
- EndoSLAM paper: https://doi.org/10.1016/j.media.2021.102058
- StereoMIS Zenodo v2: https://zenodo.org/records/8154924
- StereoMIS concept DOI: https://doi.org/10.5281/zenodo.7727691
- C3VD project: https://durrlab.github.io/C3VD/
- C3VD paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC10591895/
- C3VDv2 project: https://durrlab.github.io/C3VDv2/
