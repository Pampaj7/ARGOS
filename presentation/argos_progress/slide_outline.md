# Proposed ARGOS Presentation Slide Outline

## Slide 1: ARGOS Motivation
**Core message:** Surgical stereo depth needs robust, temporally stable, near-field disparity estimates.
**Recommended asset:** `diagrams/dataset_overview.png`
**Bullets:**
- Open-surgery-like scenes have specular tissue, instruments, occlusions, and motion.
- Frame-wise SOTA stereo can be accurate but temporally unstable.
- ARGOS is building a surgical stereo evaluation and refinement stack.
**Speaker notes:** Start with the clinical/research motivation, then transition to depth consistency as the core technical problem.
**Important caveat:** This is research infrastructure and benchmarking, not clinical validation.

## Slide 2: Why Surgical Stereo Is Hard
**Core message:** The visual domain violates many assumptions of outdoor/general stereo benchmarks.
**Recommended asset:** `qualitative/comparison_000945.png`
**Bullets:**
- Reflective tissue and smooth surfaces reduce reliable texture.
- Instruments and occlusions create local discontinuities.
- Camera motion makes frame-wise disparity flicker visible.
**Speaker notes:** Use the RGB panel and residual map to point at specular/high-gradient regions.
**Important caveat:** Qualitative examples illustrate failure modes, not aggregate performance.

## Slide 3: Data And Cache Pipeline
**Core message:** We now have a reusable SCARED temporal cache with frozen predictions from multiple streams.
**Recommended asset:** `tables/datasets_slide_ready.csv`, `diagrams/dataset_overview.png`
**Bullets:**
- 8 SCARED long streams, 1040 stereo frames, 1008 valid 5-frame windows.
- Cached streams: S2M2-S@512, S2M2-L@736, StereoAnyVideo@384x640.
- Unified validation uses 126 full-frame rows from `test_dataset_9_keyframe_3` (now with GT attached!).
**Speaker notes:** Explain why caching predictions lets us iterate on refiners quickly.
**Important caveat:** GT metrics exclude invalid pixels based on the confidence threshold.

## Slide 4: SOTA Exploration Audit
**Core message:** We explored both frame-stereo and video-stereo repos before selecting the current pipeline.
**Recommended asset:** `tables/model_audit_slide_ready.csv`
**Bullets:**
- Frame models: S2M2, Foundation/Fast-FoundationStereo, RAFT-Stereo, CREStereo, DEFOM, MonSter++.
- Video models: StereoAnyVideo, TC-Stereo, TemporalStereo, PPMStereo, DynamicStereo/BiDAStereo.
- StereoAnyVideo was the most immediately usable temporal teacher.
**Speaker notes:** Stress that statuses differ: downloaded, attempted, successful, quantitatively evaluated.
**Important caveat:** Do not rank models without matching datasets/protocols.

## Slide 5: Frame-Based S2M2 Benchmark
**Core message:** S2M2-L@736 became the practical backbone for temporal refinement.
**Recommended asset:** `tables/frame_stereo_benchmark_slide_ready.csv`, `plots/frame_accuracy_vs_runtime.png`
**Bullets:**
- S2M2-S is fast and deployment-oriented.
- S2M2-L gives a better accuracy/speed compromise.
- XL/full-size variants are useful but not automatically worth the cost.
**Speaker notes:** Separate SCARED GT keyframe results from full-frame temporal results.
**Important caveat:** These numbers come from 5-frame SCARED GT keyframe experiments.

## Slide 6: Temporal Instability Of Frame-Wise Stereo
**Core message:** Strong frame models still flicker when run independently over video.
**Recommended asset:** `videos/01_rgb_raw_s2m2.mp4`, `plots/temporal_vs_backbone.png`
**Bullets:**
- Raw S2M2-L full-frame temporal diff: 0.984 on the unified validation sequence.
- SAV teacher temporal diff: 0.925.
- This gap motivates temporal refinement.
**Speaker notes:** Play video and ask audience to watch the disparity surface rather than RGB motion.
**Important caveat:** Geometric GT evaluation uses rectified data subsets where valid depth is present.

## Slide 7: StereoAnyVideo As Teacher
**Core message:** StereoAnyVideo is a strong temporal reference but not the deployment target.
**Recommended asset:** `videos/02_main_temporal_raw_v2e30_sav.mp4`
**Bullets:**
- Video-native method improves temporal consistency.
- It is heavier and non-causal in our use.
- We use it as a teacher/upper-bound reference.
**Speaker notes:** Highlight SAV as supervision signal, not a final deployable surgical model yet.
**Important caveat:** SAV predictions are not guaranteed geometrically correct without GT.

## Slide 8: Teacher-Student Residual Refinement
**Core message:** Freeze expensive stereo models, train a small residual refiner.
**Recommended asset:** `diagrams/teacher_student_temporal_refinement.png`
**Bullets:**
- Input: frozen S2M2 disparity plus RGB context.
- Teacher: StereoAnyVideo temporal dynamics.
- Output: bounded residual added to backbone disparity.
**Speaker notes:** This is the core ARGOS-owned contribution in the current repo.
**Important caveat:** Training objective trades off backbone preservation and temporal smoothing.

## Slide 9: Tiny U-Net Prototype
**Core message:** Tiny U-Net proved the cache/model/loss pipeline but is bidirectional/non-causal.
**Recommended asset:** `diagrams/tiny_unet_refinement.png`, `videos/03_learned_refiners.mp4`
**Bullets:**
- Uses RGB center + 5-frame disparity window.
- Best full-frame temporal diff: 1.2434, modest improvement over raw.
- Good debugging vehicle, not ideal for online deployment.
**Speaker notes:** Position Tiny U-Net as a stepping stone.
**Important caveat:** It can use future frames, so do not compare as a causal deployment model.

## Slide 10: Causal ConvGRU Architecture
**Core message:** ConvGRU gives online temporal memory without future-frame leakage.
**Recommended asset:** `diagrams/convgru_refinement.png`
**Bullets:**
- Input per timestep: RGB_t + disparity_t.
- Hidden state resets at sequence boundaries.
- Produces residual disparity for current frame.
**Speaker notes:** Emphasize recurrent state and causal inference.
**Important caveat:** Causality is architectural; quality still depends on objective and data.

## Slide 11: Conservative Vs Scheduled Training
**Core message:** Scheduled temporal loss produces better temporal Pareto points than conservative training.
**Recommended asset:** `diagrams/scheduled_loss_training.png`, `plots/scheduled_loss_weight_evolution.png`
**Bullets:**
- Epochs 1-10 preserve backbone strongly.
- Epochs 11-30 transition toward stronger teacher-delta.
- Epochs 31-100 hold stronger temporal supervision.
**Speaker notes:** Explain why `best.pt` can be misleading when score favors conservative checkpoints.
**Important caveat:** Best temporal checkpoint is not the same as best training-score checkpoint.

## Slide 12: Unified Full-Frame Evaluation
**Core message:** ConvGRU V2 epoch 30 improves temporal stability with moderate backbone deviation.
**Recommended asset:** `tables/unified_fullframe_evaluation_slide_ready.csv`, `plots/checkpoint_evolution_convgru_v2.png`
**Bullets:**
- Raw S2M2-L temporal diff: 0.984.
- ConvGRU V2 epoch 30 temporal diff: 1.1545.
- ConvGRU V2 epoch 30 teacher-delta: 0.6290.
**Speaker notes:** Use epoch 30–50 as Pareto candidates, not final epoch alone.
**Important caveat:** The refined methods now also undergo GT evaluation.

## Slide 13: Learned Causal Model Vs Classical Smoothing
**Core message:** Classical smoothing can win temporal metrics, but causality and geometry matter.
**Recommended asset:** `videos/05_classical_baselines.mp4`, `plots/causal_vs_noncausal_baselines.png`
**Bullets:**
- Median5 non-causal temporal diff: 0.9447.
- It uses future frames and is not online.
- ConvGRU is causal and learned, but still has room to improve.
**Speaker notes:** Be honest: simple baselines are strong and valuable controls.
**Important caveat:** Median5 non-causal should not be presented as a deployable causal baseline.

## Slide 14: Limitations
**Core message:** The current strongest temporal result relies on GT masks which discard many pixels.
**Recommended asset:** `presentation_summary.md`
**Bullets:**
- Unified full-frame validation is limited by sparse GT depth.
- Temporal smoothness on unmasked regions does not prove overall depth correctness.
- Some long SCARED streams show teacher/backbone scale disagreement.
**Speaker notes:** State limitations plainly before next steps.
**Important caveat:** Avoid overclaiming accuracy on regions without GT data.

## Slide 15: Next Steps
**Core message:** Move from temporal behavior to geometry-validated surgical depth.
**Recommended asset:** `plots/temporal_improvement_vs_raw.png`
**Bullets:**
- Add GT/calibrated temporal evaluation subsets.
- Train/evaluate confidence masks and edge-aware objectives.
- Test real open-surgery-like RGB-D/stereo acquisitions or phantom data.
**Speaker notes:** End on concrete engineering/research next actions.
**Important caveat:** Final deployment requires calibrated, externally validated depth accuracy.
