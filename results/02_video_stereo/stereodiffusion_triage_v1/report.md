# StereoDiffusion Triage Report

## Overview
This report investigates **StereoDiffusion** as a potential surgical temporal stereo baseline for the ARGOS framework.

**Target Method:** StereoDiffusion: Temporally Consistent Stereo Depth Estimation with Diffusion Models (MICCAI 2024).
**Authors:** Haozheng Xu, Chi Xu, Stamatia Giannarou.
**Official GitHub:** [https://github.com/xuhaozheng/StereoDiff](https://github.com/xuhaozheng/StereoDiff)

*(Note: There is an unrelated CVPR 2024 paper also named "StereoDiffusion" which focuses on training-free 3D image generation. This report exclusively discusses the MICCAI 2024 paper which explicitly targets surgical depth estimation).*

## Run Status: ❌ Not Runnable

### Exact Blocker
The official repository was successfully located and cloned. However, **the repository is completely empty**. It contains only a `.gitignore`, a `LICENSE` file, and a 12-byte `README.md` containing solely the title `# StereoDiff`. No source code, inference scripts, dataset loaders, or pre-trained weights have been released by the authors. 

Because the code is closed/unavailable, it is impossible to evaluate the method on the SCARED dataset using the ARGOS temporal GT protocol.

### Metric Evaluation
- **Depth/Disparity MAE:** N/A (Cannot run)
- **Temporal Metrics:** N/A (Cannot run)
- **Runtime/VRAM:** N/A (Cannot measure, though Latent Diffusion Models typically have high VRAM footprints and slow inference times compared to ConvGRU networks).

## Capabilities and Constraints
Based on the peer-reviewed paper:
*   **Is it causal or offline?** **Causal**. The paper describes utilizing optical flow to warp the *previous* frame's disparity map, using it as prior knowledge to condition the latent diffusion process for the current frame.
*   **Official checkpoints:** **No**. None are published.
*   **Process custom sequences:** **Unknown/No**. Without the codebase, we cannot run it on custom sequences.

## Novelty Threat Assessment
Does this paper threaten our novelty claim with the `CausalWarpedFusionRefiner` (Adaptive Motion Fusion)?

**Moderate Threat (Thematic Overlap):**
*   **Similarity:** Both methods address surgical temporal stereo matching. Crucially, both methods independently arrived at the idea of using **Optical Flow to warp the previous frame's disparity** as a temporal prior for the current frame.
*   **Divergence:** Their approach feeds this warped prior into a heavy **Latent Diffusion Model** to denoise and generate the final depth map. Our approach feeds the warped prior into a lightweight **ConvGRU-based Adaptive Fusion Refiner** that dynamically gates the temporal memory versus the raw spatial prediction using an `alpha` blending mask.
*   **Our Advantage:** Our method is explicitly designed for real-time robotic surgery. We achieve SOTA geometry (2.55mm Depth MAE) with **under 1 GB of VRAM** at real-time speeds. Diffusion models (even with "efficient denoising schedulers") are notoriously slow and compute-heavy.

**Conclusion:** 
StereoDiffusion (MICCAI 2024) must be treated as **Related Work**. It should be explicitly cited in our paper when discussing "optical-flow warped temporal priors". However, we can clearly distinguish our work by highlighting our method's realtime efficiency, explicit adaptive gating (`alpha` and `reset` masks), and low memory footprint compared to diffusion-based paradigms.
