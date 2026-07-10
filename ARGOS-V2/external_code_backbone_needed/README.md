# ARGOS v2 External Code Backbone Notes

This directory contains the minimal external-code audit/export for ARGOS v2. It is not a dump of the cloned repositories.

## Why These Repositories Were Cloned

| Repository | Why it matters |
|---|---|
| BiDAVideo / BiDAStabilizer | Closest plug-in stereo stabilizer baseline; source for alignment, propagation, fusion, residual-output conventions |
| PPMStereo | Best reference for reliability-aware Pick-and-Play memory selection |
| EndoStreamDepth | Surgical streaming reference for state lifecycle, multi-scale temporal processing, and endoscopic augmentations/losses |
| SEA-RAFT | Preferred optical-flow implementation for ARGOS v2 |
| RAFT | Reference flow implementation used by BiDAStabilizer-style pipelines |

## What Was Exported

- Clean-room causal warp utility: `bidavideo/alignment/causal_warp.py`
- Clean-room top-K memory selection utility: `ppmstereo/memory_selection/pick_and_play.py`
- Clean-room streaming state helper: `endostreamdepth/state_management/streaming_state.py`
- License copies under `LICENSES/`
- Isolated smoke tests under `tests/`
- Component notes in each repository subfolder
- Machine-readable `MANIFEST.json`

## What Was Intentionally Excluded

- Full stereo/depth/flow backbones
- Training loops
- Dataset loaders
- Checkpoints
- Generated outputs, caches, binaries, build products, virtual environments
- Non-causal backward/future-frame deployable code

## Causal vs Non-Causal

- Directly usable in a causal path: `causal_warp.py`, `pick_and_play.py`, `streaming_state.py`.
- Reference-only: BiDAStabilizer full forward path, because it uses future frames and backward propagation.
- Reference-only: PPMStereo full model, because memory and read-out are cost-volume/backbone-specific and sequence-level.
- Reference-only: EndoStreamDepth full model, because it is monocular DepthAnything/DPT-coupled.

## Backbone-Specific vs Universal

- Universal: target-to-source warping, top-K memory scoring, streaming state reset/update.
- Backbone-specific: PPMStereo cost volumes, DynamicStereo/RAFT-style GRU, BiDAVideo frozen stereo wrappers, EndoStreamDepth DPT/Mamba integration.

## Dependency Graph

```text
ARGOS v2 first prototype
  -> clean-room warp
  -> clean-room top-K memory
  -> clean-room streaming state
  -> SEA-RAFT wrapper later, checkpoint configured outside this export
```

## Recommended Integration Order

1. Use `causal_warp.py` for target-to-source evidence alignment.
2. Use `streaming_state.py` only as the lifecycle smoke-test pattern; implement the real state in the ARGOS v2 model.
3. Use `pick_and_play.py` for initial memory-selection ablations.
4. Add a SEA-RAFT wrapper only after a checkpoint path is configured.
5. Keep RAFT as a reference comparator, not the default.

## Direct Import vs Reimplementation

| Component | Recommendation |
|---|---|
| Causal warp | Direct reuse or inline into ARGOS v2 geometry utilities |
| Pick-and-Play selection | Direct reuse for prototype; adapt scoring terms later |
| Streaming state helper | Reference/smoke-test only |
| SEA-RAFT inference | Wrapper around external repo, not copied here |
| RAFT inference | Reference wrapper only if SEA-RAFT fails |
| BiDAStabilizer propagation/fusion | Clean reimplementation, causal-only |
| EndoStreamDepth Mamba stack | Conceptual reference; avoid importing DPT-coupled code |

## License Constraints

License texts are copied under `LICENSES/`. Preserve attribution if any exported code is expanded with source-derived implementations.
