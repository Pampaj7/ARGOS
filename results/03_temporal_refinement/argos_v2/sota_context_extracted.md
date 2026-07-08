# ARGOS v2 SOTA Context Extracted

Generated: 2026-07-07

## Files Inspected

- `SOTA/ARGOS (5).pdf` (28-page living technical document, July 2026).

Important path note: the prompt requested `sota/`, but this repository currently contains `SOTA/` (uppercase) and no lowercase `sota/` directory. The uppercase PDF is the only SOTA source found and is treated as the current ARGOS v2 source of truth.

## Extracted Scientific Plan

ARGOS v2 is framed as a causal, online, plug-and-play, metric, safety-aware temporal stereo refiner for deformable surgical scenes. The core direction extracted from the SOTA document is:

```text
aligned local evidence -> forward propagated state -> safety-bounded metric residual
```

The document positions NVDS/NVDS+ as conceptual ancestors for plug-and-play RGB-depth video stabilisation, but not as the final methodological target. The closest direct methodological baseline is BiDAStabilizer from the BiDAStereo line: frozen stereo backbone, optical-flow alignment, local disparity feature extraction, hidden-state propagation, and residual correction.

ARGOS v2 should adapt this principle causally and safely: target-to-source flow alignment, reliability masks, bounded gated disparity residuals, raw-good preservation, and strict OOD validation.

## BiDA / NVDS Lessons

NVDS/NVDS+ lessons:

- useful paradigm: post-hoc plug-in stabilisation over frozen predictions;
- useful mechanism: RGB-depth features and local cross-frame reasoning;
- limitations for ARGOS: monocular, relative depth, short-window, offline/non-metric context, not surgical, no explicit metric safety.

BiDAStabilizer lessons:

- closest published stereo analogue;
- performs explicit alignment of neighbouring disparities to the current frame;
- extracts local aligned disparity features before global propagation;
- propagates features forward and backward;
- outputs residual corrections;
- is not causal because it uses future disparity and backward propagation;
- residuals are not ARGOS-style bounded safety corrections.

Ablation guidance:

- alignment matters;
- propagation matters;
- generic 3D convolution/attention is worse than the BiDA propagation structure;
- local matching before global propagation matters;
- propagation without valid local aligned evidence can hurt;
- temporal consistency alone is not sufficient without geometry/safety/OOD checks.

## Flow and Mask Conventions

The SOTA document explicitly states the backward-sampling convention:

```text
W(source, f)(p) = source(p + f(p))
```

Therefore, to warp frame `t-1` into current target frame `t`, the required sampling field is:

```text
flow(t -> t-1)
```

The warp-support mask must include:

- current-frame validity;
- previous-frame validity warped into current-frame coordinates;
- in-bounds support;
- target-frame occlusion / forward-backward consistency;
- finite values.

The current validity mask must not be intersected with the unwarped previous validity mask because those masks are in different coordinate systems.

## Baseline Ladder

The updated ARGOS v2 ladder is:

1. raw frozen S2M2;
2. current-only bounded refiner;
3. naive causal multi-frame concatenation;
4. NVDS-lite local correlation;
5. aligned-local-only convolutional refiner;
6. causal BiDA-style propagation without safety gate;
7. full safe causal BiDA-style model;
8. shuffled-history full model;
9. offline bidirectional upper bound, if needed.

The recommended progression is:

```text
current only -> unaligned temporal -> aligned local -> aligned local + persistent state -> safe persistent state
```

## Decision Gates

A temporal architecture should advance only if it satisfies the relevant gates:

- aligned-local-only beats naive unaligned concatenation;
- forward propagation beats aligned-local-only;
- full causal model beats current-only;
- full causal model beats shuffled-history;
- correction remains nonzero but bounded;
- raw-good preservation is acceptable;
- SCARED geometry does not materially degrade;
- D4D temporal improvement has stable sign;
- D4D sparse anchors remain geometrically safe;
- SERV-CT shows no major false activation;
- runtime remains within the online budget.

If a model improves SCARED but materially degrades D4D or SERV-CT, it should not advance as the ARGOS v2 architecture.

## Dataset Roles

- SCARED: source-domain development/training/evaluation with processed internal temporal supervision.
- D4D: no training, no tuning; temporal OOD validation with sparse Zivid anchor geometry and temporal prediction-space metrics.
- SERV-CT: no training, no tuning; static OOD geometry and safety, especially false activation when raw predictions are already good.

Cross-backbone evaluation is planned after the temporal architecture is validated. S2M2-S remains the development frozen stereo backbone for the first ARGOS v2 stage.

## Current ARGOS v2 Interpretation

The SOTA document changes the role of NVDS-lite: it is no longer the likely final architecture. It should be closed as a local-correlation/non-propagating baseline before moving to aligned-local-only and then causal forward BiDA-style propagation.

The minimum useful NVDS-lite question is not “can we run the old 18-run matrix?” It is:

```text
Does explicit corrected causal history provide benefit over current-only, shuffled-history, and naive concat while preserving geometry and safety?
```

## Contradictions With Older Material

- Older plans requested an 18-run NVDS-lite matrix. The SOTA document recommends a staged one-seed ladder before promotion and treats NVDS-lite as a baseline, not the main architecture.
- Older NVDS-lite runs used conservative gate/loss settings that collapsed to identity. Current ARGOS v2 guidance requires nonzero bounded corrections.
- Older evaluation snippets reset temporal windows arbitrarily. SOTA decision gates require proper causal behaviour and state/reset tests.
- Any material that treats source-to-target flow as the warp field for `output(p)=source(p+flow(p))` conflicts with the updated SOTA flow convention.

## Unresolved Questions

- Whether corrected NVDS-lite full history beats current-only and shuffled-history under identical corrected settings.
- Whether the current evaluator’s non-overlapping window resets materially bias temporal metrics.
- Whether warp loss adds interpretable benefit once flow and masks are corrected.
- Whether RGB helps beyond disparity-only local matching.
- Whether NVDS-lite should be closed as NOT CONFIRMED or promoted to a small 3-seed check before aligned-local-only.

## Official BiDAVideo Repository Integration Update

Official repository cloned for ARGOS v2 BiDAStabilizer extraction:

- Path: `external/bidavideo/`
- Remote: `https://github.com/TomTomTommi/bidavideo.git`
- Commit: `dae817df1ceaafcb865ebd9c7aa44b16c535e856`
- License: MIT

The official code confirms the SOTA plan: BiDAStabilizer is a compact post-hoc stereo disparity stabilizer with explicit optical-flow alignment, local aligned disparity feature extraction, forward/backward feature propagation, and additive residual output. ARGOS v2 adaptation keeps the local/propagation mechanism but removes future-frame and backward-propagation paths.

