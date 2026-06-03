# ARGOS Roadmap

## Phase 0: Baseline Grounding

Status: active.

- Build SERV-CT leaderboard.
- Test representative classical, recurrent, foundation, monodepth-prior, and scalable stereo models.
- Keep all results in compact JSON/CSV/PNG form.

## Phase 1: Dataset Unification

Status: waiting for SCARED.

- Convert SERV-CT to ARGOS unified sample format.
- Convert SCARED to ARGOS unified sample format.
- Add smoke fixtures for converter validation.
- Define cross-dataset splits.

## Phase 2: Surgical Fine-Tuning

Status: partially started with S2M2.

- Fine-tune S2M2-S/M/L/XL.
- Evaluate honest split and all-surgical upper bound.
- Add SCARED training once converted.
- Compare zero-shot, SERV-CT-tuned, SCARED-tuned, and mixed surgical tuning.
- Revisit DEFOM/MonSter++ adaptation only after S2M2 larger-scale runs establish the surgical baseline.

## Phase 3: Surgical Robustness Analysis

Status: planned.

- Boundary/detail error.
- Near-field depth bins.
- Specular/textureless masks.
- Qualitative crop figures.
- Failure case catalog.

## Phase 4: Paper Package

Status: planned.

- Reproducible benchmark scripts.
- Stable tables and figures.
- Dataset license/provenance section.
- Model checkpoint provenance.
- Ablation around surgical fine-tuning scale.
