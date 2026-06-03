# ARGOS Experiment Protocol

This document defines the current evaluation protocol for ARGOS surgical stereo depth.

## Goals

ARGOS measures whether modern stereo models can recover reliable metric depth in close-range surgical scenes. The primary target is millimetric depth around wounds, tissue surfaces, tools, and fine anatomical boundaries.

## Evaluation Regimes

### Zero-Shot

Use the upstream pretrained checkpoint without surgical training.

Examples:

- S2M2-S/M pretrained.
- Fast-FoundationStereo ONNX.
- MonSter++ MixAll large.
- RAFT-Stereo RVC.
- CREStereo bundled checkpoint.

This is the cleanest baseline for generalization.

### Honest Surgical Fine-Tuning

Train on one surgical split and evaluate on a disjoint surgical split.

Current SERV-CT split:

- Train: `Experiment_1`
- Test: `Experiment_2`

This is the primary adaptation number for paper claims.

### All-Surgical Adaptation

Train/adapt using all available surgical references and evaluate on the same benchmark family.

This is useful as an upper bound, but it is not an independent generalization metric.

### Cross-Dataset Generalization

Train on one surgical dataset and evaluate on a different one.

Planned:

- Train SERV-CT, test SCARED.
- Train SCARED, test SERV-CT.
- Train SERV-CT + SCARED, test held-out surgical sequence.

## Metrics

Primary metrics:

- Disparity MAE in pixels.
- Disparity RMSE in pixels.
- Bad-1, Bad-2, Bad-5 disparity percentages.
- Depth MAE in millimeters.
- Depth RMSE in millimeters.
- Bad-1mm, Bad-2mm, Bad-5mm depth percentages.

Secondary metrics to add:

- Runtime/FPS.
- Peak GPU memory.
- Near-field depth bins.
- Boundary/detail error.
- Textureless/specular-region error.
- Confidence calibration when the model exposes confidence.

## Reporting Rules

- Always state whether a number is zero-shot, honest fine-tune, all-surgical adaptation, or cross-dataset.
- Keep all summary files as small JSON/CSV artifacts.
- Keep visual montages for qualitative inspection.
- Do not include upstream repositories, model weights, datasets, or archives in ARGOS.

## Current Primary Benchmark

SERV-CT Reference_CT:

- 16 rectified stereo frames.
- Ground-truth disparity and metric depth.
- Used for current baseline ranking.

## Paper-Ready Claim Standard

A result is suitable for paper claims only if:

- its split is explicit,
- the checkpoint provenance is explicit,
- the evaluator script is tracked,
- the output summary is tracked,
- and the result can be regenerated from local upstream repos and ARGOS scripts.
