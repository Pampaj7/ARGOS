# EndoStreamDepth Notes

Source: `external/EndoStreamDepth` at commit `5abe89d9c0e09f64fdc5276d21bb5a34aa815cc6`.

Relevant files:

- `endostreamdepth/mamba.py`
- `endostreamdepth/model.py`
- `endostreamdepth/original_dpt.py`
- `endostreamdepth/util/loss.py`

Useful components:

- `MambaModel.start_new_sequence`: sequence-state reset.
- `MambaModel.forward_single_frame`: one-frame stateful update.
- `dpt_features_to_mamba`: frame-by-frame temporal insertion into DPT feature levels.
- `train_sequence`: multi-level supervision and temporal loss usage.
- `SiLogLoss`, `GradientEdgeLoss`, `temporal_consistency_loss`: depth loss references.

ARGOS v2 action:

- Copy the state lifecycle idea, not the DPT-coupled implementation.
- Consider multi-scale temporal state only after the minimal causal refiner is stable.

Risks:

- Full model is monocular and DepthAnything/DPT-coupled.
- Mamba dependencies are heavy.
- Temporal regularization is not motion compensated.
