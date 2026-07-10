# BiDAVideo / BiDAStabilizer Notes

Source: `external/bidavideo` at commit `dae817df1ceaafcb865ebd9c7aa44b16c535e856`.

Relevant files:

- `models/core/bidastabilizer.py`
- `train_utils/losses.py`
- `models/sea_raft_model.py`
- `evaluation/utils/eval_utils.py`

Useful components:

- `flow_warp`: target-to-source grid sampling with `align_corners=True`.
- `BiDAStabilizer.compute_flow`: computes adjacent forward/backward flow using SEA-RAFT.
- `ResidualBlocksWithInputConv`: local disparity feature extractor over `[aligned prev, current, aligned next]`.
- `backward_resblocks` and `forward_resblocks`: hidden-state propagation blocks.
- `fusion`, `conv_hr`, `conv_last`: residual disparity output.
- `consistency_loss`: bidirectional temporal consistency using photometric exponential mask.

ARGOS v2 action:

- Reuse the warp convention.
- Reimplement the stabilizer causally: no `t+1`, no backward pass.
- Do not directly import the full BiDAStabilizer deployable path.

Risks:

- `disp_abs = -disp` sign flip is local to BiDAVideo conventions.
- Full model is non-causal.
- SEA-RAFT checkpoint path is hard-coded in `models/sea_raft_model.py`.
