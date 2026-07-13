# BiDAVideo / BiDAStabilizer audit for ARGOS v2

Audited against `external/bidavideo` commit
`dae817df1ceaafcb865ebd9c7aa44b16c535e856` and
`SOTA/Match_Stereo_Videos_via_Bidirectional_Alignment.pdf`.

## Original mechanism

- The image stereo backbone is frozen in
  `train_bidastabilizer.py::fetch_optimizer`: all parameters of the selected
  `RAFTStereoModel().model` or `IGEVStereoModel().model` have
  `requires_grad=False`. The separate RAFT used by the temporal consistency loss
  is frozen too. Within `BiDAStabilizer`, parameters whose names contain `raft`
  are frozen. Only the stabilizer feature extraction, propagation, fusion, and
  residual layers train.
- `models/core/bidastabilizer.py::BiDAStabilizer.__init__` instantiates
  `models/sea_raft_model.py::SEARAFTModel`. That wrapper loads
  `third_party/SEA-RAFT/core/raft.py::RAFT` with the vendored
  `Tartan-C-T-TSKH-spring540x960-S.pth` checkpoint, four iterations, padded
  inference, and `output['flow'][-1]`. The paper's implementation-details section
  and `train_bidastabilizer.py` instead describe/use optical RAFT for the external
  temporal loss. ARGOS records this code/paper discrepancy and uses SEA-RAFT as
  the requested primary flow model, with optical RAFT only as comparator.
- Flow checkpoints used for validation are the vendored SEA-RAFT
  `Tartan-C-T-TSKH-spring540x960-S.pth` (SHA-256
  `1a21575ed6ca2c6945fb8e25c4169d241cf59ee5d12b8802c01c965206268cac`)
  and the official RAFT `raft-things.pth` from
  `third_party/RAFT/download_models.sh` (SHA-256
  `fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1`).
- SEA-RAFT and RAFT follow first-image to second-image flow. The original names
  are counterintuitive: in `BiDAStabilizer.compute_flow`,
  `raft(seq[i], seq[i+1])` is called `flow_backward` and means frame `i+1`
  target coordinates sampling frame `i`; `raft(seq[i+1], seq[i])` is called
  `flow_forward` and means frame `i` target coordinates sampling frame `i+1`.
  This is consistent with their use in target-to-source backward sampling.
- `BiDAStabilizer.flow_warp` and `train_utils/losses.py::flow_warp` construct an
  integer `(x,y)` pixel grid, add flow (`grid + flow`), normalize x by `W-1` and
  y by `H-1`, and call `torch.nn.functional.grid_sample` with bilinear mode,
  zero padding, and `align_corners=True`.
- In `BiDAStabilizer.forward`, input disparity is first negated (`disp_abs=-disp`)
  because the supported stereo wrappers emit negative disparity. The current
  disparity is concatenated with one aligned previous and one aligned next
  disparity. `feat_extract` maps these three channels to local disparity
  features.
- Global temporal propagation is explicitly bidirectional. A backward loop from
  `T-1` to `0` warps future hidden features toward each current frame and updates
  them with `backward_resblocks`. A forward loop from `0` to `T-1` warps past
  hidden features toward the current frame and updates them with
  `forward_resblocks`.
- At every frame, backward and forward propagated features are concatenated,
  fused by a 1x1 convolution, processed by `conv_hr` and `conv_last`, and added
  as an unconstrained residual to the original disparity. The sign is converted
  back before returning `[T,B,1,H,W]`.
- `forward_batch` is also non-causal: overlapping windows retain their central
  regions, and every retained output can receive future context.

## Training objectives

`train_bidastabilizer.py::forward_batch` applies:

1. `train_utils/losses.py::sequence_loss`: valid-masked L1 disparity supervision
   (one stabilizer prediction per frame; the generic implementation supports
   gamma-weighted iterative predictions with gamma 0.9).
2. `0.2 * train_utils/losses.py::consistency_loss`: both adjacent directions are
   flow-aligned; disparity disagreement is weighted by
   `sum_c exp(-50 * RGB_squared_difference)` and averaged. The paper writes the
   same spatial-plus-temporal form with lambda 0.2 for BiDAStabilizer.

## ARGOS v2 reuse decision

Reusable unchanged in convention: pairwise target-to-source backward sampling,
flow component scaling during resize, and the idea of motion-aligned disparity or
feature propagation. Reusable only as a non-causal baseline: future-disparity
local features, backward propagation, bidirectional fusion, and overlapping
window inference.

ARGOS adapts the mechanism to strictly past-only evidence. It uses positive-left
cache disparity directly, adds explicit warp support and source validity,
forward-backward and photometric reliability signals, and exposes no backbone
features, identity, cost volume, state, or confidence. The original unbounded
residual head is not copied; a future ARGOS refiner must use the bounded,
identity-preserving output specified in `ARGOS-V2/THE_PLAN.md`.
