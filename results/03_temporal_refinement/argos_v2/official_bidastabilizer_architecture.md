# Official BiDAStabilizer Architecture Audit

## External Reference

- Repository: `https://github.com/TomTomTommi/bidavideo.git`
- Local clone: `external/bidavideo/`
- Commit: `dae817df1ceaafcb865ebd9c7aa44b16c535e856`
- License: MIT

## Source Files Followed

- `models/core/bidastabilizer.py`: stabilizer architecture and forward pass.
- `train_bidastabilizer.py`: training call graph and losses.
- `demo.py`: inference call graph into stereo model wrappers and stabilizer.
- `models/raft_stereo_model.py`, `models/igev_stereo_model.py`: `forward_stabilizer` wrappers.
- `train_utils/losses.py`: sequence loss and temporal consistency loss.
- `models/core/utils/utils.py`: `InputPadder`.

## Official Runtime Call Graph

Training:

```text
train_bidastabilizer.py::forward_batch
  frozen stereo model per frame -> disparities [T,B,1,H,W]
  model_stabilizer(left_video, disparities.permute(1,0,2,3,4))
  sequence_loss(stabilized, GT)
  0.2 * consistency_loss(left_video, stabilized, frozen RAFT)
```

Demo/eval:

```text
demo.py
  stereo wrapper forward_stabilizer
    per-frame stereo disparity
    BiDAStabilizer.forward_batch(left_video, disparity_sequence)
```

## Module Tree

```text
BiDAStabilizer(mid_channels=48, num_blocks=5)
  raft = SEARAFTModel()                         # embedded flow estimator, frozen in training
  feat_extract = ResidualBlocksWithInputConv(3 -> 48, 5 blocks)
  backward_resblocks = ResidualBlocksWithInputConv(96 -> 48, 5 blocks)
  forward_resblocks  = ResidualBlocksWithInputConv(96 -> 48, 5 blocks)
  fusion = Conv2d(96 -> 48, kernel=1)
  conv_hr = Conv2d(48 -> 64, kernel=3, padding=1)
  conv_last = Conv2d(64 -> 1, kernel=3, padding=1)
  lrelu = LeakyReLU(0.1)
```

`ResidualBlocksWithInputConv`:

```text
Conv2d(in_channels -> out_channels, 3x3, pad=1, bias=True)
LeakyReLU(0.1, inplace=True)
N x ResidualBlockNoBN(out_channels)
```

`ResidualBlockNoBN`:

```text
identity + Conv2d(C->C, 3x3,pad=1) -> ReLU -> Conv2d(C->C,3x3,pad=1)
```

No normalisation layers are used inside the stabilizer blocks.

## Layer-by-Layer Tensor Shapes

Inputs to official `forward(seq1, disp)`:

- `seq1`: `[B,T,3,H,W]` left RGB video.
- `disp`: `[B,T,1,H,W]` disparity sequence from frozen stereo backbone.

Flow:

- `flow_forward`: `[B,T-1,2,H,W]`.
- `flow_backward`: `[B,T-1,2,H,W]`.

Official code comments are confusing, but the warp function is clear:

```text
output(p) = source(p + flow(p))
```

Local disparity stack:

- `disp_abs = -disp`.
- `disp_backward`: next-frame disparity warped to current, with last frame self-filled, `[B,T,1,H,W]`.
- `disp_forward`: previous-frame disparity warped to current, with first frame self-filled, `[B,T,1,H,W]`.
- `disp_concate = cat([disp_forward, disp_abs, disp_backward], dim=2)`: `[B,T,3,H,W]`.
- reshape to `[B*T,3,H,W]`.
- `feat_extract`: `[B*T,48,H,W]`, reshaped to `[B,T,48,H,W]`.

Backward propagation:

- initial hidden `[B,48,H,W] = 0`.
- iterate `i=T-1..0`.
- hidden warped with `flow_backward[:, i]` when `i<T-1`.
- concatenate `[local_feat_i, hidden]`: `[B,96,H,W]`.
- `backward_resblocks`: `[B,48,H,W]`.

Forward propagation:

- initial hidden `[B,48,H,W] = 0`.
- iterate `i=0..T-1`.
- hidden warped with `flow_forward[:, i-1]` when `i>0`.
- concatenate `[local_feat_i, hidden]`: `[B,96,H,W]`.
- `forward_resblocks`: `[B,48,H,W]`.

Fusion/output:

- concatenate backward and forward propagated features: `[B,96,H,W]`.
- `fusion 1x1 + LeakyReLU`: `[B,48,H,W]`.
- `conv_hr 3x3 + LeakyReLU`: `[B,64,H,W]`.
- `conv_last 3x3`: `[B,1,H,W]` residual in official negative-disparity convention.
- add base `disp_abs[:,i]`.
- return negated output stacked as `[T,B,1,H,W]`.

## Parameter Count

The official trainable stabilizer core excluding embedded SEA-RAFT has `740,849` parameters for `mid_channels=48`, `num_blocks=5`.

The official class also instantiates `SEARAFTModel`; training freezes names containing `raft`, so the architectural stabilizer proper is the 740,849-parameter core above.

## Weight Sharing

- Feature extractor is shared across all frames.
- Forward and backward propagation blocks do not share weights.
- The same output head is applied at every time step.

## Initialisation

The official code uses PyTorch default initialisation. It does not zero-initialise the residual head and does not include a safety gate.

## Warping and Flow Convention

Official `flow_warp` uses `grid + flow`, then `grid_sample(..., padding_mode='zeros', align_corners=True)`. This is target-to-source sampling for pulling a source tensor into the current frame.

ARGOS v2 must therefore pass `flow(t -> t-1)` to warp previous disparity/hidden state into frame `t` coordinates.

## Occlusion / Mask Handling

The official `BiDAStabilizer.forward` does not use explicit validity or occlusion masks. The training temporal consistency loss uses image-difference exponential weights, not explicit forward/backward occlusion support.

ARGOS v2 safe adaptation should add reliability-mask-aware propagation and loss masking.

## Losses

Official training uses:

- spatial sequence L1-like loss via `sequence_loss` against disparity GT;
- temporal consistency loss `0.2 * consistency_loss(...)` using frozen RAFT alignment and RGB-derived exponential masks.

## Differences Between Paper-Style Description and Code

- The official code embeds SEA-RAFT directly in `BiDAStabilizer`, although training freezes it.
- Flow naming in comments is easy to misread; the warp implementation defines the actual convention.
- `BiDAStabilizer.forward_batch` chunks long videos with overlapping windows, which is offline/chunked and not true streaming state persistence.
- No bounded residual, safety gate, raw-good preservation, or explicit validity/occlusion mask is present in the official stabilizer.

## Ambiguities / Issues

- Official `compute_flow` comments and function argument ordering are ambiguous without knowing `SEARAFTModel.forward_fullres` semantics. The unambiguous fact is that every warp samples `source(p + flow(p))`.
- The official output uses a negative-disparity internal convention (`disp_abs=-disp`, return `-out`). ARGOS v2 uses positive disparity and adapts algebra accordingly.
