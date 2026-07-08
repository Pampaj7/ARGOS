# ARGOS v2 Aligned-Local-Only Architecture

## Relation to Official BiDAStabilizer

Official BiDAStabilizer builds local disparity features from three aligned maps:

```text
[previous aligned, current, next aligned]
```

Aligned-local-only removes future access and persistent propagation. It keeps the official local feature block style but adapts the input to:

```text
[previous aligned, current]
```

## Module Tree

### `AlignedLocalOnlyFaithful`

```text
ResidualBlocksWithInputConv(2 -> 48, 5 official no-BN residual blocks)
fusion Conv2d(48 -> 48, 1x1)
conv_hr Conv2d(48 -> 64, 3x3)
conv_last Conv2d(64 -> 1, 3x3)
optional residual_bound
```

### `AlignedLocalOnlySafe`

Same faithful core plus gate head:

```text
Conv2d(56 -> 64, 3x3)
LeakyReLU(0.1)
Conv2d(64 -> 1, 3x3)
sigmoid gate
D_ref = D_raw + gate * bounded_delta
```

Gate inputs are local feature, signed disparity difference, reliability mask, optional current RGB, and RGB temporal difference.

## Tensor Shapes

At frame `t`:

- `D_t`: `[B,1,H,W]`
- `D_{t-1}`: `[B,1,H,W]`
- `flow(t->t-1)`: `[B,2,H,W]`
- `D_prev_warped`: `[B,1,H,W]`
- local input: `[B,2,H,W]`
- local feature: `[B,48,H,W]`
- residual/gate: `[B,1,H,W]`

## Parameter Count

- `AlignedLocalOnlyFaithful`: `239,393`
- `AlignedLocalOnlySafe`: `272,290`

## Difference From FaithfulCausalBiDA

Aligned-local-only has no hidden state and no propagation. FaithfulCausalBiDA warps and updates a persistent 48-channel hidden state.

## Difference From Current-Only

Current-only has no previous-frame evidence. Aligned-local-only explicitly warps previous disparity into current-frame coordinates and compares it to current raw disparity.

## Difference From Unaligned Concat

Unaligned concat would feed previous disparity without target-to-source flow alignment. Aligned-local-only uses the validated warp convention before feature extraction.

## Faithful vs Safe Variants

- Faithful: direct official-style residual, no gate by default.
- Safe: bounded gated residual with reliability-mask diagnostics.
