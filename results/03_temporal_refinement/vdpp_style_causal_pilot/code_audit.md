# Code audit — VDPP-style causal pilot

- **Data**: SCARED per-sequence shards `s2m2_gt_refiner_targets_full/targets/*.npz`
  ([T,256,320] raw/gt/valid, consecutive frames, per-frame GT). Splits from
  `proposed_balanced_split.json` (train 19 / val 4 / test 4 sequences; frozen). 8-frame
  causal clips sampled within a sequence (no cross-sequence, no future frames).
- **Model** (`train_vdpp_causal.VDPPCausal`, 752k params): per-frame 5-ch geometric input
  (raw, gx, gy, edge, valid; ÷DISP_SCALE=64) → 3-conv shared encoder (hid=96) → causal
  ConvGRU over the clip → bounded residual (3·tanh, zero-init). refined = raw + residual.
  Disparity-only; S2M2 frozen; no optical flow.
- **Reuse**: `frame_metrics`/`edge_map` (OOD safety harness), D4D shards+RAFT temporal
  pipeline (`eval_vdpp_d4d` imports `temporal_eval_d4d`). No new stereo backbone.
- **Modes**: spatial | tgm | current_frame (GRU hidden reset each frame) | shuffled (clip
  order permuted before GRU). current_frame + shuffled are the mandatory time-usage ablations.
