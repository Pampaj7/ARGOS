# Phase-1 smoke and overfit evidence

Unit tests passed: 19/19 (`test_codd_style_fusion`, canonical BiDA,
photometric and temporal-clip contracts).

The six-epoch end-to-end smoke used one seen backbone and small train/validation
clips.  It kept SEA-RAFT and ResNet-18 frozen, used no future frame, had finite
cues/weights/losses, and reduced total CODD loss from 1.26617 to 1.13327
(-10.50%).

The separate one-contiguous-clip overfit used the same causal unroll and a
diagnostic higher learning rate.  It reduced total loss from 1.01649 to
0.48766 (-52.02%), with reset loss reaching zero and fusion loss decreasing.
This verifies that the dual labels and fusion head can be fitted; it is not a
validation result.

Temporary smoke/overfit outputs were removed after recording these compact
facts, following PONYTAIL policy.
