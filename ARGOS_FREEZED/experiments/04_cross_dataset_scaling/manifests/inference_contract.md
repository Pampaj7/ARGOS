# Frozen inference contract

All evaluated temporal calls reuse the geometry_v1/validated H4 paths: direct target-to-source SEA-RAFT flow, BiDA-style pull alignment, positive-left disparity, no composed flow, no future frames, and frozen checkpoints. Immutable uses independent raw CS1/2/4/8 anchors and exact raw fallback; H4 alone carries corrected state and resets according to its baseline manifest. The audit records metrics only and never writes dense predictions or changes thresholds. D7 paths and IDs are rejected before any model load.
