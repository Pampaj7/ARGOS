# Cache readiness — manifest-only

This directory is an integrity record, not a cache builder. StereoMIS is blocked: full/raw stereo caches and flow caches are absent. D4D has two verified raw-disparity cache copies (RAFT-Stereo and StereoAnywhere), each with 416 rows at `144x180`; their 832 aggregate rows conflict with the unpinned stated 156-frame causal subset. Flow cache is absent. No D4D temporal inference may start until the source/window manifest is pinned and reconciled. Sparse Zivid anchors remain `239 usable / 362 total` and are not dense GT.

Future planned commands only (do not execute from this record):

```bash
# after a frozen source/window manifest and cache protocol are approved
python scripts/build_multidomain_backbone_cache.py --domain StereoMIS
python scripts/build_or_validate_flow_cache.py --domain StereoMIS --ages 1 2 4 8
python scripts/build_or_validate_flow_cache.py --domain D4D --ages 1 2 4 8
```

All commands require a separate frozen protocol; they must preserve direct current-to-anchor flow and must never touch D7.
