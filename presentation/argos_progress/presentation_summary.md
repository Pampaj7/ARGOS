# ARGOS Progress Presentation Summary

## Reliable headline

ARGOS now has a reproducible surgical stereo evaluation stack, SCARED temporal cache, S2M2/StereoAnyVideo cached predictions, and two learned temporal refinement families.

## Most presentation-worthy result

Unified full-frame evaluation on `test_dataset_9_keyframe_3` has no GT, so it measures temporal behavior rather than geometry.

- Raw S2M2-L temporal diff: `1.2553`
- StereoAnyVideo temporal diff: `0.9672`
- ConvGRU V2 epoch 30 temporal diff: `1.1545`
- ConvGRU V2 epoch 30 teacher-delta MAE: `0.6290`
- ConvGRU V2 epoch 30 refined-to-backbone MAE: `0.3662`

## Mandatory caveat

The current 126-frame full-frame validation sequence has `has_gt=False`. Temporal smoothness does not prove geometric correctness.
