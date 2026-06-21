# ARGOS Progress Presentation Summary

## Reliable headline

ARGOS now has a reproducible surgical stereo evaluation stack, SCARED temporal cache, S2M2/StereoAnyVideo cached predictions, and two learned temporal refinement families.

## Most presentation-worthy result

Unified full-frame evaluation on `test_dataset_9_keyframe_3` now evaluates against SCARED GT annotations on rectified frames.

- Raw S2M2-L temporal diff: `0.984`
- StereoAnyVideo temporal diff: `0.925`
- S2M2-L Disp. MAE: `8.331`
- StereoAnyVideo Disp. MAE: `8.251`

## Mandatory caveat

Temporal smoothness is not geometric correctness, but both are now tracked. Ground truth availability restricts evaluation to pixels with valid depth.
