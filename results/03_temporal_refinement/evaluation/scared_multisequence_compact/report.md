# Compact SCARED Multi-Sequence Generalization Benchmark

## Confirmed Progressive Sequences

- `test_dataset_9_keyframe_3` (103 frames)
- `dataset_1_keyframe_1` (12 frames)
- `dataset_1_keyframe_2` (12 frames)
- `dataset_1_keyframe_3` (12 frames)
- `dataset_5_keyframe_1` (12 frames)
- `dataset_5_keyframe_2` (12 frames)
- `dataset_5_keyframe_3` (12 frames)
- `dataset_8_keyframe_0` (12 frames)
- `dataset_8_keyframe_1` (12 frames)
- `dataset_8_keyframe_2` (12 frames)

## Excluded Sequences

- None

## Overall Summary

| method | depth MAE | disp MAE | Bad-2mm | raw temporal | motion-comp temporal | runtime ms | VRAM MB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| StereoAnyVideo | 2.8634 | 7.0171 | 39.6737 | 0.6958 | 0.5858 | 59.99 | 6977.1 |
| S2M2-S@512+EMA0.50 | 2.9755 | 7.4783 | 41.4374 | 0.8881 | 0.8341 | 60.97 | 371.3 |
| S2M2-S@512 | 3.0588 | 7.6098 | 41.4450 | 1.4092 | 1.3141 | 60.97 | 371.3 |
| Fast-FoundationStereo ONNX | 4.2377 | 12.5454 | 42.7982 | 3.0961 | 3.0613 | 45.36 | nan |
| S2M2-L@736+EMA0.50 | 4.5513 | 13.2945 | 44.6050 | 1.5785 | 1.4738 | 187.77 | 1672.5 |
| S2M2-L@736 | 4.6033 | 13.5567 | 44.4617 | 3.0862 | 2.8414 | 187.77 | 1672.5 |
| ConvGRU V2 e40 | 4.6194 | 13.4794 | 45.0020 | 2.9602 | 2.6724 | 209.95 | 2679.9 |

## Answer

S2M2-S@512 + EMA alpha 0.50 remains the best deployment-oriented configuration: `True`.
It uses `60.97 ms` and `371.3 MB` versus ConvGRU e40 `209.95 ms` and `2679.9 MB`.

## Reference Images

- `/dtu/p1/leopam/ARGOS/results/temporal evaluation/scared_multisequence_compact/reference_images/test_dataset_9_keyframe_3_adjacent_contact.png`
- `/dtu/p1/leopam/ARGOS/results/temporal evaluation/scared_multisequence_compact/reference_images/dataset_1_keyframe_1_adjacent_contact.png`
- `/dtu/p1/leopam/ARGOS/results/temporal evaluation/scared_multisequence_compact/reference_images/dataset_1_keyframe_2_adjacent_contact.png`
- `/dtu/p1/leopam/ARGOS/results/temporal evaluation/scared_multisequence_compact/reference_images/dataset_1_keyframe_3_adjacent_contact.png`
- `/dtu/p1/leopam/ARGOS/results/temporal evaluation/scared_multisequence_compact/reference_images/dataset_5_keyframe_1_adjacent_contact.png`
- `/dtu/p1/leopam/ARGOS/results/temporal evaluation/scared_multisequence_compact/reference_images/dataset_5_keyframe_2_adjacent_contact.png`
- `/dtu/p1/leopam/ARGOS/results/temporal evaluation/scared_multisequence_compact/reference_images/dataset_5_keyframe_3_adjacent_contact.png`
- `/dtu/p1/leopam/ARGOS/results/temporal evaluation/scared_multisequence_compact/reference_images/dataset_8_keyframe_0_adjacent_contact.png`
- `/dtu/p1/leopam/ARGOS/results/temporal evaluation/scared_multisequence_compact/reference_images/dataset_8_keyframe_1_adjacent_contact.png`
- `/dtu/p1/leopam/ARGOS/results/temporal evaluation/scared_multisequence_compact/reference_images/dataset_8_keyframe_2_adjacent_contact.png`
- `/dtu/p1/leopam/ARGOS/results/temporal evaluation/scared_multisequence_compact/reference_images/best_dataset_1_keyframe_2_comparison.png`
- `/dtu/p1/leopam/ARGOS/results/temporal evaluation/scared_multisequence_compact/reference_images/median_test_dataset_9_keyframe_3_comparison.png`
- `/dtu/p1/leopam/ARGOS/results/temporal evaluation/scared_multisequence_compact/reference_images/worst_dataset_5_keyframe_1_comparison.png`
- `/dtu/p1/leopam/ARGOS/results/temporal evaluation/scared_multisequence_compact/reference_images/test_dataset_9_keyframe_3_motion_comp_error.png`
