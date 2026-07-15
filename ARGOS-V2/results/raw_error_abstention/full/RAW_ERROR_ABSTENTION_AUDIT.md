# ARGOS v2 raw-error abstention audit

The canonical architecture/data/mask audit is
[`../../../model_design/RAW_ERROR_ABSTENTION_AUDIT.md`](../../../model_design/RAW_ERROR_ABSTENTION_AUDIT.md).

The frozen composition uses the validated A2 checkpoint at
`results/learned_t1_refiner/ablations/A2/checkpoints/best_validation.pt`, a
frozen SEA-RAFT adapter, and the 1,107-parameter S1 detector. Calibration used
only `dataset_7_keyframe_1/2`; final seen and one-shot unseen evaluation used
only `dataset_7_keyframe_3/4`.

