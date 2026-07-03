# S2M2 GT Refiner Targets Full

Compact low-resolution supervised targets for tiny temporal refinement.

- Dataset root: `/dtu/p1/leopam/ARGOS/dataset/SCARED/curated/temporal_gt_rectified`
- Valid frame source: `/dtu/p1/leopam/ARGOS/results/03_temporal_refinement/evaluation/gt_temporal_rectified_streaming_s2m2_v2_artifact_temporal/frame_metrics.csv`
- Frames: `20621`
- Sequences: `27`
- Target scale: `0.25`
- Valid-mask-aware downsample min valid ratio: `0.25`
- Saved arrays per sequence shard: `raw_disp`, `gt_disp`, `valid_mask`, `delta_disp_gt_minus_raw`
- Full-resolution prediction caches: not written
- SAV/RAFT/DINO: not run

Each `.npz` shard is under `targets/<sequence_id>.npz`; `frame_targets_index.csv` maps every frame to `target_path` and `frame_offset`.
This avoids one-file-per-frame allocation waste on filesystems with large block sizes while preserving per-frame metadata and temporal links.
