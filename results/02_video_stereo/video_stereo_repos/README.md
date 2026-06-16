# Video Stereo Repository Scouting

Repository scouting and smoke tests for video-stereo methods in ARGOS.

## Scope

No training was performed. No global dependency installation was performed. Repositories were kept isolated under `external/video_stereo_repos/`.

## Main Outputs

- `report.md`: human-readable scouting report and recommendations.
- `report.csv`: repository metadata table.
- `report.json`: repository metadata plus smoke metrics.
- `test_sequence/`: 5-frame rectified SCARED dataset_8 smoke-test package with left/right images, GT disparity, GT depth, valid masks, and metadata.
- `<model_name>/`: command, stdout/stderr, environment notes, and outputs for each smoke test.
- `smoke_metrics.csv` and `smoke_metrics.json`: metrics for StereoAnyVideo and S2M2 comparison baselines on the shared sequence.

## Current Takeaway

StereoAnyVideo is the only video-stereo repo that currently runs on ARGOS/SCARED custom stereo folders with a local checkpoint. It reaches depth MAE `2.7090 mm` on the 5-frame smoke sequence at `384x640`, comparable to S2M2-L full on this small subset.

Next integration target should be TC-Stereo if its Dropbox checkpoint is accessible; otherwise DynamicStereo real-data config is the next most concrete path.

