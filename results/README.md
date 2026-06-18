# ARGOS Stereo Results

This folder stores ARGOS-owned outputs that are small enough to keep organized: reports, scoreboards, plots, JSON/CSV metrics, smoke logs, and presentation images.

## Current Contents

| Path | Purpose |
|---|---|
| `images/` | Portable montage/scoreboard PNGs. |
| `video_stereo_repo_scouting/` | Scouting report for temporal/video stereo repositories. |
| `stereoanyvideo_scared_smoke/` | Queued/automatic Stereo Any Video smoke test on 5 rectified SCARED frames. |
| `temporal_context_comparison_scared5/` | First sequence comparison: Stereo Any Video vs S2M2-S vs Fast-FoundationStereo on 5 SCARED frames. |
| `stereoanyvideo_scared_consecutive32/` | Stereo Any Video run on 32 truly consecutive SCARED stereo-video frames. |
| `fastfoundation_scared_consecutive32/` | Fast-FoundationStereo ONNX run logs for the 32-frame SCARED consecutive clip. |
| `temporal_context_comparison_scared_consecutive32/` | Corrected temporal comparison on 32 consecutive SCARED frames, with depth montages and temporal error maps: Stereo Any Video vs S2M2-S vs Fast-FoundationStereo. |

## Convention

New experiment outputs should follow:

```text
results/<experiment_name>/
  report.md
  report.json
  metrics/
  images/
  logs/
```

Do not store raw datasets or large checkpoints here.
