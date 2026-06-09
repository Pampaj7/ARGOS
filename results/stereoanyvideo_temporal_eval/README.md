# StereoAnyVideo Temporal Evaluation

First ARGOS integration of StereoAnyVideo as a video-stereo upper-bound baseline.

## Scope

No training was performed. Previous benchmark folders were not modified.

Two sequences are evaluated:

- `gt5`: 5 rectified SCARED dataset_8 keyframes with GT disparity/depth. Useful for accuracy, but not a true consecutive video clip.
- `consecutive32`: 32 consecutive SCARED stereo frames without GT. Useful for temporal/flicker metrics.

## Models

- `S2M2-L@full`
- `S2M2-L@736`
- `S2M2-S@512`
- `StereoAnyVideo@384x640`

All resized disparities are rescaled back to original image coordinates:

```python
pred_disp_original = pred_disp_resized / scale_x
```

## Outputs

- `report.md`: main analysis and answers.
- `report.csv`: summary metrics.
- `report.json`: summary, per-frame metrics, and temporal metrics.
- `gt5/`: predictions, per-frame metrics, qualitative montages, and videos for the 5-frame GT sequence.
- `consecutive32/`: predictions, temporal metrics, qualitative montages, and videos for the 32-frame consecutive sequence.
- `run.log`: execution log.

## Current Takeaway

On the true 32-frame consecutive clip, StereoAnyVideo is the smoothest model:

- StereoAnyVideo@384x640 mean consecutive disparity diff: `1.0214`
- S2M2-S@512: `1.2217`
- S2M2-L@736: `1.6737`
- S2M2-L@full: `8.6276`

On the 5-frame GT sequence, StereoAnyVideo is essentially tied with S2M2-L@full in depth MAE:

- StereoAnyVideo@384x640: `2.7090 mm`
- S2M2-L@full: `2.7100 mm`
- S2M2-L@736: `2.7425 mm`

StereoAnyVideo is a strong temporal teacher/reference. It is promising as an upper-bound baseline, but the 32-frame run uses about `10.1 GB` VRAM, so S2M2-L@736 remains the practical deployment baseline for now.

