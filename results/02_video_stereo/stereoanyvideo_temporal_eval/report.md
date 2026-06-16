# StereoAnyVideo Temporal Evaluation

This run integrates StereoAnyVideo as the first video-stereo upper-bound baseline and compares it with frame-based S2M2 baselines.

Sequences:

- `gt5`: the 5-frame ARGOS/SCARED smoke sequence with GT disparity/depth. These frames are clean keyframes and are not guaranteed to be temporally consecutive.
- `consecutive32`: 32 consecutive SCARED stereo frames without GT, used for true temporal/flicker metrics.

All resized disparities are rescaled back to original image coordinates with `pred_disp_original = pred_disp_resized / scale_x`.

## Summary

| sequence | model | disp MAE | depth MAE | bad 2px | bad 2mm | disp diff | depth diff | temporal std | error variation | runtime ms | peak MB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gt5 | S2M2-L@full | 4.1445 | 2.7100 | 15.47 | 18.99 | 16.8292 | 17.9983 | 18.6913 | 6.6162 | 614.65 | 2898.0 |
| gt5 | S2M2-L@736 | 4.1709 | 2.7425 | 15.83 | 19.67 | 16.8488 | 18.0208 | 18.7005 | 6.6196 | 192.63 | 1671.1 |
| gt5 | S2M2-S@512 | 4.2772 | 2.8660 | 18.01 | 22.23 | 16.8823 | 18.0845 | 18.7253 | 6.6521 | 74.98 | 369.0 |
| gt5 | StereoAnyVideo@384x640 | 4.1624 | 2.7090 | 15.86 | 18.81 | 16.8049 | 17.9698 | 18.6644 | 6.6481 | 167.89 | 1748.2 |
| consecutive32 | S2M2-L@full | nan | nan | nan | nan | 8.6276 |  | 27.4150 |  | 486.46 | 2897.8 |
| consecutive32 | S2M2-L@736 | nan | nan | nan | nan | 1.6737 |  | 9.4113 |  | 182.75 | 1671.1 |
| consecutive32 | S2M2-S@512 | nan | nan | nan | nan | 1.2217 |  | 8.3814 |  | 61.33 | 369.0 |
| consecutive32 | StereoAnyVideo@384x640 | nan | nan | nan | nan | 1.0214 |  | 7.5718 |  | 144.80 | 10133.6 |

## Answers

1. Temporal consistency: on the true `consecutive32` sequence, StereoAnyVideo has mean consecutive disparity diff `1.0214` vs S2M2-L@736 `1.6737`. Lower is smoother. This run therefore gives the first direct flicker comparison, but without GT on the consecutive clip.
2. Accuracy on `gt5`: StereoAnyVideo depth MAE `2.7090 mm` vs S2M2-L@full `2.7100 mm` and S2M2-L@736 `2.7425 mm`. Disparity MAE is `4.1624 px` vs `4.1445 px` and `4.1709 px`.
3. Runtime/VRAM: StereoAnyVideo@384x640 costs `167.89 ms/frame` and `1748.2 MB` on `gt5`; compare S2M2-L@736 `192.63 ms/frame`, `1671.1 MB`.
4. Practical use: StereoAnyVideo is useful now as an upper-bound/teacher-style video baseline. At reduced resolution and short window it is not absurdly far from deployment, but S2M2-L@736 remains the practical baseline until timed larger clips prove the video prior buys enough stability.
5. Temporal teacher: yes. StereoAnyVideo should be used as the first temporal teacher/reference for a future lightweight stabilizer or distillation experiment.

Qualitative montages are in `gt5/qualitative/` and `consecutive32/qualitative/`. Short MP4 visualizations are in each sequence's `videos/` folder.
