# S2M2 Size Tradeoff On SCARED

Dataset: /home/pampaj/Desktop/ARGOS/dataset/scared_keyframes_gt_dataset8/dataset_8

Note: the requested converted path `stereo/Fast-FoundationStereo/data/surgical_stereo/scared_keyframes` is not present in this workspace, so this run uses the current ARGOS rectified SCARED dataset_8 keyframe subset.

All disparities are rescaled back to original image coordinates after input resizing with `pred_disp_original = pred_disp_resized / scale_x`.

## Summary

| model | width | disp MAE | depth MAE | depth RMSE | bad 2px | bad 2mm | avg ms | median ms | peak MB | params M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S | full | 4.1667 | 2.7452 | 7.3106 | 15.84 | 19.79 | 353.84 | 162.31 | 1488.2 | 26.50 |
| S | 1024 | 4.1933 | 2.7770 | 7.3293 | 16.16 | 20.34 | 127.40 | 116.90 | 1012.5 | 26.50 |
| S | 736 | 4.2318 | 2.8225 | 7.3511 | 16.69 | 21.19 | 88.35 | 75.92 | 605.3 | 26.50 |
| S | 512 | 4.2772 | 2.8660 | 7.3555 | 18.01 | 22.23 | 69.78 | 60.85 | 374.9 | 26.50 |
| L | full | 4.1445 | 2.7100 | 7.3108 | 15.47 | 18.99 | 486.62 | 484.18 | 2899.5 | 180.72 |
| L | 1024 | 4.1591 | 2.7265 | 7.3110 | 15.65 | 19.35 | 328.08 | 327.64 | 2244.7 | 180.72 |
| L | 736 | 4.1709 | 2.7425 | 7.3195 | 15.83 | 19.67 | 187.03 | 182.53 | 1671.1 | 180.72 |
| L | 512 | 4.2146 | 2.7962 | 7.3289 | 16.52 | 20.93 | 130.52 | 126.75 | 1351.4 | 180.72 |
| XL | full | 4.1303 | 2.6963 | 7.3048 | 15.44 | 18.83 | 890.31 | 888.56 | 5179.4 | 405.71 |
| XL | 1024 | 4.1392 | 2.7086 | 7.3090 | 15.64 | 19.04 | 574.37 | 571.04 | 4132.6 | 405.71 |
| XL | 736 | 4.1529 | 2.7281 | 7.3227 | 15.86 | 19.49 | 322.53 | 319.60 | 3318.1 | 405.71 |
| XL | 512 | 4.1779 | 2.7579 | 7.3431 | 16.11 | 20.08 | 164.57 | 161.47 | 2850.6 | 405.71 |

## Analysis

1. Best depth MAE is `2.6963 mm` from `XL` at `full`.
   At width 1024, XL vs L depth MAE delta is `-0.0179 mm`.
   At width 1024, XL vs S depth MAE delta is `-0.0685 mm`.
2. Best disparity MAE is `4.1303 px` from `XL` at `full`.
   At width 1024, XL vs L disparity MAE delta is `-0.0198 px`.
   At width 1024, XL vs S disparity MAE delta is `-0.0541 px`.
3. Lowest `pred_disp <= 0.5` ratio is `0.000000` from `S` at `full`; compare this with average error to decide if XL reduces catastrophic failures or only mean error.
4. Resize width 1024 better than full for XL? no. XL full depth MAE `2.6963`, XL 1024 `2.7086`.
5. Close faster candidate: `S` at `512` with depth MAE `2.8660` and `69.78 ms`.
6. Recommendations:
   - default evaluation baseline: `L` at `full`, because it is nearly tied with XL while being much cheaper.
   - real-time candidate: `S` at `512` for fastest inference; `S` at `736` is the safer speed/accuracy compromise.
   - teacher for future distillation: `XL` at `full`, but only as a teacher/reference model, not as the routine baseline unless larger SCARED runs show a larger hard-frame benefit.

Qualitative montages are in `qualitative/`.
