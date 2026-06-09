# S2M2 Resolution Tradeoff On SCARED

Dataset: `/home/pampaj/Desktop/ARGOS/dataset/scared_keyframes_gt_dataset8/dataset_8`

This report focuses on S2M2-S and S2M2-L across input resolutions. S2M2-XL runs are retained as reference.

Disparity rescaling after resized inference is verified in the benchmark script:

```python
pred_disp_original = pred_disp_resized / scale_x
```

## Summary

| model | width | depth MAE | depth median | depth RMSE | disp MAE | disp RMSE | bad 2px | bad 2mm | avg ms | peak MB | params M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S | full | 2.7452 | 0.6993 | 7.3106 | 4.1667 | 13.0190 | 15.84 | 19.79 | 354.33 | 1488.2 | 26.50 |
| S | 1024 | 2.7770 | 0.7273 | 7.3293 | 4.1933 | 13.0308 | 16.16 | 20.34 | 128.93 | 1012.5 | 26.50 |
| S | 736 | 2.8225 | 0.7724 | 7.3511 | 4.2318 | 13.0402 | 16.69 | 21.19 | 88.85 | 605.3 | 26.50 |
| S | 512 | 2.8660 | 0.8206 | 7.3555 | 4.2772 | 13.0382 | 18.01 | 22.23 | 69.55 | 374.9 | 26.50 |
| L | full | 2.7100 | 0.6542 | 7.3108 | 4.1445 | 13.0356 | 15.47 | 18.99 | 485.10 | 2899.5 | 180.72 |
| L | 1024 | 2.7265 | 0.6768 | 7.3110 | 4.1591 | 13.0343 | 15.65 | 19.35 | 327.12 | 2244.7 | 180.72 |
| L | 736 | 2.7425 | 0.6919 | 7.3195 | 4.1709 | 13.0339 | 15.83 | 19.67 | 185.86 | 1671.1 | 180.72 |
| L | 512 | 2.7962 | 0.7593 | 7.3289 | 4.2146 | 13.0332 | 16.52 | 20.93 | 130.82 | 1351.4 | 180.72 |
| XL | full | 2.6963 | 0.6368 | 7.3048 | 4.1303 | 13.0270 | 15.44 | 18.83 | 888.36 | 5179.4 | 405.71 |
| XL | 1024 | 2.7086 | 0.6478 | 7.3090 | 4.1392 | 13.0239 | 15.64 | 19.04 | 573.55 | 4132.6 | 405.71 |
| XL | 736 | 2.7281 | 0.6667 | 7.3227 | 4.1529 | 13.0252 | 15.86 | 19.49 | 322.70 | 3318.1 | 405.71 |
| XL | 512 | 2.7579 | 0.6957 | 7.3431 | 4.1779 | 13.0388 | 16.11 | 20.08 | 164.64 | 2850.6 | 405.71 |

## Practical Deployment Questions

- Highest accuracy under 500 ms among S/L: `L@full` with depth MAE `2.7100 mm`, disp MAE `4.1445 px`, `485.10 ms`, and `2899.5 MB` VRAM.
- Highest accuracy under 300 ms among S/L: `L@736` with depth MAE `2.7425 mm`, disp MAE `4.1709 px`, `185.86 ms`, and `1671.1 MB` VRAM.
- Lowest VRAM S/L candidate: `S@512` at `374.9 MB`, depth MAE `2.8660 mm`. Compared with best overall `XL@full`, degradation is `0.1696 mm`.
- XL reference: XL is still the most accurate at full resolution, but the gain over L/full is small compared with runtime and VRAM.

## Pareto Frontier

S/L depth accuracy vs runtime vs VRAM frontier:

- `S@512`: depth MAE `2.8660 mm`, disp MAE `4.2772 px`, `69.55 ms`, `374.9 MB`.
- `S@736`: depth MAE `2.8225 mm`, disp MAE `4.2318 px`, `88.85 ms`, `605.3 MB`.
- `S@1024`: depth MAE `2.7770 mm`, disp MAE `4.1933 px`, `128.93 ms`, `1012.5 MB`.
- `L@736`: depth MAE `2.7425 mm`, disp MAE `4.1709 px`, `185.86 ms`, `1671.1 MB`.
- `L@1024`: depth MAE `2.7265 mm`, disp MAE `4.1591 px`, `327.12 ms`, `2244.7 MB`.
- `S@full`: depth MAE `2.7452 mm`, disp MAE `4.1667 px`, `354.33 ms`, `1488.2 MB`.
- `L@full`: depth MAE `2.7100 mm`, disp MAE `4.1445 px`, `485.10 ms`, `2899.5 MB`.

All-model frontier including XL reference:

- `S@512`: depth MAE `2.8660 mm`, `69.55 ms`, `374.9 MB`.
- `S@736`: depth MAE `2.8225 mm`, `88.85 ms`, `605.3 MB`.
- `S@1024`: depth MAE `2.7770 mm`, `128.93 ms`, `1012.5 MB`.
- `XL@512`: depth MAE `2.7579 mm`, `164.64 ms`, `2850.6 MB`.
- `L@736`: depth MAE `2.7425 mm`, `185.86 ms`, `1671.1 MB`.
- `XL@736`: depth MAE `2.7281 mm`, `322.70 ms`, `3318.1 MB`.
- `L@1024`: depth MAE `2.7265 mm`, `327.12 ms`, `2244.7 MB`.
- `S@full`: depth MAE `2.7452 mm`, `354.33 ms`, `1488.2 MB`.
- `L@full`: depth MAE `2.7100 mm`, `485.10 ms`, `2899.5 MB`.
- `XL@1024`: depth MAE `2.7086 mm`, `573.55 ms`, `4132.6 MB`.
- `XL@full`: depth MAE `2.6963 mm`, `888.36 ms`, `5179.4 MB`.

## Recommendation

For practical deployment, `L@736` is the cleanest balance under 300 ms, while `L@full` is the best S/L candidate under 500 ms. `S@512` has the lowest VRAM and fastest runtime, but gives up more accuracy. Use `XL@full` only as a teacher/reference unless larger SCARED subsets show a bigger hard-frame advantage.

Qualitative montages are in `qualitative/`.
