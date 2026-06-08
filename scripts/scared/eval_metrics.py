import numpy as np


def disp_metrics(pred, gt, mask, prefix=""):
    err = np.abs(pred[mask] - gt[mask])
    return {
        f"{prefix}valid_px": int(mask.sum()),
        f"{prefix}valid_pixel_ratio": float(mask.mean()),
        f"{prefix}mae_px": float(err.mean()),
        f"{prefix}rmse_px": float(np.sqrt((err**2).mean())),
        f"{prefix}bad1_pct": float((err > 1.0).mean() * 100.0),
        f"{prefix}bad2_pct": float((err > 2.0).mean() * 100.0),
        f"{prefix}bad5_pct": float((err > 5.0).mean() * 100.0),
    }


def depth_metrics(pred_mm, gt_mm, mask, prefix=""):
    err = np.abs(pred_mm[mask] - gt_mm[mask])
    return {
        f"{prefix}depth_mae_mm": float(err.mean()),
        f"{prefix}depth_median_abs_error_mm": float(np.median(err)),
        f"{prefix}depth_rmse_mm": float(np.sqrt((err**2).mean())),
        f"{prefix}depth_bad1mm_pct": float((err > 1.0).mean() * 100.0),
        f"{prefix}depth_bad2mm_pct": float((err > 2.0).mean() * 100.0),
        f"{prefix}depth_bad5mm_pct": float((err > 5.0).mean() * 100.0),
        f"{prefix}abs_depth_error_p50": float(np.percentile(err, 50)),
        f"{prefix}abs_depth_error_p75": float(np.percentile(err, 75)),
        f"{prefix}abs_depth_error_p90": float(np.percentile(err, 90)),
        f"{prefix}abs_depth_error_p95": float(np.percentile(err, 95)),
        f"{prefix}abs_depth_error_p99": float(np.percentile(err, 99)),
    }


def failure_aware_metrics(pred_disp, pred_depth, gt_disp, gt_depth, gt_mask, raw_mask):
    metrics = {}
    metrics.update(disp_metrics(pred_disp, gt_disp, raw_mask))
    metrics.update(depth_metrics(pred_depth, gt_depth, raw_mask))

    gt_valid_px = max(int(gt_mask.sum()), 1)
    metrics["pred_disp_le_0_1_ratio"] = float((gt_mask & (pred_disp <= 0.1)).sum() / gt_valid_px)
    metrics["pred_disp_le_0_5_ratio"] = float((gt_mask & (pred_disp <= 0.5)).sum() / gt_valid_px)
    metrics["excluded_pred_disp_le_0_1_ratio"] = metrics["pred_disp_le_0_1_ratio"]

    valid_pred_mask = gt_mask & np.isfinite(pred_depth) & (pred_disp > 0.1)
    metrics.update(disp_metrics(pred_disp, gt_disp, valid_pred_mask, prefix="valid_disp_"))
    metrics.update(depth_metrics(pred_depth, gt_depth, valid_pred_mask, prefix="valid_disp_"))

    pred_depth_cap100 = np.minimum(pred_depth, 100.0)
    pred_depth_cap200 = np.minimum(pred_depth, 200.0)
    err100 = np.abs(pred_depth_cap100[raw_mask] - gt_depth[raw_mask])
    err200 = np.abs(pred_depth_cap200[raw_mask] - gt_depth[raw_mask])
    metrics["depth_mae_cap100_mm"] = float(err100.mean())
    metrics["depth_rmse_cap100_mm"] = float(np.sqrt((err100**2).mean()))
    metrics["depth_mae_cap200_mm"] = float(err200.mean())
    metrics["depth_rmse_cap200_mm"] = float(np.sqrt((err200**2).mean()))
    return metrics


def summarize_rows(rows, skip_keys=("dataset", "frame", "gt_source")):
    numeric_keys = [k for k, v in rows[0].items() if k not in skip_keys and isinstance(v, (int, float))]
    summary = {k: float(np.mean([r[k] for r in rows])) for k in numeric_keys}
    summary["frames"] = len(rows)
    return summary
