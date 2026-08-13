#!/usr/bin/env python3
"""Official GT evaluation over the validated causal temporal-comparison path."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model_design.comparison.run_comparison import (
    ALL_BACKBONES, D4D_BACKBONES, DEFAULT_MODULE, _d4d, _scared, atomic_csv,
    atomic_json, check_adapter, default_output, load_factory, prepare_output, sha256, validate_cuda,
)
from model_design.metrics.unified_metrics import MetricConfig, evaluate_argos_prediction


RESULTS = ROOT.parent / "results/definitive_temporal_evaluation"


def evaluate_scared_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate one backbone/sequence bundle; adapter support remains diagnostic."""
    calibration = bundle.get("calibration")
    config = MetricConfig(**{key: calibration[key] for key in ("fx_px", "baseline_mm")}) if calibration else MetricConfig()
    report = evaluate_argos_prediction(
        raw_disparity=bundle["raw_disparity"], refined_disparity=bundle["refined_disparity"],
        gt_disparity=bundle["gt_disparity"], gt_valid=bundle["gt_valid"],
        protocol_mask=bundle["protocol_mask"], config=config,
        keyframe_mask=np.asarray(bundle["keyframe_mask"], bool)[None],
        sequence_ids=[bundle["sequence_id"]], frame_ids=bundle["frame_ids"],
    )
    report["dataset"] = bundle["dataset"]
    report["split"] = bundle["split"]
    report["protocol"] = bundle["protocol"]
    report["backbone"] = bundle["backbone"]
    report["applicability"] = {
        "depth_mm": "APPLICABLE" if calibration else "NOT_PROVIDED",
        "boundary_metrics": "NOT_PROVIDED",
        "gate_metrics": "NOT_PROVIDED",
        "risk_metrics": "NOT_PROVIDED",
        "flow_referenced_temporal_metrics": "NOT_APPLICABLE",
        "official_support": "gt_valid AND protocol_mask; adapter_support is diagnostic only",
    }
    report["diagnostics"] = {"adapter_support_coverage": float(np.asarray(bundle["adapter_support"], bool).mean()),
                             "calibration": calibration}
    json.dumps(report, allow_nan=False)
    return report


def non_gt_applicability(dataset: str) -> dict[str, Any]:
    reason = "static stereo pairs have no temporal adjacency" if dataset == "servct" else "D4D has no ground truth"
    return {"dataset": dataset.upper(), "unified_gt_metric_families": "NOT_APPLICABLE",
            "numeric_gt_metrics": None, "reason": reason}


def _value(report: Mapping[str, Any], label: str, method: str, metric: str) -> Any:
    return report.get("aggregate", {}).get(label, {}).get(method, {}).get(metric, {}).get("macro_sequence")


def _summary_row(report: Mapping[str, Any]) -> dict[str, Any]:
    row = {"dataset": report["dataset"], "split": report["split"], "backbone": report["backbone"],
           "sequence": report["sequence_ids"][0], "raw_mae_px": _value(report, "disparity_px", "raw", "MAE"),
           "refined_mae_px": _value(report, "disparity_px", "refined", "MAE"),
           "raw_bad1": _value(report, "disparity_px", "raw", "Bad1"),
           "refined_bad1": _value(report, "disparity_px", "refined", "Bad1"),
           "raw_mae_mm": _value(report, "depth_mm", "raw", "MAE"),
           "refined_mae_mm": _value(report, "depth_mm", "refined", "MAE"),
           "depth_mm": report["applicability"]["depth_mm"]}
    return row


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("scared-d2", "scared-d7", "d4d", "servct"), required=True)
    parser.add_argument("--backbones", nargs="+"); parser.add_argument("--module", default=DEFAULT_MODULE)
    parser.add_argument("--output", type=Path); parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--flow-batch-size", type=int, default=32); parser.add_argument("--max-frames", type=int)
    parser.add_argument("--sequences", nargs="+"); parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--static-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    config = arguments(); factory = load_factory(config.module)
    if config.static_check:
        adapter = factory(device="cpu"); check_adapter(adapter)
        print(json.dumps({"status": "PASS", "provenance": adapter.describe()}, sort_keys=True)); return
    config.backbones = tuple(config.backbones or (D4D_BACKBONES if config.dataset in {"d4d", "servct"} else ALL_BACKBONES))
    if config.dataset in {"d4d", "servct"} and set(config.backbones) - set(D4D_BACKBONES):
        raise ValueError("D4D/SERV-CT only have cached RAFT-Stereo and StereoAnywhere")
    gpu = None if config.dataset == "servct" else validate_cuda(config.device)
    adapter = factory(device=config.device); check_adapter(adapter)
    output = config.output or default_output(RESULTS, config)
    prepare_output(output)
    metrics_path = Path(__file__).resolve().parents[1] / "metrics/unified_metrics.py"
    evaluator_path, driver_path = Path(__file__).resolve(), Path(__file__).with_name("run_comparison.py")
    manifest = {"project": "ARGOS v2", "status": "INCOMPLETE", "dataset": config.dataset,
                "module_provenance": adapter.describe(), "backbones": list(config.backbones),
                "CUDA_VISIBLE_DEVICES": gpu, "no_gt_in_adapter": True, "dense_predictions_written": False,
                "unified_metrics": {"path": str(metrics_path), "sha256": sha256(metrics_path)},
                "evaluator": {"path": str(evaluator_path), "sha256": sha256(evaluator_path)},
                "comparison_driver": {"path": str(driver_path), "sha256": sha256(driver_path)}}
    atomic_json(output / "run_manifest.json", manifest)
    try:
        applicability: dict[str, Any]
        if config.dataset.startswith("scared-"):
            reports: list[dict[str, Any]] = []
            def save(bundle: Mapping[str, Any]) -> None:
                report = evaluate_scared_bundle(bundle); reports.append(report)
                atomic_json(output / "reports" / bundle["backbone"] / f"{bundle['sequence_id']}.json", report)
            _scared(config, adapter, save)
            rows = [_summary_row(report) for report in reports]
            atomic_csv(output / "summary.csv", rows)
            applicability = {"dataset": "SCARED-C", "unified_gt_metric_families": "APPLICABLE",
                             "adapter_support": "DIAGNOSTIC_ONLY", "flow_referenced_temporal_metrics": "NOT_APPLICABLE"}
            summary: dict[str, Any] = {"dataset": "SCARED-C", "reports": rows, "applicability": applicability}
        elif config.dataset == "d4d":
            rows, diagnostics = _d4d(config, adapter)
            atomic_csv(output / "d4d_diagnostics.csv", rows)
            applicability = non_gt_applicability(config.dataset)
            summary = {"dataset": "D4D", "diagnostics": diagnostics, "applicability": applicability}
            atomic_csv(output / "summary.csv", [{"dataset": "D4D", "numeric_gt_metrics": None}])
        else:
            applicability = non_gt_applicability(config.dataset)
            summary = {"dataset": "SERV-CT", "applicability": applicability}
            atomic_csv(output / "summary.csv", [{"dataset": "SERV-CT", "numeric_gt_metrics": None}])
        atomic_json(output / "summary.json", summary); atomic_json(output / "applicability.json", applicability)
        atomic_json(output / "run_manifest.json", manifest | {"status": "COMPLETE", "outputs": sorted(str(path.relative_to(output)) for path in output.rglob("*") if path.is_file())})
    except BaseException as error:
        atomic_json(output / "run_manifest.json", manifest | {"error": f"{type(error).__name__}: {error}"}); raise


if __name__ == "__main__":
    main()
