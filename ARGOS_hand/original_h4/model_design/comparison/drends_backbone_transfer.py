"""DRENDS evaluation with a selectable frozen stereo backbone.

`drends_evaluation.py` is pinned by several freeze manifests and is hardwired to
RAFT-Stereo, so it is imported unchanged here rather than edited.  This sibling
reuses its validated record loading, canonical-frame contract, depth conversion
and flow adapter, and swaps only the stereo predictor, so that the joint
unseen-backbone x external-domain cell can be measured.

The temporal module, its checkpoint, the H=4 reset protocol and the metric
implementation are untouched.  Nothing is trained and no threshold is tuned.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from model_design.comparison import drends_evaluation as base

ROOT = base.ROOT
CANONICAL_SIZE = base.CANONICAL_SIZE
SEEN_BACKBONES = ("S2M2-S", "RAFT-Stereo", "StereoAnywhere")
UNSEEN_BACKBONES = ("CREStereo", "Fast-FoundationStereo")
ALL_BACKBONES = SEEN_BACKBONES + UNSEEN_BACKBONES
# Every DRENDS recording. This used to default to Vid14 alone, which is a quiet way to
# produce a one-sequence run that still labels itself "complete_recordings" and then gets
# averaged against a five-sequence baseline as if the two were comparable. That happened,
# and it invented a threefold improvement out of nothing. The default is now the full set.
ALL_RECORDINGS = ("Vid10_Liver_Med", "Vid11_Liver_High", "Vid12_Pancreas_Ext",
                  "Vid13_Pancreas_Med", "Vid14_Pancreas_High")


def _load_backbone(name: str, device: Any) -> tuple[str, Any]:
    """Uniform predictor for any of the five frozen estimators."""
    source = ROOT.parent.parent / "ARGOS-V2/scripts"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from argos_v2.backbones import build_predictor
    method, checkpoint, predict = build_predictor(name, device)
    return f"{method} ({checkpoint})", predict


def evaluate_drends_backbone(*, output: Path, module_spec: str, device_name: str, backbone: str,
                             recordings: list[str] | None, max_frames: int | None) -> None:
    import torch
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT.parent.parent / "ARGOS_FREEZED/src")]
    from argos_freezed.alignment.sea_raft_adapter import SEARAFTFlowAdapter
    from model_design.comparison.run_comparison import (atomic_json, check_adapter, drive, load_factory,
                                                        prepare_output, sha256, validate_cuda)
    from model_design.metrics.unified_metrics import MetricConfig, evaluate_argos_prediction

    if backbone not in ALL_BACKBONES:
        raise ValueError(f"unknown backbone: {backbone}")
    selected = recordings or list(ALL_RECORDINGS)
    if len(selected) != len(set(selected)):
        raise ValueError("duplicate DRENDS recording")
    status = "training_seen" if backbone in SEEN_BACKBONES else "training_unseen"

    gpu = validate_cuda(device_name)
    adapter = load_factory(module_spec)(device=device_name)
    check_adapter(adapter)
    prepare_output(output)
    metrics_path = ROOT / "model_design/metrics/unified_metrics.py"
    manifest = {
        "project": "ARGOS v2", "status": "INCOMPLETE", "dataset": "drends", "backbones": [backbone],
        "backbone_training_status": status, "module_provenance": adapter.describe(),
        "CUDA_VISIBLE_DEVICES": gpu, "no_gt_in_adapter": True, "dense_predictions_written": False,
        "training_performed": False, "threshold_tuning_performed": False,
        "unified_metrics": {"path": str(metrics_path), "sha256": sha256(metrics_path)},
        "evaluator": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "reused_pinned_evaluator": {"path": str(Path(base.__file__).resolve()),
                                    "sha256": sha256(Path(base.__file__).resolve()),
                                    "note": "imported unchanged; only the stereo predictor is replaced"},
    }
    atomic_json(output / "run_manifest.json", manifest)
    try:
        flow_model = SEARAFTFlowAdapter(device=torch.device(device_name))
        checkpoint, predict = _load_backbone(backbone, torch.device(device_name))
        manifest = manifest | {"flow": {"checkpoint": str(flow_model.checkpoint), "sha256": sha256(flow_model.checkpoint)},
                               "backbone": {"name": backbone, "checkpoint": checkpoint, "training_status": status}}
        atomic_json(output / "run_manifest.json", manifest)
        reports = []
        for recording in selected:
            records, info = base.load_drends_records(recording, max_frames)
            frames, (height, width) = base._canonical_frames(records, predict, device=device_name)
            scale = CANONICAL_SIZE[0] / width
            raw = [frame["raw"][0, 0].cpu().numpy() for frame in frames]

            def flow(current: Mapping[str, Any], past: Mapping[str, Any]):
                return base._drends_flow(flow_model, current, past)

            outputs = dict(drive(adapter, frames, flow))
            depth_values, valid_values, coverages = zip(
                *(base._depth(item["_depth_left"], item["_mask_left"], scale) for item in records))
            product_mm = info["focal_baseline_native_px_m"] * 1000.0 * scale
            gt = [product_mm / np.maximum(value, 1e-6) for value in depth_values]
            refined = [outputs[index]["disparity"][0, 0].detach().cpu().numpy() for index in range(len(records))]
            raw_depth = [base._prediction_depth_mm(value, product_mm) for value in raw]
            refined_depth = [base._prediction_depth_mm(value, product_mm) for value in refined]
            report = evaluate_argos_prediction(
                raw_disparity=np.asarray(raw), refined_disparity=np.asarray(refined), gt_disparity=np.asarray(gt),
                raw_depth=np.asarray(raw_depth), refined_depth=np.asarray(refined_depth),
                gt_depth=np.asarray(depth_values), depth_input_unit="mm",
                gt_valid=np.asarray(valid_values), protocol_mask=np.asarray(valid_values), config=MetricConfig(),
                keyframe_mask=np.asarray([index == 0 or bool(outputs[index]["reset"])
                                          for index in range(len(records))])[None],
                sequence_ids=[recording], frame_ids=[item["frame_id"] for item in records])
            report.update({
                "dataset": "DRENDS", "split": "external", "backbone": backbone, "backbone_status": status,
                "protocol": "chronological rectified frames; fixed H=4 causal recurrence; reset at recording boundary",
                "applicability": {"depth_reference": "APPLICABLE_WITH_CAVEAT",
                                  "independent_stereo_disparity_gt": "NOT_APPLICABLE",
                                  "caveat": "depth reference is temporally smoothed ToF and is not independent disparity ground truth"},
                "diagnostics": {"adapter_support_coverage": float(np.mean([outputs[i]["support"].float().mean().item() for i in outputs])),
                                "backbone_checkpoint": checkpoint, "native_resolution": [width, height],
                                "canonical_resolution": list(CANONICAL_SIZE), "disparity_scale": scale,
                                "coverage_area_mean": float(np.mean(coverages)), **info}})
            json.dumps(report, allow_nan=False)
            atomic_json(output / "reports" / backbone / f"{recording}.json", report)
            reports.append(report)
            print(f"DRENDS {backbone}/{recording}: {len(records)} frames", flush=True)
        atomic_json(output / "summary.json", {
            "dataset": "DRENDS", "backbone": backbone, "backbone_training_status": status,
            "recordings": selected, "status": "pilot" if max_frames is not None else "complete_recordings",
            "reports": [str(report["sequence_ids"][0]) for report in reports]})
        atomic_json(output / "run_manifest.json", manifest | {
            "status": "COMPLETE",
            "outputs": sorted(str(path.relative_to(output)) for path in output.rglob("*") if path.is_file())})
    except BaseException as error:
        atomic_json(output / "run_manifest.json", manifest | {"error": f"{type(error).__name__}: {error}"})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", required=True, choices=ALL_BACKBONES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--module", default="model_design.comparison.canonical_h4:factory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--recordings", nargs="+")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    evaluate_drends_backbone(output=args.output, module_spec=args.module, device_name=args.device,
                             backbone=args.backbone, recordings=args.recordings, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
