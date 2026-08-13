#!/usr/bin/env python3
"""Frozen raw-versus-canonical-H4 SCARED-C transfer evaluator.

The default is the paper protocol: D2, warm-up index 8, and the strict
all-anchor mask.  D7 is deliberately a separately named H4-only-support
protocol; its numbers are never mixed with the paper table.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from canonical_h4_provenance import sha256, verify_canonical_inputs


ROOT = Path(__file__).resolve().parents[1]
ARGOS = ROOT.parents[1]
FROZEN = ARGOS / "ARGOS_FREEZED"
OUT = ROOT.parent / "results" / "frozen_transfer_eval"
CHECKPOINT = ROOT / "model_design" / "checkpoints" / "codd_style_h4_best_validation.pt"
POLICY = ROOT / "model_design" / "checkpoints" / "codd_style_h4_policy.json"
INFERENCE_MANIFEST = ROOT / "model_design" / "checkpoints" / "inference_manifest.json"
CHECKPOINT_SHA256 = "99c5745c164fd4903b8aa8acf8f57efccacd9cdcd0d4ed4305cd10609324d725"
BACKBONES = ("S2M2-S", "RAFT-Stereo", "StereoAnywhere", "CREStereo", "Fast-FoundationStereo")
PAPER_PROTOCOL = "paper_d2_strict_all_anchors"
H4_ONLY_PROTOCOL = "h4_only_common_support"
STRICT_SUPPORT = "GT coverage > 0.5 & raw valid & H4 support & all CS1/2/4/8 aligned-valid & warp support"
H4_ONLY_SUPPORT = "GT coverage > 0.5 & raw valid & recurrent aligned-valid & warp support"


def validate_cuda_assignment(device: str) -> str | None:
    """SEA-RAFT may use logical CUDA 0 internally; require one remapped GPU."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not device.startswith("cuda"):
        return visible
    if device != "cuda:0" or visible is None or not visible.isdecimal():
        raise RuntimeError(
            "CUDA evaluation requires exactly one physical GPU via CUDA_VISIBLE_DEVICES "
            "and logical --device cuda:0; use CUDA_VISIBLE_DEVICES=1 --device cuda:0"
        )
    return visible


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        Path(temporary).replace(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("refusing to publish an empty transfer CSV")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with tempfile.NamedTemporaryFile("w", newline="", delete=False, dir=path.parent, prefix=f".{path.name}.") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
        temporary = Path(handle.name)
    try:
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def check_frozen_inputs(checkpoint: Path, policy: Path) -> dict:
    return verify_canonical_inputs(checkpoint, policy)


def _summary(rows: list[dict]) -> tuple[list[dict], dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["backbone"], row["method"])].append(row)
    per_backbone = []
    for (backbone, method), values in sorted(groups.items()):
        count = sum(int(row["valid_pixel_count"]) for row in values)
        error = sum(float(row["error_sum"]) for row in values)
        per_backbone.append({"backbone": backbone, "method": method, "frames": len(values), "valid_pixel_count": count,
                             "epe": error / count if count else None})
    by_method: dict[str, list[float]] = defaultdict(list)
    for row in per_backbone:
        if row["epe"] is not None:
            by_method[row["method"]].append(row["epe"])
    return per_backbone, {method: sum(values) / len(values) for method, values in sorted(by_method.items())}


def _h4_only_rows(args: argparse.Namespace, policy: dict) -> tuple[list[dict], dict]:
    import sys
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
    from run_codd_style_bounded_memory_validation import evaluate as h4_evaluate

    split = "validation" if args.split == "d2" else "test"
    with tempfile.TemporaryDirectory(prefix=".frozen_transfer_", dir=args.output) as scratch:
        config = SimpleNamespace(
            mode="evaluate", split=split, output=Path(scratch), checkpoint=args.checkpoint,
            device=args.device, workers=args.workers, preload_workers=args.preload_workers,
            seed=args.seed, policy_name="fixed_h4", max_age=None,
            accumulated_update_max=None, disagreement_max=None, warp_support_min=None,
            fb_confidence_min=None, temporal_activation_max=None, update_magnitude_max=None,
            hard_threshold=None, memory_state="recurrent", disable_learned_stereo_evidence=False,
            backbones=tuple(args.backbones), sequences=tuple(args.sequences) if args.sequences else None,
            frozen_policy=args.policy, selection_root=Path(scratch), tiny=args.smoke,
        )
        h4_evaluate(config)
        with (Path(scratch) / "frame_metrics.csv").open(newline="") as handle:
            frames = list(csv.DictReader(handle))
    rows = []
    for frame in frames:
        common = {"dataset": "SCARED-C", "protocol": H4_ONLY_PROTOCOL, "split": args.split,
                  "backbone": frame["backbone"], "sequence": frame["sequence"], "frame_id": frame["frame_id"],
                  "valid_pixel_count": int(frame["valid_count"]), "reset": int(frame["reset"]),
                  "step_since_reset": int(frame["step_since_reset"]), "checkpoint_sha256": CHECKPOINT_SHA256}
        count = common["valid_pixel_count"]
        rows.extend((common | {"method": "raw", "epe": float(frame["raw_epe"]), "error_sum": float(frame["raw_epe"]) * count},
                     common | {"method": "canonical_bounded_h4", "epe": float(frame["fused_epe"]), "error_sum": float(frame["fused_epe"]) * count}))
    return rows, {"policy": policy["policy"], "support": H4_ONLY_SUPPORT, "warmup_index": 1}


def _paper_d2_strict_rows(args: argparse.Namespace, policy: dict) -> tuple[list[dict], dict]:
    """Exact paper mask, using only frozen D2 data/SEA-RAFT alignment primitives."""
    import sys
    import cv2
    import numpy as np
    import torch

    sys.path.insert(0, str(FROZEN / "src"))
    sys.path.insert(0, str(FROZEN / "experiments/02_massive_training/scripts"))
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
    from argos_freezed.alignment.sea_raft_adapter import SEARAFTFlowAdapter
    from argos_freezed.alignment.bida_pull_warp import temporal_disparity_evidence
    from campaign_common import ANCHOR_AGES, VALIDATION_SEQUENCES
    from argos_v2.cache_io import load_sequence_cache
    from argos_v2.scared_c_data import load_frame_gt, load_sequence_info, read_rgb
    from model_design.models.codd_bounded_memory import BoundedMemoryPolicy, advance_state_age
    from model_design.models.codd_style_fusion import CODDStyleFusionHead
    from provenance.codd_style_fusion import FrozenResNet18Layer1, build_codd_cues

    if args.split != "d2":
        raise RuntimeError(f"{PAPER_PROTOCOL} is D2-only")
    sequences = tuple(args.sequences) if args.sequences else tuple(VALIDATION_SEQUENCES)
    if any(not sequence.startswith("dataset_2_") for sequence in sequences):
        raise RuntimeError("paper protocol accepts only SCARED-C dataset_2 sequences")
    backbones = tuple(args.backbones)
    if args.smoke:
        sequences, backbones = sequences[:1], backbones[:1]
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    h4_model = CODDStyleFusionHead(state["cue_channels"]).to(device)
    h4_model.load_state_dict(state["model"], strict=True); h4_model.eval().requires_grad_(False)
    extractor = FrozenResNet18Layer1().to(device).eval().requires_grad_(False)
    adapter = SEARAFTFlowAdapter(device=device)

    def gt_cache(info, frame_id):
        disparity, valid = load_frame_gt(info, frame_id)
        coverage = cv2.resize(valid.astype(np.float32), (180, 144), interpolation=cv2.INTER_AREA)
        numerator = cv2.resize(disparity * valid.astype(np.float32), (180, 144), interpolation=cv2.INTER_AREA)
        return (numerator / np.maximum(coverage, 1e-6) * (180 / disparity.shape[1])).astype(np.float32), coverage

    def rgb(path):
        image = cv2.resize(read_rgb(path), (180, 144), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().to(device)

    def flows(images, indices, ages):
        result = {}
        for age in ages:
            forward, backward = [], []
            for start in range(0, len(indices), args.flow_batch_size):
                current = torch.stack([images[index] for index in indices[start:start + args.flow_batch_size]])
                past = torch.stack([images[index - age] for index in indices[start:start + args.flow_batch_size]])
                inferred = adapter.infer(torch.cat((current, past)), torch.cat((past, current)))
                size = len(current); forward.extend(inferred[:size].cpu().numpy()); backward.extend(inferred[size:].cpu().numpy())
            result[age] = (np.asarray(forward, dtype=np.float32), np.asarray(backward, dtype=np.float32))
        return result

    @torch.inference_mode()
    def h4_outputs(raw, valid, images, right, one_age_flows):
        memory = None; age = 0; output, support, reset_flags = {}, {}, {}
        reset = BoundedMemoryPolicy(name="fixed_h4", max_age=4)
        for index in range(1, len(raw)):
            item = {"raw": torch.from_numpy(raw[index:index + 1, None]).float().to(device),
                    "raw_valid": torch.from_numpy(valid[index:index + 1, None]).bool().to(device),
                    "current_rgb": images[index][None], "past_rgb": images[index - 1][None], "current_right_rgb": right[index][None]}
            pre_reset = memory is None or reset.pre_reset(age=age, accumulated_update=0.0)
            if pre_reset:
                memory = {"disparity": torch.from_numpy(raw[index - 1:index, None]).float().to(device),
                          "valid": torch.from_numpy(valid[index - 1:index, None]).bool().to(device)}
                age = 0
            forward = torch.from_numpy(one_age_flows[0][index - 1:index]).float().to(device)
            backward = torch.from_numpy(one_age_flows[1][index - 1:index]).float().to(device)
            evidence = temporal_disparity_evidence(item["raw"], memory["disparity"], forward, backward,
                current_valid=item["raw_valid"], past_valid=memory["valid"], current_rgb=item["current_rgb"], past_rgb=item["past_rgb"])
            cues = build_codd_cues(extractor, raw=item["raw"], aligned_memory=evidence.aligned_past_disparity,
                current_rgb=item["current_rgb"], current_right_rgb=item["current_right_rgb"], past_rgb=item["past_rgb"],
                flow_current_to_past=forward, flow_magnitude=evidence.flow_magnitude,
                forward_backward_confidence=evidence.forward_backward_confidence, warp_support=evidence.warp_support,
                aligned_valid=evidence.aligned_validity, include_learned_stereo_evidence=True)
            result = h4_model(cues, item["raw"], evidence.aligned_past_disparity)
            output[index] = result.fused_disparity[0, 0].cpu().numpy()
            support[index] = (evidence.aligned_validity & evidence.warp_support)[0, 0].cpu().numpy()
            reset_flags[index] = int(pre_reset)
            memory = {"disparity": result.fused_disparity, "valid": item["raw_valid"]}
            age = advance_state_age(age, reset=pre_reset)
        return output, support, reset_flags

    rows = []
    for sequence in sequences:
        info = load_sequence_info(sequence); frame_ids = info.frame_ids
        if args.max_frames:
            frame_ids = frame_ids[:args.max_frames]
        if len(frame_ids) <= max(ANCHOR_AGES):
            raise RuntimeError(f"{sequence} has no strict all-anchor frame")
        images = [rgb(info.seq_dir / "left" / f"{frame_id}.png") for frame_id in frame_ids]
        right = [rgb(info.seq_dir / "right" / f"{frame_id}.png") for frame_id in frame_ids]
        gt, coverage = zip(*(gt_cache(info, frame_id) for frame_id in frame_ids))
        gt, coverage = np.stack(gt), np.stack(coverage)
        indices = list(range(max(ANCHOR_AGES), len(frame_ids)))
        one_age = flows(images, list(range(1, len(frame_ids))), (1,))[1]
        anchor_flows = flows(images, indices, ANCHOR_AGES)
        for backbone in backbones:
            disparity, validity, cache_ids, _ = load_sequence_cache(backbone, sequence)
            if [str(value) for value in cache_ids[:len(frame_ids)]] != frame_ids:
                raise RuntimeError(f"frame-ID mismatch: {backbone}/{sequence}")
            raw = np.asarray(disparity[:len(frame_ids)], dtype=np.float32)
            valid = np.asarray(validity[:len(frame_ids)]).astype(bool)
            h4_map, h4_support, resets = h4_outputs(raw, valid, images, right, one_age)
            for offset, index in enumerate(indices):
                candidates = []
                for age in ANCHOR_AGES:
                    forward = torch.from_numpy(anchor_flows[age][0][offset:offset + 1]).float().to(device)
                    backward = torch.from_numpy(anchor_flows[age][1][offset:offset + 1]).float().to(device)
                    evidence = temporal_disparity_evidence(torch.from_numpy(raw[index:index + 1, None]).float().to(device),
                        torch.from_numpy(raw[index - age:index - age + 1, None]).float().to(device), forward, backward,
                        current_valid=torch.from_numpy(valid[index:index + 1, None]).bool().to(device),
                        past_valid=torch.from_numpy(valid[index - age:index - age + 1, None]).bool().to(device),
                        current_rgb=images[index][None], past_rgb=images[index - age][None])
                    candidates.append((evidence.aligned_validity & evidence.warp_support)[0, 0].cpu().numpy())
                strict = (coverage[index] > .5) & valid[index] & h4_support[index] & np.logical_and.reduce(candidates)
                if not strict.any():
                    continue
                count = int(strict.sum())
                common = {"dataset": "SCARED-C", "protocol": PAPER_PROTOCOL, "split": "d2", "backbone": backbone,
                          "sequence": sequence, "frame_id": frame_ids[index], "frame_index": index, "reset": resets[index],
                          "step_since_reset": (index - 1) % 4 + 1, "valid_pixel_count": count, "checkpoint_sha256": CHECKPOINT_SHA256}
                for method, prediction in (("raw", raw[index]), ("canonical_bounded_h4", h4_map[index])):
                    error = float(np.abs(prediction[strict] - gt[index][strict]).sum())
                    rows.append(common | {"method": method, "error_sum": error, "epe": error / count})
    if not rows:
        raise RuntimeError("empty paper D2 strict-all-anchor evaluation")
    return rows, {"policy": policy["policy"], "support": STRICT_SUPPORT, "warmup_index": 8,
                  "aggregation": "per-backbone valid-pixel weighted; paper aggregate is equal-backbone mean"}


def evaluate(args: argparse.Namespace) -> None:
    visible_devices = validate_cuda_assignment(args.device)
    policy = check_frozen_inputs(args.checkpoint, args.policy)
    atomic_text(args.output / "run_manifest.json", json.dumps({
        "project": "ARGOS v2", "status": "INCOMPLETE", "protocol": args.protocol,
        "device_argument": args.device, "CUDA_VISIBLE_DEVICES": visible_devices,
        "physical_gpu_assignment": "SEA-RAFT physical GPU assignment is controlled by CUDA_VISIBLE_DEVICES",
        "reason": "evaluation in progress; metrics are not citable until COMPLETE",
    }, indent=2, sort_keys=True) + "\n")
    if args.protocol == PAPER_PROTOCOL:
        rows, protocol = _paper_d2_strict_rows(args, policy)
    else:
        rows, protocol = _h4_only_rows(args, policy)
    per_backbone, aggregate = _summary(rows)
    atomic_csv(args.output / "metrics.csv", rows)
    atomic_csv(args.output / "per_backbone_metrics.csv", per_backbone)
    atomic_text(args.output / "aggregate_summary.json", json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    atomic_text(args.output / "run_manifest.json", json.dumps({
        "project": "ARGOS v2", "status": "COMPLETE", "scope": "SCARED-C only", "protocol": args.protocol,
        "split": args.split, "backbones": list(args.backbones), "methods": ["raw", "canonical_bounded_h4"],
        "device_argument": args.device, "CUDA_VISIBLE_DEVICES": visible_devices,
        "physical_gpu_assignment": "SEA-RAFT physical GPU assignment is controlled by CUDA_VISIBLE_DEVICES",
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": CHECKPOINT_SHA256,
        "policy": protocol["policy"], "policy_sha256": sha256(args.policy),
        "hard_threshold": None, "recurrence": "raw t-1 after reset; preceding fused state otherwise; reset at H=4; no future access",
        "support": protocol["support"], "warmup_index": protocol["warmup_index"], "aggregation": protocol.get("aggregation", "valid-pixel weighted"),
        "inference_manifest": str(INFERENCE_MANIFEST),
        "paper_cue_construction": "paper protocol uses frozen provenance FrozenResNet18Layer1/build_codd_cues with the local canonical H4 head; this reproduces the validated D2 cue path without geometry-v1 inference",
        "d4d_geometry": "NOT APPLICABLE: cache/geometry contract not validated by this evaluator",
        "servct_temporal": "NOT APPLICABLE: static pairs have no proven temporal adjacency",
        "legacy_note": "/dtu/p1/leopam/ARGOS/results/03_temporal_refinement/frozen_transfer_eval is preserved but noncanonical and not comparable",
        "published_files": {
            "metrics.csv": sha256(args.output / "metrics.csv"),
            "per_backbone_metrics.csv": sha256(args.output / "per_backbone_metrics.csv"),
            "aggregate_summary.json": sha256(args.output / "aggregate_summary.json"),
        },
        "runtime_sources": {
            "frozen_transfer_eval.py": sha256(ROOT / "scripts/frozen_transfer_eval.py"),
            "run_codd_style_bounded_memory_validation.py": sha256(ROOT / "scripts/run_codd_style_bounded_memory_validation.py"),
            "canonical_h4_provenance.py": sha256(ROOT / "scripts/canonical_h4_provenance.py"),
        },
    }, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=(PAPER_PROTOCOL, H4_ONLY_PROTOCOL), default=PAPER_PROTOCOL)
    parser.add_argument("--split", choices=("d2", "d7"), default="d2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--policy", type=Path, default=POLICY)
    parser.add_argument("--backbones", nargs="+", default=list(BACKBONES))
    parser.add_argument("--sequences", nargs="+")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--preload-workers", type=int, default=20)
    parser.add_argument("--flow-batch-size", type=int, default=32)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--static-check", action="store_true")
    args = parser.parse_args()
    if args.protocol == PAPER_PROTOCOL and args.split != "d2":
        parser.error(f"{PAPER_PROTOCOL} is D2-only; use {H4_ONLY_PROTOCOL} for D7")
    if args.static_check:
        print(json.dumps({"status": "PASS", "checkpoint_sha256": sha256(args.checkpoint), "policy": check_frozen_inputs(args.checkpoint, args.policy)["policy"], "protocol": args.protocol}))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    evaluate(args)


if __name__ == "__main__":
    main()
