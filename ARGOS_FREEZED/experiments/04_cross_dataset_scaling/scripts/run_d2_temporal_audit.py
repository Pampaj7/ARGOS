#!/usr/bin/env python3
"""D2-only temporal sidecar: reuses validated inference primitives and never opens D7."""
from __future__ import annotations
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path

# Keep the invariant check usable on login nodes without importing PyTorch or
# model/provenance dependencies. Full evaluation follows below.
if "--static-check" in sys.argv:
    _static = argparse.ArgumentParser(add_help=False); _static.add_argument("--dataset-id", type=int, default=2); _static.add_argument("--static-check", action="store_true")
    _args, _ = _static.parse_known_args()
    if _args.dataset_id != 2: raise RuntimeError("D2-only audit refuses every dataset ID except 2")
    print(json.dumps({"status": "PASS", "dataset_7_opened": False, "min_immutable_temporal_index": 16}))
    raise SystemExit(0)

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
LEGACY = Path("/dtu/p1/leopam/ARGOS/ARGOS_FREEZED/experiments/02_massive_training/scripts")
sys.path[:0] = [str(EXP), str(LEGACY), "/dtu/p1/leopam/ARGOS/ARGOS_FREEZED/src"]
from metrics.temporal_metrics import gt_tce, nr_tce, stereo_photo, temporal_photo
from campaign_common import ANCHOR_AGES, ROOT, VALIDATION_SEQUENCES, guard_no_d7, sha256, strict_common_support, verify_frozen_core, write_csv, atomic_json
from data_pipeline import _align, infer_age_flows, load_frame_gt, load_sequence_cache, load_sequence_info, read_rgb, resize_gt_to_cache_masked, rgb_tensor
from evaluate_d2_common_support import h4_outputs, model
from argos_freezed.alignment.sea_raft_adapter import SEARAFTFlowAdapter
from argos_freezed.models.raw_multi_anchor_refiner import MultiAnchorEvidence, retrieve_and_fuse
from provenance.codd_style_fusion import CODDStyleFusionHead, FrozenResNet18Layer1

OUT = EXP / "scared_geometry_temporal"
CANONICAL = ROOT / "checkpoints/raw_multi_anchor_best_validation.pt"
H4_MANIFEST = ROOT / "baselines/bounded_h4/MANIFEST.json"
POLICY = LEGACY / "provenance/canonical_geometry_v1_policy.json"
BACKBONES = ("S2M2-S", "RAFT-Stereo", "StereoAnywhere", "CREStereo", "Fast-FoundationStereo")
MIN_IMMUTABLE_TEMPORAL_INDEX = 2 * max(ANCHOR_AGES)  # current and CS8-past outputs both have CS1/2/4/8 history
RESNET18 = EXP / "dependencies/resnet18-f37072fd.pth"
RESNET18_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"


def _policy() -> tuple[float, float]:
    value = json.loads(POLICY.read_text())["policy"]
    return float(value["probability_threshold"]), float(value["utility_threshold_px"])


def _rgb(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().permute(1, 2, 0).numpy()


def _aggregate(rows: list[dict], group: tuple[str, ...]) -> list[dict]:
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows: buckets[tuple(row[key] for key in group) + (row["method"],)].append(row)
    out = []
    for key, values in sorted(buckets.items()):
        row = dict(zip((*group, "method"), key)); row["frames"] = len(values)
        for name, count_name in (("epe", "epe_count"), ("gt_tce", "gt_tce_count"), ("nr_tce", "nr_tce_count"),
                                 ("stereo_photo", "stereo_photo_count"), ("temporal_photo", "temporal_photo_count")):
            valid = [item for item in values if np.isfinite(item[name]) and item[count_name] > 0]
            row[name] = float(np.average([item[name] for item in valid], weights=[item[count_name] for item in valid])) if valid else None
        out.append(row)
    return out


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> None:
    if args.dataset_id != 2:
        raise RuntimeError("D2-only audit refuses every dataset ID except 2")
    guard_no_d7(VALIDATION_SEQUENCES, args.dataset_id, args.output)
    if not str(Path(args.output).resolve()).startswith(str(EXP.resolve())):
        raise RuntimeError("output must remain under experiment 04")
    verify_frozen_core(); device = torch.device(args.device)
    resnet_checkpoint = Path(args.resnet_checkpoint)
    if sha256(resnet_checkpoint) != RESNET18_SHA256:
        raise RuntimeError(f"ResNet-18 checkpoint hash mismatch: {resnet_checkpoint}")
    immutable, _ = model(CANONICAL, device); prob, utility = _policy()
    h4_meta = json.loads(H4_MANIFEST.read_text()); h4_path = Path(h4_meta["checkpoint_path"])
    if sha256(h4_path) != h4_meta["checkpoint_sha256"]: raise RuntimeError("H4 checkpoint hash mismatch")
    h4_state = torch.load(h4_path, map_location="cpu", weights_only=False)
    h4 = CODDStyleFusionHead(h4_state["cue_channels"]).to(device); h4.load_state_dict(h4_state["model"], strict=True); h4.eval().requires_grad_(False)
    extractor = FrozenResNet18Layer1(checkpoint=resnet_checkpoint).to(device).eval(); adapter = SEARAFTFlowAdapter(device=device)
    rows, exclusions = [], []
    sequences = VALIDATION_SEQUENCES[:1] if args.smoke else VALIDATION_SEQUENCES
    backbones = BACKBONES[:1] if args.smoke else BACKBONES
    for sequence in sequences:
        info = load_sequence_info(sequence); ids = info.frame_ids[:args.max_frames] if args.max_frames else info.frame_ids
        print(f"D2 temporal sequence={sequence} frames={len(ids)}", flush=True)
        if len(ids) <= max(ANCHOR_AGES): raise RuntimeError(f"{sequence} has no CS8 frame")
        left, right, gt, coverage = [], [], [], []
        for frame_id in ids:
            left.append(rgb_tensor(read_rgb(info.seq_dir / "left" / f"{frame_id}.png"), device)); right.append(rgb_tensor(read_rgb(info.seq_dir / "right" / f"{frame_id}.png"), device))
            native, valid_gt = load_frame_gt(info, frame_id); resized, cov = resize_gt_to_cache_masked(native, valid_gt); gt.append(resized); coverage.append(cov)
        gt, coverage = np.stack(gt), np.stack(coverage); indices = list(range(8, len(ids)))
        h4_flows, _ = infer_age_flows(adapter, left, list(range(1, len(ids))), [1], args.flow_batch_size, device)
        age_flows, _ = infer_age_flows(adapter, left, indices, list(ANCHOR_AGES), args.flow_batch_size, device)
        for backbone in backbones:
            print(f"D2 temporal backbone={backbone} sequence={sequence}", flush=True)
            raw_m, valid_m, cache_ids, _ = load_sequence_cache(backbone, sequence)
            if [str(value) for value in cache_ids[:len(ids)]] != ids: raise RuntimeError(f"frame-ID mismatch: {backbone}/{sequence}")
            raw, valid = np.asarray(raw_m[:len(ids)], np.float32), np.asarray(valid_m[:len(ids)]).astype(bool)
            aligned = _align(raw, valid, left, indices, age_flows, device, args.flow_batch_size)
            h4_map, h4_support = h4_outputs(raw, valid, gt, coverage, left, right, h4_flows[1], h4, extractor, device)
            immutable_map: dict[int, np.ndarray] = {}
            for offset, index in enumerate(indices):
                evidence = MultiAnchorEvidence(torch.from_numpy(raw[index:index+1, None]).float().to(device), torch.from_numpy(aligned["candidates"][offset:offset+1]).float().to(device), torch.from_numpy(aligned["candidate_valid"][offset:offset+1]).bool().to(device), torch.from_numpy(aligned["warp_support"][offset:offset+1]).bool().to(device), torch.from_numpy(aligned["fb_confidence"][offset:offset+1]).float().to(device), torch.tensor(ANCHOR_AGES, device=device), torch.zeros(4, device=device))
                prediction, _, _, _ = retrieve_and_fuse(evidence.raw, evidence, immutable(evidence), probability_threshold=prob, utility_threshold_px=utility, hard=False)
                immutable_map[index] = prediction[0, 0].cpu().numpy()
            for offset, index in enumerate(indices):
                print(f"D2 temporal frame={index + 1}/{len(ids)} backbone={backbone} sequence={sequence}", flush=True)
                strict = strict_common_support(torch.from_numpy(coverage[index] > .5), torch.from_numpy(valid[index]), torch.from_numpy(h4_support[index]), torch.from_numpy((aligned["candidate_valid"][offset] & aligned["warp_support"][offset]).all(axis=0))).numpy()
                if not strict.any():
                    exclusions.append({"backbone": backbone, "sequence": sequence, "frame_id": ids[index], "reason": "empty_strict_common_support"}); continue
                methods = {"raw": raw[index], "h4": h4_map[index], "immutable": immutable_map[index]}
                # Geometry is available at the first CS8 frame. Temporal comparison is delayed
                # until the CS8-past immutable output itself has a complete anchor history.
                for name, prediction in methods.items():
                    geometry_valid = strict & np.isfinite(prediction)
                    epe = np.abs(prediction[geometry_valid] - gt[index][geometry_valid])
                    rows.append({"dataset_id": 2, "backbone": backbone, "sequence": sequence, "frame_id": ids[index], "frame_index": index, "age_frames": 0, "method": name, "valid_pixel_count": int(strict.sum()), "epe": float(epe.mean()), "epe_count": int(epe.size), "gt_tce": float("nan"), "gt_tce_count": 0, "rgt_tce": float("nan"), "nr_tce": float("nan"), "nr_tce_count": 0, "stereo_photo": float("nan"), "stereo_photo_count": 0, "temporal_photo": float("nan"), "temporal_photo_count": 0, "physical_span_ms": 0.0, "temporal_status": "not_eligible: immutable past output lacks full CS1/2/4/8 history" if index < MIN_IMMUTABLE_TEMPORAL_INDEX else "geometry_only"})
                if index < MIN_IMMUTABLE_TEMPORAL_INDEX:
                    continue
                for age_offset, age in enumerate(ANCHOR_AGES):
                    support = strict & aligned["candidate_valid"][offset, age_offset] & aligned["warp_support"][offset, age_offset] & (aligned["fb_confidence"][offset, age_offset] > 0)
                    flow = age_flows[age][0][offset]
                    for name, prediction in methods.items():
                        past = raw[index-age] if name == "raw" else (h4_map[index-age] if name == "h4" else immutable_map[index-age])
                        geometry_valid = strict & np.isfinite(prediction)
                        epe = np.abs(prediction[geometry_valid] - gt[index][geometry_valid])
                        g = gt_tce(prediction, past, gt[index], gt[index-age], flow, support=support, current_gt_valid=coverage[index] > .5, past_gt_valid=coverage[index-age] > .5, fb_support=support)
                        n = nr_tce(prediction, past, flow, support=support, current_valid=valid[index], past_valid=valid[index-age], fb_support=support)
                        sp = stereo_photo(_rgb(left[index]), _rgb(right[index]), prediction, support=geometry_valid)
                        tp = temporal_photo(_rgb(left[index]), _rgb(left[index-age]), flow, support=support)
                        rows.append({"dataset_id": 2, "backbone": backbone, "sequence": sequence, "frame_id": ids[index], "frame_index": index, "age_frames": age, "method": name, "valid_pixel_count": int(strict.sum()), "epe": float(epe.mean()), "epe_count": int(epe.size), "gt_tce": g["gt_tce"], "gt_tce_count": g["count"], "rgt_tce": g["rgt_tce"], "nr_tce": n["nr_tce"], "nr_tce_count": n["trimmed_count"], "stereo_photo": sp["stereo_photo"], "stereo_photo_count": sp["count"], "temporal_photo": tp["temporal_photo"], "temporal_photo_count": tp["count"], "physical_span_ms": 1000.0 * age / 25.0, "temporal_status": "eligible"})
    destination = Path(args.output); write_csv(destination / "frame_metrics.csv", rows); write_csv(destination / "sequence_metrics.csv", _aggregate(rows, ("backbone", "sequence", "age_frames"))); write_csv(destination / "backbone_metrics.csv", _aggregate(rows, ("backbone", "age_frames"))); write_csv(destination / "exclusions.csv", exclusions)
    atomic_json(destination / "aggregate_summary.json", {"project": "ARGOS v2", "status": "PASS", "dataset": "SCARED-C validation ID 2", "dataset_7_opened": False, "frame_rows": len(rows), "strict_support": "GT coverage & raw validity & H4 support & all CS1/2/4/8 support", "metric_definitions": str(EXP / "metrics/metric_definitions.json"), "canonical_checkpoint_sha256": sha256(CANONICAL)})


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset-id", type=int, default=2); parser.add_argument("--device", default="cuda:0"); parser.add_argument("--flow-batch-size", type=int, default=32); parser.add_argument("--max-frames", type=int); parser.add_argument("--smoke", action="store_true"); parser.add_argument("--static-check", action="store_true"); parser.add_argument("--resnet-checkpoint", default=str(RESNET18)); parser.add_argument("--output", default=str(OUT))
    args = parser.parse_args()
    if args.static_check:
        if args.dataset_id != 2: raise RuntimeError("D2-only audit refuses every dataset ID except 2")
        assert MIN_IMMUTABLE_TEMPORAL_INDEX == 16
        print(json.dumps({"status": "PASS", "dataset_7_opened": False, "min_immutable_temporal_index": MIN_IMMUTABLE_TEMPORAL_INDEX}))
        return
    evaluate(args)


if __name__ == "__main__": main()
