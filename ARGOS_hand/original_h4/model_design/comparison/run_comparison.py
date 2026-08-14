#!/usr/bin/env python3
"""Dataset/protocol/metric base shared by frozen temporal-module comparisons."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT.parent / "results/temporal_module_comparison"
DEFAULT_MODULE = "model_design.comparison.canonical_h4:factory"
ALL_BACKBONES = ("S2M2-S", "RAFT-Stereo", "StereoAnywhere", "CREStereo", "Fast-FoundationStereo")
D4D_BACKBONES = ("RAFT-Stereo", "StereoAnywhere")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, prefix=f".{path.name}.") as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
        temporary.replace(path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty metric file: {path.name}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with tempfile.NamedTemporaryFile("w", newline="", delete=False, dir=path.parent, prefix=f".{path.name}.") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        temporary = Path(stream.name)
    temporary.replace(path)


def load_factory(spec: str):
    module, mark, name = spec.partition(":")
    if not mark or not module or not name:
        raise ValueError("--module must be import.path:factory")
    factory = getattr(importlib.import_module(module), name)
    if not callable(factory):
        raise TypeError(f"factory is not callable: {spec}")
    return factory


def check_adapter(adapter: Any) -> None:
    for name in ("start", "step", "describe"):
        if not callable(getattr(adapter, name, None)):
            raise TypeError(f"temporal adapter must provide {name}()")


def validate_cuda(device: str) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if device != "cuda:0" or visible is None or not visible.isdecimal():
        raise RuntimeError("evaluation requires one numeric CUDA_VISIBLE_DEVICES and logical --device cuda:0")
    return visible


def prepare_output(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    output.mkdir(parents=True)


def default_output(results: Path, config: argparse.Namespace) -> Path:
    name = config.module.replace(":", "__").replace(".", "_")
    return results / config.dataset / f"{name}{'__smoke' if config.smoke else ''}"


def _horizon(adapter: Any, horizon: int | None | object = ... ) -> int | None:
    """Resolve an explicitly declared finite horizon; the canonical default is H=4."""
    value = getattr(adapter, "horizon", 4) if horizon is ... else horizon
    if value is not None and (not isinstance(value, int) or value < 1):
        raise ValueError("horizon must be a positive integer or None for continuous recurrence")
    return value


def _finite(value: Any) -> bool:
    """Small adapter-boundary guard; booleans and missing test flows are not numeric evidence."""
    if value is None or isinstance(value, (bool, str)):
        return True
    if hasattr(value, "isfinite"):
        return bool(value.isfinite().all())
    try:
        import math
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        import numpy as np
        return bool(np.isfinite(np.asarray(value)).all())


def drive(adapter: Any, frames: list[Mapping[str, Any]], flow: Callable[[Mapping[str, Any], Mapping[str, Any]], tuple[Any, Any]], *,
          horizon: int | None | object = ...):
    """Causal bounded recurrence; adapter frames deliberately omit GT and metadata."""
    if not frames:
        return []
    horizon = _horizon(adapter, horizon)
    first = {key: frames[0][key] for key in ("raw", "raw_valid", "rgb", "right_rgb", "index") if key in frames[0]}
    if not _finite(first["raw"]):
        raise ValueError("non-finite first-frame raw disparity")
    outputs = [(0, adapter.start(first))]
    state = None; age = 0
    for index in range(1, len(frames)):
        current, previous = frames[index], frames[index - 1]
        reanchor = state is None or (horizon is not None and age >= horizon)
        memory = ({"disparity": previous["raw"], "valid": previous["raw_valid"], "rgb": previous["rgb"], "index": previous.get("index")}
                  if reanchor else state)
        forward, backward = flow(current, {"rgb": memory["rgb"], "index": memory["index"]})
        if not all(_finite(value) for value in (current["raw"], memory["disparity"], forward, backward)):
            raise ValueError("non-finite causal adapter input")
        item = {"raw": current["raw"], "raw_valid": current["raw_valid"], "current_rgb": current["rgb"],
                "current_right_rgb": current["right_rgb"], "past_rgb": memory["rgb"],
                "past_disparity": memory["disparity"], "past_valid": memory["valid"],
                "forward_flow": forward, "backward_flow": backward, "reanchor": reanchor,
                "state_age": 1 if reanchor else age + 1, "horizon": horizon}
        result = adapter.step(item)
        for key in ("disparity", "support", "reset", "state_age", "diagnostics"):
            if key not in result:
                raise RuntimeError(f"adapter.step result missing {key}")
        if bool(result["reset"]) != reanchor or int(result["state_age"]) != item["state_age"]:
            raise RuntimeError("adapter changed the declared recurrence protocol state")
        if not _finite(result["disparity"]):
            raise ValueError("non-finite temporal adapter output")
        outputs.append((index, result))
        state = {"disparity": result["disparity"], "valid": current["raw_valid"], "rgb": current["rgb"], "index": current.get("index")}
        age = int(result["state_age"])
    return outputs


def filter_d4d_windows(windows: list[dict[str, str]], contexts: Mapping[str, Mapping[str, str]], paths: Mapping[str, Mapping[str, str]]):
    """Keep only complete four-frame source windows; cache presence is insufficient."""
    accepted, unavailable = [], defaultdict(int)
    for window in windows:
        context = contexts.get(window["sequence_id"])
        stems = context["context_stems"].split(";") if context else []
        ids = [f"{window['specimen']}__{window['session']}__{stem}" for stem in stems]
        complete = len(ids) == 4 and all(
            frame_id in paths and Path(paths[frame_id]["left_path"]).exists() and Path(paths[frame_id]["right_path"]).exists()
            for frame_id in ids
        )
        if complete:
            accepted.append(window)
        else:
            unavailable[window["specimen"]] += 1
    return accepted, dict(sorted(unavailable.items()))


def official_scared_protocol_mask(raw_valid: Any, adjacent_support: Any, anchor_supports: Iterable[Any]=()) -> Any:
    """Prediction-independent SCARED support owned by the evaluation base."""
    import numpy as np
    mask = np.asarray(raw_valid, dtype=bool) & np.asarray(adjacent_support, dtype=bool)
    for support in anchor_supports:
        mask &= np.asarray(support, dtype=bool)
    return mask


def _scared(config: argparse.Namespace, adapter: Any, bundle_sink: Callable[[Mapping[str, Any]], None] | None=None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import cv2
    import numpy as np
    import torch
    sys.path[:0] = [str(ROOT / "scripts"), str(ROOT.parents[1] / "ARGOS_FREEZED/src"),
                    str(ROOT.parents[1] / "ARGOS_FREEZED/experiments/02_massive_training/scripts")]
    from argos_freezed.alignment.bida_pull_warp import temporal_disparity_evidence
    from argos_freezed.alignment.sea_raft_adapter import SEARAFTFlowAdapter
    from argos_v2.cache_io import load_sequence_cache
    from argos_v2.scared_c_data import load_frame_gt, load_sequence_info, read_rgb
    from campaign_common import ANCHOR_AGES, VALIDATION_SEQUENCES

    d2 = config.dataset == "scared-d2"
    sequences = tuple(config.sequences or (VALIDATION_SEQUENCES if d2 else ("dataset_7_keyframe_1", "dataset_7_keyframe_2", "dataset_7_keyframe_3", "dataset_7_keyframe_4")))
    if d2 and any(not value.startswith("dataset_2_") for value in sequences):
        raise ValueError("SCARED-D2 paper protocol accepts only dataset_2 sequences")
    if not d2 and any(not value.startswith("dataset_7_") for value in sequences):
        raise ValueError("SCARED-D7 protocol accepts only dataset_7 sequences")
    backbones = tuple(config.backbones)
    if config.smoke: sequences, backbones = sequences[:1], backbones[:1]
    device = torch.device(config.device); flow_model = SEARAFTFlowAdapter(device=device)
    rows: list[dict[str, Any]] = []

    def rgb(path: Path):
        value = cv2.resize(read_rgb(path), (180, 144), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(np.ascontiguousarray(value)).permute(2, 0, 1).float().to(device)[None]

    native_width: int | None = None

    def gt(info, frame_id):
        nonlocal native_width
        value, valid = load_frame_gt(info, frame_id)
        native_width = value.shape[1]
        coverage = cv2.resize(valid.astype(np.float32), (180, 144), interpolation=cv2.INTER_AREA)
        numerator = cv2.resize(value * valid.astype(np.float32), (180, 144), interpolation=cv2.INTER_AREA)
        return numerator / np.maximum(coverage, 1e-6) * (180 / value.shape[1]), coverage

    for sequence in sequences:
        info = load_sequence_info(sequence); ids = info.frame_ids[:config.max_frames] if config.max_frames else info.frame_ids
        if len(ids) < 2: raise RuntimeError(f"short sequence: {sequence}")
        images, right = [rgb(info.seq_dir / "left" / f"{value}.png") for value in ids], [rgb(info.seq_dir / "right" / f"{value}.png") for value in ids]
        gts, covers = zip(*(gt(info, value) for value in ids))
        pair_flows: dict[tuple[int, int], tuple[Any, Any]] = {}
        anchor_flows: dict[tuple[int, int], tuple[Any, Any]] = {}
        def precompute(indices, ages, destination):
            for age in ages:
                for start in range(0, len(indices), config.flow_batch_size):
                    batch = indices[start:start + config.flow_batch_size]
                    current = torch.cat([images[index] for index in batch])
                    past = torch.cat([images[index - age] for index in batch])
                    inferred = flow_model.infer(torch.cat((current, past)), torch.cat((past, current))).cpu().numpy()
                    size = len(batch)
                    for offset, index in enumerate(batch):
                        destination[(index, index - age)] = (inferred[offset:offset + 1], inferred[size + offset:size + offset + 1])
        precompute(list(range(1, len(ids))), (1,), pair_flows)
        if d2:
            precompute(list(range(max(ANCHOR_AGES), len(ids))), ANCHOR_AGES, anchor_flows)
        def flow(current, past):
            key = (current["index"], past["index"])
            forward, backward = pair_flows[key]
            return torch.from_numpy(forward).float().to(device), torch.from_numpy(backward).float().to(device)
        for backbone in backbones:
            disparity, validity, cache_ids, _ = load_sequence_cache(backbone, sequence)
            if [str(value) for value in cache_ids[:len(ids)]] != ids: raise RuntimeError(f"frame-ID mismatch: {backbone}/{sequence}")
            frames = [{"index": index, "raw": torch.from_numpy(np.asarray(disparity[index:index + 1], np.float32))[:, None].to(device),
                       "raw_valid": torch.from_numpy(np.asarray(validity[index:index + 1]) > 0)[:, None].to(device),
                       "rgb": images[index], "right_rgb": right[index]} for index in range(len(ids))]
            outputs = dict(drive(adapter, frames, flow))
            begin = max(ANCHOR_AGES) if d2 else 1
            bundle = []
            for index in range(begin, len(ids)):
                result = outputs[index]; raw_valid = np.asarray(validity[index]).astype(bool)
                if bundle_sink is not None:
                    forward_np, backward_np = pair_flows[(index, index - 1)]
                    evidence = temporal_disparity_evidence(
                        frames[index]["raw"], frames[index - 1]["raw"],
                        torch.from_numpy(forward_np).float().to(device), torch.from_numpy(backward_np).float().to(device),
                        current_valid=frames[index]["raw_valid"], past_valid=frames[index - 1]["raw_valid"],
                        current_rgb=images[index], past_rgb=images[index - 1])
                    anchors = []
                    if d2:
                        for age in ANCHOR_AGES:
                            forward_np, backward_np = anchor_flows[(index, index - age)]
                            anchor = temporal_disparity_evidence(
                                frames[index]["raw"], frames[index - age]["raw"],
                                torch.from_numpy(forward_np).float().to(device), torch.from_numpy(backward_np).float().to(device),
                                current_valid=frames[index]["raw_valid"], past_valid=frames[index - age]["raw_valid"],
                                current_rgb=images[index], past_rgb=images[index - age])
                            anchors.append((anchor.aligned_validity & anchor.warp_support)[0, 0].cpu().numpy())
                    bundle.append({"raw": frames[index]["raw"][0, 0].cpu().numpy(),
                                   "refined": result["disparity"][0, 0].detach().cpu().numpy(),
                                   "aligned_memory": result.get("aligned_memory", frames[index - 1]["raw"])[0, 0].detach().cpu().numpy(),
                                   "gt": gts[index], "gt_valid": covers[index] > .5,
                                   "protocol_mask": official_scared_protocol_mask(
                                       raw_valid, (evidence.aligned_validity & evidence.warp_support)[0, 0].cpu().numpy(), anchors),
                                   "adapter_support": result["support"][0, 0].detach().cpu().numpy().astype(bool),
                                   "frame_id": ids[index], "reset": bool(result["reset"])})
                mask = (covers[index] > .5) & raw_valid & result["support"][0, 0].detach().cpu().numpy().astype(bool)
                if d2:
                    anchor_masks = []
                    for age in ANCHOR_AGES:
                        forward_np, backward_np = anchor_flows[(index, index - age)]
                        forward, backward = torch.from_numpy(forward_np).float().to(device), torch.from_numpy(backward_np).float().to(device)
                        evidence = temporal_disparity_evidence(frames[index]["raw"], frames[index - age]["raw"], forward, backward,
                            current_valid=frames[index]["raw_valid"], past_valid=frames[index - age]["raw_valid"], current_rgb=images[index], past_rgb=images[index - age])
                        anchor_masks.append((evidence.aligned_validity & evidence.warp_support)[0, 0].cpu().numpy())
                    mask &= np.logical_and.reduce(anchor_masks)
                if not mask.any(): continue
                count = int(mask.sum()); raw_np = frames[index]["raw"][0, 0].cpu().numpy(); fused = result["disparity"][0, 0].detach().cpu().numpy()
                common = {"dataset": "SCARED-C", "split": "d2" if d2 else "d7", "protocol": "paper_d2_strict_all_anchors" if d2 else "h4_only_common_support",
                          "backbone": backbone, "sequence": sequence, "frame_id": ids[index], "frame_index": index,
                          "reset": int(result["reset"]), "step_since_reset": int(result["state_age"]), "valid_pixel_count": count}
                for method, prediction in (("raw", raw_np), ("temporal_module", fused)):
                    error = float(np.abs(prediction[mask] - gts[index][mask]).sum())
                    rows.append(common | {"method": method, "epe": error / count, "error_sum": error,
                                           "update_magnitude": result["diagnostics"]["update_magnitude"]})
            if bundle_sink is not None and bundle:
                calibration = None
                try:
                    fx, baseline = float(info.fx) * 180 / native_width, float(info.baseline_mm)
                    if np.isfinite(fx) and fx > 0 and np.isfinite(baseline) and baseline > 0:
                        calibration = {"fx_px": fx, "baseline_mm": baseline, "width_scale": 180 / native_width}
                except (AttributeError, TypeError, ValueError, ZeroDivisionError):
                    pass
                bundle_sink({"dataset": "SCARED-C", "split": "d2" if d2 else "d7", "protocol": "paper_d2_strict_all_anchors" if d2 else "h4_only_common_support",
                             "backbone": backbone, "sequence_id": sequence,
                             "raw_disparity": np.asarray([item["raw"] for item in bundle]),
                             "refined_disparity": np.asarray([item["refined"] for item in bundle]),
                             "aligned_memory": np.asarray([item["aligned_memory"] for item in bundle]),
                             "gt_disparity": np.asarray([item["gt"] for item in bundle]),
                             "gt_valid": np.asarray([item["gt_valid"] for item in bundle]),
                             "protocol_mask": np.asarray([item["protocol_mask"] for item in bundle]),
                             "adapter_support": np.asarray([item["adapter_support"] for item in bundle]),
                             "reset_mask": np.asarray([item["reset"] for item in bundle], dtype=bool),
                             "keyframe_mask": np.asarray([index == 0 or item["reset"] for index, item in enumerate(bundle)], dtype=bool),
                             "frame_ids": [item["frame_id"] for item in bundle], "calibration": calibration})
    if not rows and bundle_sink is None: raise RuntimeError("empty SCARED evaluation")
    return rows, {"protocol": "paper_d2_strict_all_anchors" if d2 else "h4_only_common_support", "sequences": list(sequences), "backbones": list(backbones)}


def _d4d(config: argparse.Namespace, adapter: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import numpy as np
    import torch
    v2 = ROOT.parents[1] / "ARGOS-V2/scripts"; sys.path.insert(0, str(v2))
    from build_multidomain_backbone_cache import d4d_records, read_pair
    from model_design.external_components.bidavideo import BiDAFlowInferenceAdapter, temporal_disparity_evidence
    cache_root, context = ROOT.parents[1] / "ARGOS-V2/cache_multidomain_backbones", ROOT.parents[1] / "results/03_temporal_refinement/ood/d4d_s2m2_zero_shot"
    contexts = {row["anchor_id"]: row for row in csv.DictReader((context / "context_manifest.csv").open())}
    windows = [row for row in csv.DictReader((context / "d4d_index.csv").open()) if row["sequence_id"] in contexts]
    if config.sequences: windows = [row for row in windows if row["sequence_id"] in set(config.sequences)]
    manifest_root = cache_root / config.backbones[0] / "D4D"
    cache_paths = {row["frame_id"]: row for row in csv.DictReader((manifest_root / "frame_manifest.csv").open())}
    windows, unavailable = filter_d4d_windows(windows, contexts, cache_paths)
    if not windows:
        raise RuntimeError("no compatible D4D four-frame windows after source-availability filtering")
    if config.smoke: windows = windows[:1]
    available_specimens = {row["specimen"] for row in windows}
    source = {row.frame_id: row for row in d4d_records(specimens=available_specimens)}; device = torch.device(config.device); flow_model = BiDAFlowInferenceAdapter("sea_raft", device=device)
    rows: list[dict[str, Any]] = []
    def image(value):
        import cv2
        value = cv2.resize(value, (180, 144), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(np.ascontiguousarray(value)).permute(2, 0, 1)[None].float().to(device)
    for backbone in config.backbones:
        base = cache_root / backbone / "D4D"; metadata = json.loads((base / "metadata.json").read_text())
        if not metadata.get("completion_status") or metadata.get("disparity_convention") != "positive_left_disparity": raise RuntimeError(f"incompatible D4D cache: {base}")
        disp, valid = np.load(base / "disparity.npy", mmap_mode="r"), np.load(base / "valid_mask.npy", mmap_mode="r")
        ids = [str(value) for value in np.load(base / "frame_ids.npy", allow_pickle=True).tolist()]; lookup = {value: index for index, value in enumerate(ids)}
        if len(lookup) != len(ids): raise RuntimeError(f"duplicate D4D frame IDs: {base}")
        rectification = {}
        for window_index, window in enumerate(windows):
            stems = contexts[window["sequence_id"]]["context_stems"].split(";")[::-1]
            if len(stems) != 4: raise RuntimeError(f"invalid D4D context: {window['sequence_id']}")
            if config.max_frames: stems = stems[:config.max_frames]
            frames = []
            for stem in stems:
                frame_id = f"{window['specimen']}__{window['session']}__{stem}"; index = lookup[frame_id]; left, right = read_pair(source[frame_id], rectification)
                raw = np.asarray(disp[index], np.float32); mask = np.asarray(valid[index], bool) & np.isfinite(raw) & (raw > 0)
                frames.append({"raw": torch.from_numpy(raw)[None, None].to(device), "raw_valid": torch.from_numpy(mask)[None, None].to(device), "rgb": image(left), "right_rgb": image(right), "frame_id": frame_id})
            def flow(current, past): return flow_model.current_to_past(current["rgb"], past["rgb"]), flow_model.past_to_current(past["rgb"], current["rgb"])
            for age, result in drive(adapter, frames, flow)[1:]:
                raw_memory, current = frames[age - 1], frames[age]
                forward, backward = flow(current, raw_memory)
                evidence = temporal_disparity_evidence(current["raw"], raw_memory["raw"], forward, backward, current_valid=current["raw_valid"], past_valid=raw_memory["raw_valid"], current_rgb=current["rgb"], past_rgb=raw_memory["rgb"])
                mask = result["support"] & evidence.aligned_validity & evidence.warp_support
                def value(tensor): return float(tensor[mask].mean()) if bool(mask.any()) else None
                raw_mc = value((current["raw"] - evidence.aligned_past_disparity).abs()); fused_mc = value((result["disparity"] - result.get("aligned_memory", evidence.aligned_past_disparity)).abs())
                rows.append({"dataset": "D4D", "metric_scope": "no_reference_prediction_space", "backbone": backbone, "specimen": window["specimen"], "session": window["session"], "sequence": window["sequence_id"], "window_index": window_index, "frame_id": current["frame_id"], "step_since_reset": age, "reset": int(result["reset"]), "support_coverage": float(mask.float().mean()), "raw_mc_inconsistency": raw_mc, "temporal_module_mc_inconsistency": fused_mc, "update_magnitude": result["diagnostics"]["update_magnitude"]})
    if not rows: raise RuntimeError("empty D4D evaluation")
    return rows, {"protocol": "four-frame curated causal windows, past-to-present, reset per context", "geometry_status": "NOT_APPLICABLE", "backbones": list(config.backbones), "available_specimens": sorted(available_specimens), "unavailable_windows_by_specimen": unavailable, "unavailable_reason": "cached frames whose four raw left/right source pairs are unavailable were excluded before smoke/limit"}


def _group(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    if rows and "method" in rows[0]:
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[(row[key], row["method"])].append(row)
        output = []
        for (value, method), values in sorted(groups.items()):
            count = sum(int(row["valid_pixel_count"]) for row in values)
            output.append({key: value, "method": method, "frames": len(values), "valid_pixel_count": count,
                           "epe": sum(float(row["error_sum"]) for row in values) / count if count else None})
        return output
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows: groups[row[key]].append(row)
    output = []
    for value, values in sorted(groups.items()):
        record = {key: value, "frames": len(values)}
        for name in ("epe", "raw_mc_inconsistency", "temporal_module_mc_inconsistency", "update_magnitude", "support_coverage"):
            present = [float(row[name]) for row in values if row.get(name) is not None]
            if present: record[name] = sum(present) / len(present)
        output.append(record)
    return output


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if rows and "error_sum" in rows[0]:
        values = {}
        for method in sorted({row["method"] for row in rows}):
            group = [row for row in rows if row["method"] == method]
            count = sum(int(row["valid_pixel_count"]) for row in group)
            values[method] = sum(float(row["error_sum"]) for row in group) / count if count else None
        return {"pixel_weighted_epe": values, "frames": len(rows) // 2}
    return {"frames": len(rows), "mean_update_magnitude": sum(float(row["update_magnitude"]) for row in rows) / len(rows),
            "mean_raw_mc_inconsistency": sum(float(row["raw_mc_inconsistency"]) for row in rows if row["raw_mc_inconsistency"] is not None) / max(1, sum(row["raw_mc_inconsistency"] is not None for row in rows)),
            "mean_temporal_module_mc_inconsistency": sum(float(row["temporal_module_mc_inconsistency"]) for row in rows if row["temporal_module_mc_inconsistency"] is not None) / max(1, sum(row["temporal_module_mc_inconsistency"] is not None for row in rows))}


def _servct(output: Path, backbones: tuple[str, ...]) -> None:
    rows = [{"dataset": "SERV-CT", "backbone": backbone, "temporal_h4_evaluation": "NOT_APPLICABLE", "reset_identity": "raw exactly", "reason": "static stereo pairs have no temporal adjacency"} for backbone in backbones]
    atomic_csv(output / "static_audit.csv", rows); atomic_json(output / "aggregate_summary.json", {"dataset": "SERV-CT", "temporal_h4_evaluation": "NOT_APPLICABLE"})


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("scared-d2", "scared-d7", "d4d", "servct"), required=True)
    parser.add_argument("--backbones", nargs="+"); parser.add_argument("--module", default=DEFAULT_MODULE)
    parser.add_argument("--output", type=Path); parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=20); parser.add_argument("--preload-workers", type=int, default=20)
    parser.add_argument("--flow-batch-size", type=int, default=32); parser.add_argument("--max-frames", type=int); parser.add_argument("--sequences", nargs="+")
    parser.add_argument("--smoke", action="store_true"); parser.add_argument("--static-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    config = arguments(); factory = load_factory(config.module)
    if config.static_check:
        adapter = factory(device="cpu"); check_adapter(adapter); print(json.dumps({"status": "PASS", "provenance": adapter.describe()}, sort_keys=True)); return
    config.backbones = tuple(config.backbones or (D4D_BACKBONES if config.dataset in {"d4d", "servct"} else ALL_BACKBONES))
    if config.dataset in {"d4d", "servct"} and set(config.backbones) - set(D4D_BACKBONES): raise ValueError("D4D/SERV-CT only have cached RAFT-Stereo and StereoAnywhere")
    gpu = None if config.dataset == "servct" else validate_cuda(config.device)
    adapter = factory(device=config.device); check_adapter(adapter); provenance = adapter.describe()
    output = config.output or default_output(RESULTS, config)
    prepare_output(output); manifest = {"project": "ARGOS v2", "status": "INCOMPLETE", "dataset": config.dataset, "module_provenance": provenance, "backbones": list(config.backbones), "CUDA_VISIBLE_DEVICES": gpu, "no_gt_in_adapter": True}
    atomic_json(output / "run_manifest.json", manifest)
    try:
        if config.dataset.startswith("scared-"):
            rows, summary = _scared(config, adapter); summary["aggregate"] = _aggregate(rows); atomic_csv(output / "frame_metrics.csv", rows); atomic_csv(output / "per_backbone_metrics.csv", _group(rows, "backbone")); atomic_csv(output / "per_sequence_metrics.csv", _group(rows, "sequence"))
        elif config.dataset == "d4d":
            rows, summary = _d4d(config, adapter); summary["aggregate"] = _aggregate(rows); atomic_csv(output / "frame_metrics.csv", rows)
            for key in ("backbone", "specimen", "session", "sequence"): atomic_csv(output / f"per_{key}_metrics.csv", _group(rows, key))
        else: _servct(output, config.backbones); summary = {"temporal_h4_evaluation": "NOT_APPLICABLE"}
        atomic_json(output / "aggregate_summary.json", summary); atomic_json(output / "run_manifest.json", manifest | {"status": "COMPLETE", "outputs": sorted(path.name for path in output.iterdir())})
    except BaseException as error:
        atomic_json(output / "run_manifest.json", manifest | {"error": f"{type(error).__name__}: {error}"}); raise


if __name__ == "__main__": main()
