#!/usr/bin/env python3
"""Audit whether curated D4D disparity agrees with its rectified stereo images.

This is a data-contract audit, not a model evaluation.  It neither trains nor
modifies a model, a cache, a calibration, or a ground-truth map.  At the
canonical 144x180 grid it compares deterministic current-frame left/right
reprojection costs of (a) curated Zivid disparity at preregistered fixed
scales and (b) frozen cache disparities.  A small SCARED-C control runs the
same cost implementation on its established rectified GT.

If a non-unit D4D scale consistently has substantially lower image matching
cost while SCARED-C selects unit scale, the correct conclusion is a geometry
contract mismatch to investigate upstream -- *not* a GT rescale or a model
correction.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

V2_ROOT = Path(__file__).resolve().parents[1]
ARGOS_ROOT = V2_ROOT.parent
sys.path[:0] = [str(V2_ROOT), str(V2_ROOT / "scripts"), str(ARGOS_ROOT)]

from model_design.data.temporal_pair_dataset import (  # noqa: E402
    TemporalPairDataset,
    resize_gt_to_cache_masked,
)
from model_design.external_components.stereo_photometric import (  # noqa: E402
    stereo_photometric_evidence,
)


D4D_ROOT = ARGOS_ROOT / "results/03_temporal_refinement/ood/d4d_s2m2_zero_shot"
OOD_CACHE = V2_ROOT / "cache_multidomain_backbones"
CACHE_HW = (144, 180)
SCALES = (0.125, 0.20, 0.25, 1.0 / 3.0, 0.50, 0.75, 1.0)


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-anchors", type=int, default=0, help="positive smoke limit; zero audits all anchors")
    parser.add_argument("--scared-control-pairs", type=int, default=5,
                        help="pairs per fixed SCARED-C control sequence")
    parser.add_argument("--local-kernel", type=int, default=21)
    parser.add_argument("--census-kernel", type=int, default=7)
    return parser.parse_args()


def clean(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        keys.extend(key for key in row if key not in keys)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def resize_disparity(value: np.ndarray) -> np.ndarray:
    """Resize a positive-left disparity to canonical cache grid and units."""
    value = np.asarray(value, dtype=np.float32)
    h, w = CACHE_HW
    return cv2.resize(value, (w, h), interpolation=cv2.INTER_LINEAR) * (w / value.shape[1])


def d4d_imports():
    source = ARGOS_ROOT / "scripts/temporal_refinement/ood/d4d"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from d4d_keyframe_gt import load_cam, rectify_maps, session_root
    return load_cam, rectify_maps, session_root


class D4DRightReader:
    """Read rectified source stereo images; no array is persisted."""

    def __init__(self) -> None:
        self.maps: dict[tuple[str, str], tuple[Path, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]] = {}

    def pair(self, row: dict, stem: str) -> tuple[np.ndarray, np.ndarray]:
        key = (row["specimen"], row["session"])
        if key not in self.maps:
            load_cam, rectify_maps, session_root = d4d_imports()
            root = session_root(row["specimen"]) / row["session"]
            left = load_cam(root / "camera_info/left.yaml")
            right = load_cam(root / "camera_info/right.yaml")
            self.maps[key] = (root, rectify_maps(left), rectify_maps(right))
        root, left_map, right_map = self.maps[key]
        left = cv2.imread(str(root / "left_images" / f"{stem}.png"), cv2.IMREAD_COLOR)
        right = cv2.imread(str(root / "right_images" / f"{stem}.png"), cv2.IMREAD_COLOR)
        if left is None or right is None:
            raise FileNotFoundError(f"missing D4D pair {key}/{stem}")
        left = cv2.remap(left, left_map[0], left_map[1], cv2.INTER_LINEAR)
        right = cv2.remap(right, right_map[0], right_map[1], cv2.INTER_LINEAR)
        return cv2.cvtColor(left, cv2.COLOR_BGR2RGB), cv2.cvtColor(right, cv2.COLOR_BGR2RGB)


class FrozenD4DCache:
    """Read only the separately stored frozen D4D backbone predictions."""

    def __init__(self, backbone: str) -> None:
        root = OOD_CACHE / backbone / "D4D"
        if not (root / ".complete").exists():
            raise FileNotFoundError(f"incomplete frozen D4D cache: {root}")
        self.disparity = np.load(root / "disparity.npy", mmap_mode="r")
        self.valid = np.load(root / "valid_mask.npy", mmap_mode="r")
        ids = np.load(root / "frame_ids.npy", allow_pickle=False).tolist()
        self.indices = {str(value): index for index, value in enumerate(ids)}

    def get(self, frame_id: str) -> tuple[np.ndarray, np.ndarray]:
        index = self.indices[frame_id]
        disparity = np.asarray(self.disparity[index], np.float32)
        valid = np.asarray(self.valid[index], bool) & np.isfinite(disparity) & (disparity > 0)
        return disparity, valid


def candidate_rows(
    *, dataset: str, item_id: str, left: np.ndarray, right: np.ndarray,
    candidates: dict[str, tuple[np.ndarray, np.ndarray]], support: np.ndarray,
    device: torch.device, local_kernel: int, census_kernel: int, comparison: str,
) -> list[dict]:
    """Evaluate every candidate on one common valid/census support."""
    names = list(candidates)
    values = torch.from_numpy(np.stack([candidates[name][0] for name in names]))[:, None].to(device)
    left_t = torch.from_numpy(np.repeat(left[None], len(names), axis=0)).permute(0, 3, 1, 2).to(device)
    right_t = torch.from_numpy(np.repeat(right[None], len(names), axis=0)).permute(0, 3, 1, 2).to(device)
    with torch.no_grad():
        evidence = stereo_photometric_evidence(
            left_t, right_t, values, local_kernel=local_kernel, census_kernel=census_kernel,
        )
    candidate_valid = np.stack([candidates[name][1] for name in names])
    shared = support.astype(bool) & candidate_valid.all(axis=0)
    shared &= evidence.right_support[:, 0].detach().cpu().numpy().all(axis=0)
    shared &= evidence.census_support[:, 0].detach().cpu().numpy().all(axis=0)
    local = evidence.local_rgb_l1[:, 0].detach().cpu().numpy()
    census = evidence.ternary_census_cost[:, 0].detach().cpu().numpy()
    result=[]
    for index, name in enumerate(names):
        result.append({
            "dataset": dataset, "item_id": item_id, "candidate": name, "comparison": comparison,
            "shared_valid_count": int(shared.sum()),
            "shared_valid_ratio": float(shared.mean()),
            "median_disparity_px": float(np.median(values[index, 0].detach().cpu().numpy()[shared])) if shared.any() else math.nan,
            "mean_local_rgb_l1": float(local[index][shared].mean()) if shared.any() else math.nan,
            "mean_ternary_census": float(census[index][shared].mean()) if shared.any() else math.nan,
        })
    return result


def run_d4d(device: torch.device, maximum: int, local_kernel: int, census_kernel: int) -> list[dict]:
    contexts = {row["anchor_id"]: row for row in csv.DictReader((D4D_ROOT / "context_manifest.csv").open())}
    rows = list(csv.DictReader((D4D_ROOT / "d4d_index.csv").open()))
    if maximum:
        rows = rows[:maximum]
    caches = {name: FrozenD4DCache(name) for name in ("RAFT-Stereo", "StereoAnywhere")}
    reader = D4DRightReader()
    output=[]
    for index, row in enumerate(rows, 1):
        context = contexts[row["sequence_id"]]["context_stems"].split(";")
        if len(context) != 4:
            raise RuntimeError(f"non-causal D4D context {row['sequence_id']}")
        # Context manifest is current-to-past.  Shards are past-to-current, so
        # raw_disp[3] and gt_disp[3] correspond to this first/current stem.
        stem = context[0]
        shard = np.load(row["target_path"])
        gt_native = np.asarray(shard["gt_disp"][3], np.float32)
        gt_valid_native = np.asarray(shard["valid_mask"][3], bool) & np.isfinite(gt_native) & (gt_native > 0)
        gt, coverage = resize_gt_to_cache_masked(gt_native, gt_valid_native)
        gt_valid = (coverage > .50) & np.isfinite(gt) & (gt > 0)
        # The coverage-normalized resize intentionally preserves invalid NaNs
        # for a supervised loader.  Here candidates go through grid_sample, so
        # replace only *masked-out* values to keep a NaN outside support from
        # contaminating a valid local photometric window.
        gt = np.where(gt_valid, gt, 0.0).astype(np.float32)
        left_native, right_native = reader.pair(row, stem)
        left = cv2.resize(left_native, (180, 144), interpolation=cv2.INTER_AREA)
        right = cv2.resize(right_native, (180, 144), interpolation=cv2.INTER_AREA)
        s2 = resize_disparity(np.asarray(shard["raw_disp"][3], np.float32))
        scales = {f"zivid_scale_{scale:g}": (gt * scale, gt_valid) for scale in SCALES}
        output.extend(candidate_rows(
            dataset="D4D", item_id=row["sequence_id"], left=left, right=right,
            candidates=scales, support=gt_valid, device=device,
            local_kernel=local_kernel, census_kernel=census_kernel, comparison="zivid_scales_common",
        ))
        candidates = {"raw_S2M2-S": (s2, np.isfinite(s2) & (s2 > 0))}
        key = f"{row['specimen']}__{row['session']}__{stem}"
        for name, cache in caches.items():
            candidates[f"raw_{name}"] = cache.get(key)
        # A prediction may legitimately have a different valid region.  Do
        # not turn that into a nearly empty three-backbone intersection: each
        # frozen raw candidate is compared on its own exact support together
        # with the two directly relevant fixed Zivid references.
        for raw_name, raw_value in candidates.items():
            pair = {
                "zivid_scale_0.25": scales["zivid_scale_0.25"],
                "zivid_scale_1": scales["zivid_scale_1"], raw_name: raw_value,
            }
            output.extend(candidate_rows(
                dataset="D4D", item_id=row["sequence_id"], left=left, right=right,
                candidates=pair, support=gt_valid, device=device,
                local_kernel=local_kernel, census_kernel=census_kernel, comparison=f"{raw_name}_paired",
            ))
        if index % 25 == 0:
            print(f"D4D anchors: {index}/{len(rows)}", flush=True)
    return output


def run_scared_control(device: torch.device, pairs: int, local_kernel: int, census_kernel: int) -> list[dict]:
    # One deterministic keyframe sequence per acquisition ID; never a model
    # selection split and never an argument for changing D4D labels.
    sequences = ("dataset_1_keyframe_2", "dataset_2_keyframe_2", "dataset_3_keyframe_1", "dataset_6_keyframe_1", "dataset_7_keyframe_1")
    ds = TemporalPairDataset(("S2M2-S",), sequences, coverage_threshold=.50,
                             max_pairs_per_sequence=pairs, include_right_rgb=True)
    ds.preload_frame_data(workers=min(16, len(sequences) * max(pairs, 1)))
    output=[]
    for index in range(len(ds)):
        item = ds[index]
        gt = item["gt"][0].numpy()
        valid = item["gt_valid"][0].numpy().astype(bool) & np.isfinite(gt) & (gt > 0)
        gt = np.where(valid, gt, 0.0).astype(np.float32)
        left = item["current_rgb"].permute(1, 2, 0).numpy().astype(np.uint8)
        right = item["current_right_rgb"].permute(1, 2, 0).numpy().astype(np.uint8)
        candidates = {f"gt_scale_{scale:g}": (gt * scale, valid) for scale in SCALES}
        output.extend(candidate_rows(
            dataset="SCARED-C-control", item_id=f"{item['sequence']}::{item['current_frame_id']}",
            left=left, right=right, candidates=candidates, support=valid, device=device,
            local_kernel=local_kernel, census_kernel=census_kernel, comparison="gt_scales_common",
        ))
    return output


def summarize(rows: list[dict]) -> tuple[list[dict], dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["comparison"], row["candidate"])].append(row)
    summary=[]
    for (dataset, comparison, candidate), values in sorted(grouped.items()):
        local=np.asarray([value["mean_local_rgb_l1"] for value in values], dtype=np.float64)
        census=np.asarray([value["mean_ternary_census"] for value in values], dtype=np.float64)
        disparity=np.asarray([value["median_disparity_px"] for value in values], dtype=np.float64)
        summary.append({
            "dataset": dataset, "comparison": comparison, "candidate": candidate, "items": len({value['item_id'] for value in values}),
            "finite_local_item_count": int(np.isfinite(local).sum()),
            "finite_census_item_count": int(np.isfinite(census).sum()),
            "mean_local_rgb_l1_per_item": float(np.nanmean(local)) if np.isfinite(local).any() else math.nan,
            "mean_ternary_census_per_item": float(np.nanmean(census)) if np.isfinite(census).any() else math.nan,
            "median_disparity_px_per_item": float(np.nanmedian(disparity)) if np.isfinite(disparity).any() else math.nan,
            "mean_shared_valid_count": float(np.mean([value["shared_valid_count"] for value in values])),
        })
    conclusion={}
    for dataset in sorted({row["dataset"] for row in rows}):
        expected = "zivid_scales_common" if dataset == "D4D" else "gt_scales_common"
        scales=[row for row in summary if row["dataset"] == dataset and row["comparison"] == expected and "scale_" in row["candidate"]]
        local=[row for row in scales if math.isfinite(row["mean_local_rgb_l1_per_item"])]
        census=[row for row in scales if math.isfinite(row["mean_ternary_census_per_item"])]
        conclusion[dataset]={
            "lowest_local_rgb_l1_candidate": min(local, key=lambda row: row["mean_local_rgb_l1_per_item"])["candidate"] if local else None,
            "lowest_ternary_census_candidate": min(census, key=lambda row: row["mean_ternary_census_per_item"])["candidate"] if census else None,
        }
    return summary, conclusion


def main() -> int:
    options=args()
    if options.max_anchors < 0 or options.scared_control_pairs < 1:
        raise ValueError("limits must be non-negative and control pairs positive")
    device=torch.device(options.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    options.output.mkdir(parents=True, exist_ok=True)
    begin=time.perf_counter()
    rows=run_d4d(device, options.max_anchors, options.local_kernel, options.census_kernel)
    rows.extend(run_scared_control(device, options.scared_control_pairs, options.local_kernel, options.census_kernel))
    summary, conclusion=summarize(rows)
    write_csv(options.output / "anchor_metrics.csv", rows)
    write_csv(options.output / "scale_summary.csv", summary)
    write_json(options.output / "aggregate_summary.json", {
        "purpose": "data-validity audit only; no GT or prediction scale was changed",
        "grid": "144x180 cache grid; positive-left disparity pixels",
        "candidate_scales": list(SCALES), "local_kernel": options.local_kernel,
        "census_kernel": options.census_kernel, "conclusion": conclusion,
        "anchors": len({row["item_id"] for row in rows if row["dataset"] == "D4D"}),
        "scared_control_frames": len({row["item_id"] for row in rows if row["dataset"] == "SCARED-C-control"}),
        "wall_seconds": time.perf_counter() - begin,
    })
    (options.output / "README.md").write_text(
        "# D4D stereo-geometry consistency audit\n\n"
        "This is a frozen data-contract check. `zivid_scale_*` rows are diagnostic fixed copies only; "
        "they must never be used to rescale GT, train a model, or report corrected D4D geometry. "
        "Candidate costs share exact GT/candidate/right/census support per anchor. The SCARED-C control "
        "checks the same code on established rectified stereo GT.\n"
    )
    print(json.dumps(clean({"output": options.output, "conclusion": conclusion})), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
