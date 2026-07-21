#!/usr/bin/env python3
"""Frozen SERV-CT GT/image stereo-consistency audit at the canonical grid.

This audit is intentionally narrow: it tests whether CT-derived positive-left
disparity agrees with the corresponding rectified left/right images under the
declared unit convention.  It trains nothing and writes no prediction or flow
cache.  Fixed GT scale candidates are diagnostic only and are never used to
rescale labels or predictions.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

V2_ROOT = Path(__file__).resolve().parents[1]
ARGOS_ROOT = V2_ROOT.parent
sys.path[:0] = [str(V2_ROOT), str(V2_ROOT / "scripts")]

from model_design.data.temporal_pair_dataset import resize_gt_to_cache_masked  # noqa: E402
from model_design.external_components.stereo_photometric import stereo_photometric_evidence  # noqa: E402


MANIFEST = ARGOS_ROOT / "results/03_temporal_refinement/ood/prepared/servct/sequence_manifest.csv"
SCALES = (.25, .50, .75, 1.0)


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-kernel", type=int, default=21)
    parser.add_argument("--census-kernel", type=int, default=7)
    return parser.parse_args()


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text(""); return
    fields=[]
    for row in rows: fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="") as handle:
        writer=csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def read_rgb(path: str) -> np.ndarray:
    value=cv2.imread(path, cv2.IMREAD_COLOR)
    if value is None: raise FileNotFoundError(path)
    return cv2.cvtColor(value, cv2.COLOR_BGR2RGB)


def main() -> int:
    args=parse(); device=torch.device(args.device); args.output.mkdir(parents=True, exist_ok=True)
    rows=[]
    for source in csv.DictReader(MANIFEST.open()):
        native=np.load(source["gt_disp_path"]).astype(np.float32)
        valid=np.load(source["valid_mask_path"]).astype(bool) & np.isfinite(native) & (native>0)
        gt,coverage=resize_gt_to_cache_masked(native,valid)
        valid=(coverage>.50) & np.isfinite(gt) & (gt>0)
        gt=np.where(valid,gt,0.).astype(np.float32)
        left=cv2.resize(read_rgb(source["left_path"]),(180,144),interpolation=cv2.INTER_AREA)
        right=cv2.resize(read_rgb(source["right_path"]),(180,144),interpolation=cv2.INTER_AREA)
        disp=torch.from_numpy(np.stack([gt*scale for scale in SCALES]))[:,None].to(device)
        left_t=torch.from_numpy(np.repeat(left[None],len(SCALES),axis=0)).permute(0,3,1,2).to(device)
        right_t=torch.from_numpy(np.repeat(right[None],len(SCALES),axis=0)).permute(0,3,1,2).to(device)
        with torch.no_grad():
            evidence=stereo_photometric_evidence(left_t,right_t,disp,local_kernel=args.local_kernel,census_kernel=args.census_kernel)
        support=valid & evidence.right_support[:,0].cpu().numpy().all(0) & evidence.census_support[:,0].cpu().numpy().all(0)
        for index,scale in enumerate(SCALES):
            local=evidence.local_rgb_l1[index,0].cpu().numpy(); census=evidence.ternary_census_cost[index,0].cpu().numpy()
            rows.append({"sequence":source["sequence_id"],"frame_id":source["frame_id"],"gt_scale":scale,
                         "valid_count":int(support.sum()),"local_rgb_l1":float(local[support].mean()) if support.any() else math.nan,
                         "ternary_census":float(census[support].mean()) if support.any() else math.nan})
    grouped=defaultdict(list)
    for row in rows: grouped[row["gt_scale"]].append(row)
    summary=[]
    for scale,values in sorted(grouped.items()):
        summary.append({"gt_scale":scale,"frames":len(values),"finite_frames":sum(math.isfinite(x["local_rgb_l1"]) for x in values),
                        "mean_local_rgb_l1":float(np.nanmean([x["local_rgb_l1"] for x in values])),
                        "mean_ternary_census":float(np.nanmean([x["ternary_census"] for x in values])),
                        "median_valid_count":float(np.median([x["valid_count"] for x in values]))})
    winners={}
    by_frame=defaultdict(list)
    for row in rows: by_frame[(row["sequence"],row["frame_id"])].append(row)
    for metric in ("local_rgb_l1","ternary_census"):
        winners[metric]={str(scale):int(count) for scale,count in Counter(min(v,key=lambda x:x[metric])["gt_scale"] for v in by_frame.values()).items()}
    save_csv(args.output/"frame_metrics.csv",rows); save_csv(args.output/"scale_summary.csv",summary)
    save_json(args.output/"aggregate_summary.json",{"purpose":"frozen GT/image consistency audit only; no rescaling applied",
        "frames":len(by_frame),"scales":list(SCALES),"winners":winners,"grid":"144x180 positive-left disparity pixels",
        "local_kernel":args.local_kernel,"census_kernel":args.census_kernel})
    (args.output/"README.md").write_text("# SERV-CT stereo-geometry consistency audit\n\n"
        "Fixed `gt_scale` candidates are a data-validity diagnostic only. They are not a GT correction, a training target, or a model result.\n")
    print(json.dumps(clean({"output":str(args.output),"winners":winners})),flush=True)
    return 0


if __name__=="__main__": raise SystemExit(main())
