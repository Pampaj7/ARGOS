#!/usr/bin/env python3
"""Frozen-only ARGOS v2 OOD validation; contains no training path."""
from __future__ import annotations

import argparse
import csv
import hashlib
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

from model_design.data.temporal_pair_dataset import TemporalPairDataset  # noqa: E402
from model_design.external_components.bidavideo import (  # noqa: E402
    BiDAFlowInferenceAdapter,
    causal_warp,
    temporal_disparity_evidence,
)
from model_design.models.abstention import (  # noqa: E402
    OperatingMode,
    authorization_mask,
    authorized_update,
)
from model_design.models.learned_t1_refiner import LearnedT1Refiner  # noqa: E402
from model_design.models.raw_error_detector import RawErrorDetector, RawErrorEvidence  # noqa: E402

DETECTOR_CKPT = V2_ROOT / "results/raw_error_abstention/full/checkpoints/best_validation.pt"
A2_CKPT = V2_ROOT / "results/learned_t1_refiner/ablations/A2/checkpoints/best_validation.pt"
MODES_PATH = V2_ROOT / "results/raw_error_abstention/full/operating_modes.json"
SOURCE_PATHS = {
    "bidavideo_source": V2_ROOT / "model_design/external_components/bidavideo.py",
    "a2_source": V2_ROOT / "model_design/models/learned_t1_refiner.py",
    "detector_source": V2_ROOT / "model_design/models/raw_error_detector.py",
    "abstention_source": V2_ROOT / "model_design/models/abstention.py",
}
EXPECTED_HASHES = {
    "detector": "78b1bb6cf809dc76448222e41e3bcfafb754bc9b7b6629edcdfa2e1a33444e67",
    "a2": "6cd29277397001333ef3ce630b2f3bc04ec393cdc72e65aa5eb087afd3b389ea",
    "modes": "791f27d21e3f9fa63fe267d5742c4fb85226f49e6027b285aeb90754fbe10b69",
    "bidavideo_source": "133a13f8a4dd89065f736484f1dba1811b40e0f1272d0bbec87d74074bf5c530",
    "a2_source": "6b0b8de0616506058e889c05c5a35af7ec40cc6464fc3ae38357f19ad6dc6bde",
    "detector_source": "fd0fc03be3b3b02d61d047ba415e09e3e09aa1601f043ef6ce004b9cc1b9829f",
    "abstention_source": "ff8af2cbf28a9b85cd83c5d449473f52734c531a5710993408fd3b01a1f1a9f6",
}
H, W = 144, 180


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("crestereo", "servct", "scared_sl", "d4d", "stereomis", "aggregate"), required=True)
    p.add_argument("--output", type=Path, default=V2_ROOT / "results/ood_generalization")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-frames", type=int, default=0, help="smoke/debug limit per sequence; zero means all")
    p.add_argument("--stereomis-sequences", nargs="+", default=["P1", "P2_8", "P3"])
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def verify_frozen() -> dict:
    paths = {"detector": DETECTOR_CKPT, "a2": A2_CKPT, "modes": MODES_PATH}
    paths.update(SOURCE_PATHS)
    actual = {name: sha256(path) for name, path in paths.items()}
    if actual != EXPECTED_HASHES:
        raise RuntimeError(f"frozen artifact hash mismatch: {actual}")
    return {name: {"path": str(paths[name]), "sha256": actual[name]} for name in paths}


def save_json(path: Path, value) -> None:
    def clean(x):
        if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [clean(v) for v in x]
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, float) and not math.isfinite(x): return None
        return x
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: path.write_text(""); return
    keys = list(rows[0])
    for row in rows[1:]: keys += [k for k in row if k not in keys]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None: raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_temporal_contact_sheet(path: Path, rgb: np.ndarray, result: dict) -> None:
    """Save one compact no-reference diagnostic; never persists dense tensors."""
    rgb_small = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
    vmax = max(float(np.quantile(result["raw"], .99)), 1e-3)
    def colour(value, scale):
        u8 = np.clip(value / max(scale, 1e-6) * 255, 0, 255).astype(np.uint8)
        return cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    raw = colour(result["raw"], vmax)
    refined = colour(result["refined"], vmax)
    update = colour(np.abs(result["update"]), 3.0)
    sheet = np.concatenate((rgb_small, raw, refined, update), axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def rgb_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    resized = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(np.ascontiguousarray(resized)).permute(2, 0, 1)[None].float().to(device)


def resize_disparity(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, np.float32)
    return cv2.resize(value, (W, H), interpolation=cv2.INTER_LINEAR) * (W / value.shape[1])


def resize_mask(value: np.ndarray, threshold: float = .5) -> np.ndarray:
    coverage = cv2.resize(value.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    return coverage > threshold


def resize_gt(value: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coverage = cv2.resize(valid.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    numerator = cv2.resize(value.astype(np.float32) * valid, (W, H), interpolation=cv2.INTER_AREA)
    gt = numerator / np.maximum(coverage, 1e-6)
    gt *= W / value.shape[1]
    return gt, coverage > .5, coverage


class FrozenARGOS:
    def __init__(self, device: torch.device):
        self.device = device
        detector_state = torch.load(DETECTOR_CKPT, map_location="cpu", weights_only=False)
        self.detector = RawErrorDetector(detector_state["architecture"], channels=int(detector_state["channels"]))
        self.detector.load_state_dict(detector_state["model"], strict=True)
        self.detector.to(device).eval().requires_grad_(False)
        a2_state = torch.load(A2_CKPT, map_location="cpu", weights_only=False)
        self.a2 = LearnedT1Refiner("A2", tau_px=float(a2_state.get("tau_px", 3.0)))
        self.a2.load_state_dict(a2_state["model"], strict=True)
        self.a2.to(device).eval().requires_grad_(False)
        frozen = json.loads(MODES_PATH.read_text())
        self.mode = OperatingMode(**frozen["modes"]["balanced"])
        self.temperature = float(frozen["temperature"])
        self.flow = BiDAFlowInferenceAdapter("sea_raft", device=device)
        assert all(not p.requires_grad for p in self.detector.parameters())
        assert all(not p.requires_grad for p in self.a2.parameters())
        assert all(not p.requires_grad for p in self.flow.model.parameters())

    @torch.inference_mode()
    def step(self, raw, past, raw_valid, past_valid, current_rgb, past_rgb, past_refined=None):
        raw_t = torch.from_numpy(raw)[None, None].float().to(self.device)
        past_t = torch.from_numpy(past)[None, None].float().to(self.device)
        rv = torch.from_numpy(raw_valid)[None, None].bool().to(self.device)
        pv = torch.from_numpy(past_valid)[None, None].bool().to(self.device)
        cr = rgb_tensor(current_rgb, self.device); pr = rgb_tensor(past_rgb, self.device)
        if self.device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        flows = self.flow.infer(torch.cat((cr, pr)), torch.cat((pr, cr))).clone()
        evidence_obj = temporal_disparity_evidence(raw_t, past_t, flows[:1], flows[1:],
            current_valid=rv, past_valid=pv, current_rgb=cr, past_rgb=pr)
        evidence = {k: v.detach() for k, v in evidence_obj.as_dict().items()}
        evidence["current_valid"] = rv
        proposal = self.a2(raw_t, evidence, cr)
        inp = RawErrorEvidence(raw_t, rv, evidence["aligned_past_disparity"],
            evidence["aligned_validity"], evidence["warp_support"],
            evidence["forward_backward_error"], evidence["forward_backward_confidence"],
            evidence["photometric_residual"], evidence["flow_magnitude"],
            proposal.update, proposal.g_error, proposal.c_memory)
        prediction = self.detector(inp)
        auth = authorization_mask(prediction, mode=self.mode, temperature=self.temperature,
            aligned_valid=evidence["aligned_validity"], warp_support=evidence["warp_support"],
            proposal_update=proposal.update)
        refined = authorized_update(raw_t, proposal.update, auth)
        if past_refined is None: past_refined_t = past_t
        else: past_refined_t = torch.from_numpy(past_refined)[None, None].float().to(self.device)
        aligned_refined = causal_warp(past_refined_t, flows[:1], source_valid=pv)
        if self.device.type == "cuda": torch.cuda.synchronize()
        runtime_ms = (time.perf_counter() - t0) * 1000
        def n(x): return x[0, 0].float().cpu().numpy()
        return {
            "raw": raw, "refined": n(refined), "update": n(refined-raw_t),
            "authorization": n(auth.float()) > .5, "probability": n(prediction.probability),
            "mu": n(prediction.mu), "sigma": n(prediction.sigma),
            "aligned_raw_past": n(evidence["aligned_past_disparity"]),
            "aligned_refined_past": n(aligned_refined.warped),
            "support": n(evidence["aligned_validity"].float()) > .5,
            "runtime_ms": runtime_ms,
        }


def edge_mask(gt: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gt, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(gt, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.dilate((np.hypot(gx, gy) > 1).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def geometry_metrics(raw, refined, update, support, valid, gt, change_threshold=.05) -> dict:
    """Paired geometry/safety metrics for a supplied evaluation grid."""
    common = valid & support & np.isfinite(gt) & np.isfinite(raw) & np.isfinite(refined)
    if not common.any():
        return {"valid_count": 0, "clean_count": 0, "modified_count": 0,
                **{k: math.nan for k in ("raw_epe", "refined_epe", "epe_gain", "raw_bad1",
                    "refined_bad1", "raw_bad3", "refined_bad3", "raw_boundary_epe",
                    "refined_boundary_epe", "new_bad3", "false_update_rate",
                    "clean_degradation", "intervention_precision", "harmful_modified_rate",
                    "mean_clean_update")}}
    er = np.abs(raw-gt); ef = np.abs(refined-gt); changed = np.abs(update) > change_threshold
    clean = common & (er <= .5); modified = common & changed
    helpful = modified & (ef + .02 < er); harmful = modified & (ef > er + .02)
    raw_good3 = common & (er <= 3); boundary = common & edge_mask(gt)
    return {
        "valid_count": int(common.sum()), "clean_count": int(clean.sum()), "modified_count": int(modified.sum()),
        "raw_epe": float(er[common].mean()), "refined_epe": float(ef[common].mean()),
        "epe_gain": float((er-ef)[common].mean()),
        "raw_bad1": float((er[common]>1).mean()), "refined_bad1": float((ef[common]>1).mean()),
        "raw_bad3": float((er[common]>3).mean()), "refined_bad3": float((ef[common]>3).mean()),
        "raw_boundary_epe": float(er[boundary].mean()) if boundary.any() else math.nan,
        "refined_boundary_epe": float(ef[boundary].mean()) if boundary.any() else math.nan,
        "new_bad3": float((ef[raw_good3]>3).mean()) if raw_good3.any() else 0.,
        "false_update_rate": float((changed&clean).sum()/max(clean.sum(),1)),
        "clean_degradation": float((harmful&clean).sum()/max(clean.sum(),1)),
        "intervention_precision": float(helpful.sum()/max(modified.sum(),1)),
        "harmful_modified_rate": float(harmful.sum()/max(modified.sum(),1)),
        "mean_clean_update": float(np.abs(update)[clean].mean()) if clean.any() else 0.,
        "support_ratio": float(support.mean()), "intervention_coverage_all": float((changed&support).sum()/max(support.sum(),1)),
        "mean_update_abs": float(np.abs(update)[support].mean()) if support.any() else 0.,
        "max_update_abs": float(np.abs(update)[support].max()) if support.any() else 0.,
        "catastrophic_update_gt3": float((np.abs(update)[support] > 3*(raw.shape[1]/W)+1e-4).mean()) if support.any() else 0.,
    }


def native_geometry_from_cache_update(result, raw_native, gt_native, valid_native):
    h, w = raw_native.shape; scale = w/W
    update = cv2.resize(result["update"], (w,h), interpolation=cv2.INTER_LINEAR) * scale
    support = cv2.resize(result["support"].astype(np.uint8), (w,h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return geometry_metrics(raw_native, raw_native+update, update, support, valid_native, gt_native,
                            change_threshold=.05*scale)


def metrics(result: dict, valid: np.ndarray | None = None, gt: np.ndarray | None = None) -> dict:
    support = result["support"]
    raw_temporal = np.abs(result["raw"] - result["aligned_raw_past"])
    ref_temporal = np.abs(result["refined"] - result["aligned_refined_past"])
    changed = np.abs(result["update"]) > .05
    out = {
        "runtime_ms": result["runtime_ms"], "support_ratio": float(support.mean()),
        "intervention_coverage_all": float((changed & support).sum()/max(support.sum(), 1)),
        "authorization_ratio": float((result["authorization"] & support).sum()/max(support.sum(), 1)),
        "mean_update_abs": float(np.abs(result["update"])[support].mean()) if support.any() else 0.,
        "p95_update_abs": float(np.quantile(np.abs(result["update"])[support], .95)) if support.any() else 0.,
        "max_update_abs": float(np.abs(result["update"])[support].max()) if support.any() else 0.,
        "raw_mc_temporal_error": float(raw_temporal[support].mean()) if support.any() else math.nan,
        "refined_mc_temporal_error": float(ref_temporal[support].mean()) if support.any() else math.nan,
        "temporal_delta": float((ref_temporal-raw_temporal)[support].mean()) if support.any() else math.nan,
        "raw_temporal_variance": float(raw_temporal[support].var()) if support.any() else math.nan,
        "refined_temporal_variance": float(ref_temporal[support].var()) if support.any() else math.nan,
        "temporal_worsened_ratio": float(((ref_temporal > raw_temporal+.02)&support).sum()/max(support.sum(),1)),
        "catastrophic_update_gt3": float((np.abs(result["update"])[support] > 3.0001).mean()) if support.any() else 0.,
    }
    if gt is None or valid is None: return out
    common = valid & support & np.isfinite(gt) & np.isfinite(result["raw"])
    er = np.abs(result["raw"]-gt); ef = np.abs(result["refined"]-gt)
    if not common.any():
        out.update({
            "valid_count": 0, "clean_count": 0, "modified_count": 0,
            "raw_epe": math.nan, "refined_epe": math.nan, "epe_gain": math.nan,
            "raw_bad1": math.nan, "refined_bad1": math.nan,
            "raw_bad3": math.nan, "refined_bad3": math.nan,
            "raw_boundary_epe": math.nan, "refined_boundary_epe": math.nan,
            "new_bad3": math.nan, "false_update_rate": math.nan,
            "clean_degradation": math.nan, "intervention_precision": math.nan,
            "harmful_modified_rate": math.nan, "mean_clean_update": math.nan,
        })
        return out
    clean = common & (er <= .5); modified = common & changed
    helpful = modified & (ef + .02 < er); harmful = modified & (ef > er + .02)
    raw_good3 = common & (er <= 3); boundary = common & edge_mask(gt)
    out.update({
        "valid_count": int(common.sum()), "clean_count": int(clean.sum()), "modified_count": int(modified.sum()),
        "raw_epe": float(er[common].mean()), "refined_epe": float(ef[common].mean()),
        "epe_gain": float((er-ef)[common].mean()),
        "raw_bad1": float((er[common]>1).mean()), "refined_bad1": float((ef[common]>1).mean()),
        "raw_bad3": float((er[common]>3).mean()), "refined_bad3": float((ef[common]>3).mean()),
        "raw_boundary_epe": float(er[boundary].mean()) if boundary.any() else math.nan,
        "refined_boundary_epe": float(ef[boundary].mean()) if boundary.any() else math.nan,
        "new_bad3": float((ef[raw_good3]>3).mean()) if raw_good3.any() else 0.,
        "false_update_rate": float((changed&clean).sum()/max(clean.sum(),1)),
        "clean_degradation": float((harmful&clean).sum()/max(clean.sum(),1)),
        "intervention_precision": float(helpful.sum()/max(modified.sum(),1)),
        "harmful_modified_rate": float(harmful.sum()/max(modified.sum(),1)),
        "mean_clean_update": float(np.abs(result["update"])[clean].mean()) if clean.any() else 0.,
    })
    return out


def aggregate(rows: list[dict], dataset: str, protocol: str) -> dict:
    out = {"dataset": dataset, "protocol": protocol, "frames": len(rows)}
    numeric = sorted({k for r in rows for k,v in r.items() if isinstance(v,(int,float)) and k not in {"valid_count","clean_count","modified_count"}})
    for key in numeric:
        vals = [float(r[key]) for r in rows if key in r and math.isfinite(float(r[key]))]
        if vals: out[key] = float(np.mean(vals))
    maxima = [float(r["max_update_abs"]) for r in rows
              if "max_update_abs" in r and math.isfinite(float(r["max_update_abs"]))]
    if maxima: out["max_update_abs"] = max(maxima)
    if any("valid_count" in r for r in rows):
        valid = sum(r.get("valid_count",0) for r in rows); clean=sum(r.get("clean_count",0) for r in rows); modified=sum(r.get("modified_count",0) for r in rows)
        out["evaluated_frames"] = sum(r.get("valid_count", 0) > 0 for r in rows)
        out["valid_count"] = valid; out["clean_count"] = clean; out["modified_count"] = modified
        for key in ("raw_epe","refined_epe","epe_gain","raw_bad1","refined_bad1","raw_bad3","refined_bad3","new_bad3"):
            terms = [(r[key], r.get("valid_count", 0)) for r in rows
                     if r.get("valid_count", 0) > 0 and key in r and math.isfinite(float(r[key]))]
            out[key] = sum(value * count for value, count in terms) / max(sum(count for _, count in terms), 1)
        for key in ("false_update_rate","clean_degradation","mean_clean_update"):
            terms = [(r[key], r.get("clean_count", 0)) for r in rows
                     if r.get("clean_count", 0) > 0 and key in r and math.isfinite(float(r[key]))]
            out[key] = sum(value * count for value, count in terms) / max(sum(count for _, count in terms), 1)
        terms = [(r["intervention_precision"], r.get("modified_count", 0)) for r in rows
                 if r.get("modified_count", 0) > 0 and "intervention_precision" in r
                 and math.isfinite(float(r["intervention_precision"]))]
        out["intervention_precision"] = sum(value * count for value, count in terms) / max(sum(count for _, count in terms), 1)
        degradations=[r["refined_epe"]-r["raw_epe"] for r in rows
                      if r.get("valid_count", 0) > 0 and math.isfinite(float(r.get("refined_epe", math.nan)))
                      and math.isfinite(float(r.get("raw_epe", math.nan)))]
        out["frames_worsened"] = float(np.mean(np.array(degradations)>0)) if degradations else math.nan
        out["worst_frame_degradation"] = float(max(degradations)) if degradations else math.nan
        out["p95_frame_degradation"] = float(np.quantile(degradations,.95)) if degradations else math.nan
    return out


def finalize(out_dir: Path, dataset: str, protocol: str, rows: list[dict], extra=None):
    write_csv(out_dir / "frame_metrics.csv", rows)
    by_sequence=[]
    for sequence in sorted({r["sequence"] for r in rows}):
        group=[r for r in rows if r["sequence"]==sequence]
        by_sequence.append({"sequence":sequence, **aggregate(group,dataset,protocol)})
    write_csv(out_dir / "sequence_metrics.csv", by_sequence)
    summary=aggregate(rows,dataset,protocol)
    if extra: summary.update(extra)
    save_json(out_dir / "summary.json", summary)
    report_keys=("frames","evaluated_frames","valid_count","raw_epe","refined_epe","epe_gain",
                 "raw_bad1","refined_bad1","raw_bad3","refined_bad3","raw_boundary_epe",
                 "refined_boundary_epe","false_update_rate","clean_degradation","new_bad3",
                 "intervention_coverage_all","intervention_precision","mean_clean_update",
                 "raw_mc_temporal_error","refined_mc_temporal_error","temporal_delta",
                 "frames_worsened","worst_frame_degradation","p95_frame_degradation",
                 "catastrophic_update_gt3","runtime_ms","stereo_runtime_ms")
    lines=[f"# {dataset}","",protocol,"", "| Metric | Value |","|---|---:|"]
    for key in report_keys:
        if key in summary and not isinstance(summary[key],dict): lines.append(f"| `{key}` | {summary[key]} |")
    lines += ["", "Raw and refined geometry use the same paired valid/support mask. See `frame_metrics.csv` and `sequence_metrics.csv` for disaggregated results.", ""]
    (out_dir/"README.md").write_text("\n".join(lines))
    return summary


def run_crestereo(pipe, args):
    out=args.output/"crestereo"; ds=TemporalPairDataset(["CREStereo"],["dataset_7_keyframe_3","dataset_7_keyframe_4"],coverage_threshold=.5,max_pairs_per_sequence=args.max_frames or 160,random_clip_start=False)
    rows=[]; previous={}
    for i in range(len(ds)):
        item=ds[i]; seq=item["sequence"]; raw=item["raw"][0].numpy(); past=item["past"][0].numpy()
        result=pipe.step(raw,past,item["raw_valid"][0].numpy(),item["past_valid"][0].numpy(),item["current_rgb"].permute(1,2,0).numpy().astype(np.uint8),item["past_rgb"].permute(1,2,0).numpy().astype(np.uint8),previous.get(seq))
        previous[seq]=result["refined"]
        row={"sequence":seq,"frame_id":item["current_frame_id"],"backbone":"CREStereo",**metrics(result,item["gt_valid"][0].numpy(),item["gt"][0].numpy())}; rows.append(row)
    return finalize(out,"CREStereo","second unseen backbone; frozen SCARED-C keyframes 3/4",rows)


def run_servct(pipe,args):
    root=ARGOS_ROOT/"results/03_temporal_refinement/ood/prepared/servct"; manifest=list(csv.DictReader((root/"sequence_manifest.csv").open())); rows=[]
    for seq in sorted({r["sequence_id"] for r in manifest}):
        seqrows=sorted([r for r in manifest if r["sequence_id"]==seq],key=lambda r:int(r["order_index"])); z=np.load(root/"shards"/f"{seq}.npz"); previous=None
        limit=args.max_frames or len(seqrows)
        for j in range(1,min(len(seqrows),limit)):
            cur,prev=seqrows[j],seqrows[j-1]; raw=z["raw_disp"][j].astype(np.float32); past=z["raw_disp"][j-1].astype(np.float32); valid=np.isfinite(raw)&(raw>0); pvalid=np.isfinite(past)&(past>0)
            result=pipe.step(raw,past,valid,pvalid,read_rgb(Path(cur["left_path"])),read_rgb(Path(prev["left_path"])),previous); previous=result["refined"]
            rows.append({"sequence":seq,"frame_id":cur["frame_id"],"continuity":"weak_sparse",**metrics(result,z["valid_mask"][j].astype(bool),z["gt_disp"][j].astype(np.float32))})
    return finalize(args.output/"servct","SERV-CT","two weak-sparse 8-frame causal replays; CT-reference GT",rows)


def s2m2_model(device):
    path=ARGOS_ROOT/"scripts/temporal_refinement/data_prep"; sys.path.insert(0,str(path))
    from predict_s2m2_long_sequences import build_model, infer
    return build_model(device,"S"),infer


def run_scared_sl(pipe,args):
    model,infer=s2m2_model(pipe.device); root=ARGOS_ROOT/"dataset/SCARED/curated/geometric_gt/strong_keyframes_rectified"; rows=[]; native_rows=[]
    dirs=[p for p in root.glob("dataset_*/keyframe_*") if p.is_dir()]
    if args.max_frames: dirs=dirs[:args.max_frames]
    for d in dirs:
        left=read_rgb(d/"left_rectified.png"); right=read_rgb(d/"right_rectified.png"); raw_native,stereo_ms,_=infer(model,left,right,pipe.device,512); raw=resize_disparity(raw_native); valid=np.isfinite(raw)&(raw>0)
        gt_native=np.load(d/"gt_disparity.npy").astype(np.float32); mask=cv2.imread(str(d/"valid_mask.png"),0)>0; gt,gtvalid,_=resize_gt(gt_native,mask)
        result=pipe.step(raw,raw,valid,valid,left,left,raw); row={"sequence":d.parent.name,"frame_id":d.name,"stereo_runtime_ms":stereo_ms,"protocol_note":"static_repeat",**metrics(result,gtvalid,gt)}; rows.append(row)
        native_rows.append({"sequence":d.parent.name,"frame_id":d.name,"stereo_runtime_ms":stereo_ms,
                            **native_geometry_from_cache_update(result,raw_native,gt_native,mask)})
    cache=finalize(args.output/"scared_structured_light","SCARED structured-light","45 direct-GT static-repeat preservation anchors at 144x180",rows)
    native=finalize(args.output/"scared_structured_light_native_grid","SCARED structured-light native grid",
                    "native stereo plus upscaled cache-grid correction; static-repeat preservation",native_rows)
    return {"cache_grid":cache,"native_grid_from_cache_correction":native}


def d4d_rgb(row, stem, cache):
    key=(row["specimen"],row["session"])
    if key not in cache:
        path=ARGOS_ROOT/"scripts/temporal_refinement/ood/d4d"; sys.path.insert(0,str(path))
        from d4d_keyframe_gt import load_cam,rectify_maps,session_root
        sr=session_root(row["specimen"])/row["session"]; cam=load_cam(sr/"camera_info/left.yaml"); cache[key]=(sr,rectify_maps(cam))
    sr,maps=cache[key]; bgr=cv2.imread(str(sr/"left_images"/f"{stem}.png")); rect=cv2.remap(bgr,maps[0],maps[1],cv2.INTER_LINEAR)
    return cv2.cvtColor(rect,cv2.COLOR_BGR2RGB)


def run_d4d(pipe,args):
    root=ARGOS_ROOT/"results/03_temporal_refinement/ood/d4d_s2m2_zero_shot"; index=list(csv.DictReader((root/"d4d_index.csv").open())); contexts={r["anchor_id"]:r for r in csv.DictReader((root/"context_manifest.csv").open())}; cache={}; anchor_rows=[]; native_anchor_rows=[]; temporal_rows=[]
    if args.max_frames:index=index[:args.max_frames]
    for row in index:
        z=np.load(row["target_path"]); raws=z["raw_disp"].astype(np.float32); stems=contexts[row["sequence_id"]]["context_stems"].split(";")[::-1]; rgbs=[d4d_rgb(row,s,cache) for s in stems]; previous=None
        for j in range(1,4):
            raw=resize_disparity(raws[j]); past=resize_disparity(raws[j-1]); valid=np.isfinite(raw)&(raw>0); pvalid=np.isfinite(past)&(past>0)
            result=pipe.step(raw,past,valid,pvalid,rgbs[j],rgbs[j-1],previous); previous=result["refined"]
            base={"sequence":f"{row['specimen']}__{row['session']}","window":row["sequence_id"],"frame_id":stems[j],"transition":j,"quality":row["quality"]}
            temporal_rows.append({**base,**metrics(result)})
            if j==3:
                gt_native=z["gt_disp"][3].astype(np.float32); mask=z["valid_mask"][3].astype(bool); gt,gtvalid,_=resize_gt(gt_native,mask)
                anchor_rows.append({**base,**metrics(result,gtvalid,gt)})
                native_anchor_rows.append({**base,**native_geometry_from_cache_update(result,raws[3],gt_native,mask)})
    anchor=finalize(args.output/"d4d_anchors","D4D anchors","curated-pose Zivid GT at final frame of causal windows",anchor_rows)
    native_anchor=finalize(args.output/"d4d_anchors_native_grid","D4D anchors native grid",
                           "original 178x224 raw plus upscaled cache-grid correction",native_anchor_rows)
    temporal=finalize(args.output/"d4d_temporal","D4D temporal windows","156 validated four-frame windows; three causal transitions each",temporal_rows)
    return {"anchors_cache_grid":anchor,"anchors_native_grid_from_cache_correction":native_anchor,"temporal":temporal}


def run_stereomis(pipe,args):
    model,infer=s2m2_model(pipe.device); root=ARGOS_ROOT/"dataset/StereoMIS/curated/geometric_gt/temporal_sequences"; all_summaries={}
    for seq in args.stereomis_sequences:
        out=args.output/"stereomis"/seq; complete=out/"complete.json"
        if complete.exists() and not args.max_frames:
            all_summaries[seq]=json.loads(complete.read_text()); continue
        lefts=sorted((root/seq/"left").glob("*")); rights={p.stem:p for p in (root/seq/"right").glob("*")}; lefts=[p for p in lefts if p.stem in rights]
        if args.max_frames:lefts=lefts[:args.max_frames]
        rows=[]; previous=None; prev_raw=None; prev_rgb=None; stereo_times=[]
        diagnostic_indices = {1, max(1, len(lefts)//2), max(1, len(lefts)-1)}
        for i,left_path in enumerate(lefts):
            left=read_rgb(left_path); right=read_rgb(rights[left_path.stem]); raw_native,stereo_ms,_=infer(model,left,right,pipe.device,512); stereo_times.append(stereo_ms); raw=resize_disparity(raw_native); valid=np.isfinite(raw)&(raw>0)
            if i:
                result=pipe.step(raw,prev_raw,valid,np.isfinite(prev_raw)&(prev_raw>0),left,prev_rgb,previous); previous=result["refined"]
                rows.append({"sequence":seq,"frame_id":left_path.stem,"stereo_runtime_ms":stereo_ms,**metrics(result)})
                if not args.max_frames and i in diagnostic_indices:
                    save_temporal_contact_sheet(out/"diagnostics"/f"{seq}_{left_path.stem}.png", left, result)
            else: previous=raw
            prev_raw=raw; prev_rgb=left
            if i and i%500==0: print(f"StereoMIS {seq}: {i}/{len(lefts)}",flush=True)
        summary=finalize(out,"StereoMIS","rectified no-GT continuous video; no-reference temporal metrics only",rows,{"stereo_runtime_ms":float(np.mean(stereo_times))})
        if not args.max_frames: save_json(complete,summary)
        all_summaries[seq]=summary
    combined=[]
    for seq in args.stereomis_sequences:
        p=args.output/"stereomis"/seq/"frame_metrics.csv"
        if p.exists(): combined.extend(list(csv.DictReader(p.open())))
    # Convert numeric fields back for aggregation.
    numeric_rows=[]
    for row in combined:
        numeric_rows.append({k:(float(v) if k not in {"sequence","frame_id"} and v not in {"","nan"} else v) for k,v in row.items()})
    overall=finalize(args.output/"stereomis","StereoMIS","three rectified no-GT sequences; no-reference only",numeric_rows,{"per_sequence":all_summaries})
    return overall


def main():
    args=parse_args(); args.output.mkdir(parents=True,exist_ok=True); frozen=verify_frozen(); save_json(args.output/"frozen_manifest.json",frozen)
    if args.dataset=="aggregate":
        summaries={}
        for d in ("crestereo","servct","scared_structured_light","scared_structured_light_native_grid",
                  "d4d_anchors","d4d_anchors_native_grid","d4d_temporal","stereomis"):
            p=args.output/d/"summary.json"
            summaries[d]=json.loads(p.read_text()) if p.exists() else {"status":"missing"}
        save_json(args.output/"aggregate_summary.json",{"frozen_artifacts":frozen,"datasets":summaries}); print(json.dumps(summaries,indent=2)); return
    device=torch.device(args.device); pipe=FrozenARGOS(device)
    fn={"crestereo":run_crestereo,"servct":run_servct,"scared_sl":run_scared_sl,"d4d":run_d4d,"stereomis":run_stereomis}[args.dataset]
    result=fn(pipe,args); print(json.dumps(result,indent=2,default=str))


if __name__=="__main__": main()
