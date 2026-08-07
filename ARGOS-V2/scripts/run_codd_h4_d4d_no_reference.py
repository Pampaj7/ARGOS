#!/usr/bin/env python3
"""Frozen canonical H=4 no-reference replay over cached D4D anchor windows."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ARGOS = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from build_multidomain_backbone_cache import d4d_records, read_pair  # noqa: E402
from model_design.external_components.bidavideo import BiDAFlowInferenceAdapter, temporal_disparity_evidence  # noqa: E402
from model_design.models.codd_style_fusion import CODDStyleFusionHead, FrozenResNet18Layer1, build_codd_cues  # noqa: E402

CHECKPOINT = ROOT / "results/codd_style_fusion_probe/bida_memory_phase1/full_phase1/seed_0/checkpoints/best_validation.pt"
CACHE = ROOT / "cache_multidomain_backbones"
CONTEXT = ARGOS / "results/03_temporal_refinement/ood/d4d_s2m2_zero_shot"
H, W = 144, 180


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbones", nargs="+", default=("RAFT-Stereo", "StereoAnywhere"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def image(value: np.ndarray, device: torch.device) -> torch.Tensor:
    import cv2
    value = cv2.resize(value, (W, H), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(np.ascontiguousarray(value)).permute(2, 0, 1)[None].float().to(device)


class Cache:
    def __init__(self, backbone: str):
        path = CACHE / backbone / "D4D"
        if not (path / ".complete").exists(): raise FileNotFoundError(path)
        self.disp = np.load(path / "disparity.npy", mmap_mode="r")
        self.valid = np.load(path / "valid_mask.npy", mmap_mode="r")
        ids = np.load(path / "frame_ids.npy", allow_pickle=True).tolist()
        self.index = {str(x): i for i, x in enumerate(ids)}
        if len(self.index) != len(ids): raise RuntimeError(f"duplicate D4D cache IDs: {path}")

    def get(self, frame_id: str):
        i = self.index[frame_id]
        d = np.asarray(self.disp[i], np.float32)
        return d, np.asarray(self.valid[i], bool) & np.isfinite(d) & (d > 0)


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: path.write_text(""); return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def mean(rows: list[dict]) -> dict:
    out = {"frames": len(rows)}
    for key in rows[0] if rows else ():
        if isinstance(rows[0][key], (int, float)) and key not in {"window_index", "step_since_reset"}:
            values = [float(r[key]) for r in rows if np.isfinite(float(r[key]))]
            if values: out[key] = float(np.mean(values))
    return out


@torch.inference_mode()
def main() -> None:
    cfg = args(); device = torch.device(cfg.device)
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = CODDStyleFusionHead(state["cue_channels"]).to(device).eval()
    model.load_state_dict(state["model"])
    extractor = FrozenResNet18Layer1().to(device).eval()
    flow = BiDAFlowInferenceAdapter("sea_raft", device=device)
    rows: list[dict] = []; rectification = {}
    contexts = {r["anchor_id"]: r for r in csv.DictReader((CONTEXT / "context_manifest.csv").open())}
    index = list(csv.DictReader((CONTEXT / "d4d_index.csv").open()))
    index = [r for r in index if r["sequence_id"] in contexts]
    if cfg.limit: index = index[:cfg.limit]
    source_records = {record.frame_id: record for record in d4d_records()}
    for backbone in cfg.backbones:
        cache = Cache(backbone)
        for window_index, record in enumerate(index):
            stems = contexts[record["sequence_id"]]["context_stems"].split(";")[::-1]
            if len(stems) != 4: raise RuntimeError(f"invalid D4D context: {record['sequence_id']}")
            previous = None
            for age, stem in enumerate(stems):
                frame_id = f"{record['specimen']}__{record['session']}__{stem}"
                raw_np, valid_np = cache.get(frame_id)
                left, right = read_pair(source_records[frame_id], rectification)
                current = (torch.from_numpy(raw_np)[None, None].to(device), torch.from_numpy(valid_np)[None, None].to(device), image(left, device), image(right, device))
                if previous is None:
                    previous = (*current, torch.from_numpy(raw_np)[None, None].to(device))
                    continue
                raw, raw_valid, rgb, right_rgb = current
                past_raw, past_valid, past_rgb, _past_right, past_fused = previous
                forward = flow.current_to_past(rgb, past_rgb); backward = flow.past_to_current(past_rgb, rgb)
                evidence = temporal_disparity_evidence(raw, past_fused, forward, backward, current_valid=raw_valid, past_valid=past_valid, current_rgb=rgb, past_rgb=past_rgb)
                cues = build_codd_cues(extractor, raw=raw, aligned_memory=evidence.aligned_past_disparity, current_rgb=rgb, current_right_rgb=right_rgb, past_rgb=past_rgb, flow_current_to_past=forward, flow_magnitude=evidence.flow_magnitude, forward_backward_confidence=evidence.forward_backward_confidence, warp_support=evidence.warp_support, aligned_valid=evidence.aligned_validity)
                out = model(cues, raw, evidence.aligned_past_disparity)
                mask = raw_valid.bool() & evidence.aligned_validity.bool() & evidence.warp_support.bool()
                def v(x): return float(x[mask].mean()) if bool(mask.any()) else float("nan")
                raw_mc = (raw - evidence.aligned_past_disparity).abs()
                fused_mc = (out.fused_disparity - evidence.aligned_past_disparity).abs()
                rows.append({"dataset": "D4D", "backbone": backbone, "specimen": record["specimen"], "session": record["session"], "sequence": record["sequence_id"], "frame_id": frame_id, "window_index": window_index, "step_since_reset": age, "reset": 0, "support_coverage": float(mask.float().mean()), "raw_mc_inconsistency": v(raw_mc), "fused_mc_inconsistency": v(fused_mc), "mc_delta": v(fused_mc - raw_mc), "update_magnitude": v((out.fused_disparity - raw).abs()), "temporal_weight": v(out.temporal_weight), "fb_confidence": v(evidence.forward_backward_confidence)})
                previous = (raw, raw_valid, rgb, right_rgb, out.fused_disparity)
    cfg.output.mkdir(parents=True, exist_ok=True)
    save_csv(cfg.output / "frame_metrics.csv", rows)
    for key, name in (("backbone", "per_backbone_metrics.csv"), ("specimen", "per_specimen_metrics.csv"), ("session", "per_session_metrics.csv"), ("sequence", "per_sequence_metrics.csv")):
        save_csv(cfg.output / name, [{key: value, **mean([r for r in rows if r[key] == value])} for value in sorted({r[key] for r in rows})])
    summary = mean(rows)
    summary.update({"project": "ARGOS v2", "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256(CHECKPOINT), "protocol": "four-frame curated causal windows; no-reference temporal diagnostic only", "geometry_status": "NOT REPORTED: D4D Zivid stereo-disparity contract is inconsistent"})
    (cfg.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__": main()
