#!/usr/bin/env python3
"""Motion-compensated multi-horizon temporal fidelity (TEPE_corr, OPW) on SCARED-C.

`unified_metrics.py` implements TEPE_corr and OPW, but they are gated on an
`alignment_by_horizon` argument that no existing caller supplies, so they have never been
computed. The paper therefore reports only grid-based DTCE and explicitly disclaims it as
not motion compensated. The flows needed to lift that disclaimer are already computed by
the driver for warping; this script assembles them into the alignment contract and passes
them through.

Alignment convention, read off `unified_metrics.py`: `flow_t_to_tk[:, i]` is applied as
`warp_with_flow(x[:, i+k], flow)`, i.e. it pulls frame `i+k` onto the grid of frame `i`.
SEA-RAFT's `infer(target, source)` returns the flow living on `target`'s grid, so the
required map is `infer(frame_i, frame_{i+k})` -- the *backward* member of the pair the
driver already computes.

`run_comparison.py` and `definitive_evaluation.py` are pinned by freeze manifests and are
imported unchanged. Nothing is trained and no threshold is tuned.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARGOS = ROOT.parents[1]
OUT = ROOT.parent / "results" / "temporal_corrected"
HORIZONS = (1, 2, 4, 8)
D2 = ("dataset_2_keyframe_2", "dataset_2_keyframe_3", "dataset_2_keyframe_4")
D7 = ("dataset_7_keyframe_1", "dataset_7_keyframe_2", "dataset_7_keyframe_3", "dataset_7_keyframe_4")
BACKBONES = ("S2M2-S", "RAFT-Stereo", "StereoAnywhere", "CREStereo", "Fast-FoundationStereo")


def _paths() -> None:
    for path in (str(ROOT), str(ROOT / "scripts"), str(ARGOS / "ARGOS_FREEZED/src"),
                 str(ARGOS / "ARGOS-V2/scripts"),
                 str(ARGOS / "ARGOS_FREEZED/experiments/02_massive_training/scripts")):
        if path not in sys.path:
            sys.path.insert(0, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("d2", "d7"), default="d2")
    parser.add_argument("--backbones", nargs="+", default=list(BACKBONES))
    parser.add_argument("--sequences", nargs="+")
    parser.add_argument("--module", default="model_design.comparison.canonical_h4_masked:factory")
    # A second head must not land on the first one's record: the default path holds the
    # 142-channel run and overwriting it would leave two configurations behind one name.
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--flow-batch-size", type=int, default=32)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()

    import cv2
    import torch
    _paths()
    from argos_freezed.alignment.bida_pull_warp import forward_backward_consistency
    from argos_freezed.alignment.sea_raft_adapter import SEARAFTFlowAdapter
    from argos_v2.cache_io import load_sequence_cache
    from argos_v2.scared_c_data import load_frame_gt, load_sequence_info, read_rgb
    from model_design.comparison.run_comparison import drive, load_factory
    from model_design.metrics.unified_metrics import MetricConfig, evaluate_argos_prediction

    device = torch.device(args.device)
    adapter = load_factory(args.module)(device=args.device)
    flow_model = SEARAFTFlowAdapter(device=device)
    sequences = tuple(args.sequences or (D2 if args.split == "d2" else D7))
    rows = []

    def rgb(path: Path):
        value = cv2.resize(read_rgb(path), (180, 144), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(np.ascontiguousarray(value)).permute(2, 0, 1).float().to(device)[None]

    for sequence in sequences:
        info = load_sequence_info(sequence)
        ids = info.frame_ids[:args.max_frames] if args.max_frames else info.frame_ids
        images = [rgb(info.seq_dir / "left" / f"{v}.png") for v in ids]
        right = [rgb(info.seq_dir / "right" / f"{v}.png") for v in ids]
        gts, covers = [], []
        for frame_id in ids:
            native, valid = load_frame_gt(info, frame_id)
            coverage = cv2.resize(valid.astype(np.float32), (180, 144), interpolation=cv2.INTER_AREA)
            numerator = cv2.resize(native * valid.astype(np.float32), (180, 144), interpolation=cv2.INTER_AREA)
            gts.append(numerator / np.maximum(coverage, 1e-6) * (180 / native.shape[1]))
            covers.append(coverage)
        gts, covers = np.stack(gts), np.stack(covers)
        T = len(ids)

        # Backward flows: entry i pulls frame i+k onto frame i, plus a forward-backward
        # consistency mask used as the correspondence validity.
        alignment: dict[int, dict] = {}
        pair_flow: dict[int, np.ndarray] = {}
        with torch.inference_mode():
            for k in HORIZONS:
                if k >= T:
                    continue
                back, corr = [], []
                for start in range(0, T - k, args.flow_batch_size):
                    idx = list(range(start, min(start + args.flow_batch_size, T - k)))
                    early = torch.cat([images[i] for i in idx])
                    late = torch.cat([images[i + k] for i in idx])
                    onto_early = flow_model.infer(early, late)   # lives on frame i, pulls i+k
                    onto_late = flow_model.infer(late, early)    # reverse, for consistency only
                    fb = forward_backward_consistency(onto_early, onto_late)
                    back.append(onto_early.cpu().numpy())
                    corr.append(fb.confidence[:, 0].cpu().numpy() > 0)
                flow = np.concatenate(back)[None]                # [1, T-k, 2, H, W]
                alignment[k] = {"flow_t_to_tk": flow,
                                "corr_valid": np.concatenate(corr)[None]}
                pair_flow[k] = flow
        print(f"{sequence}: flows ready for horizons {sorted(alignment)}", flush=True)

        for backbone in args.backbones:
            disparity, validity, cache_ids, _ = load_sequence_cache(backbone, sequence)
            if [str(v) for v in cache_ids[:T]] != ids:
                raise RuntimeError(f"frame-ID mismatch: {backbone}/{sequence}")
            frames = [{"index": i,
                       "raw": torch.from_numpy(np.asarray(disparity[i:i + 1], np.float32))[:, None].to(device),
                       "raw_valid": torch.from_numpy(np.asarray(validity[i:i + 1]) > 0)[:, None].to(device),
                       "rgb": images[i], "right_rgb": right[i]} for i in range(T)]

            def flow_pair(current, past):
                a, b = current["index"], past["index"]
                f = flow_model.infer(images[a], images[b])
                r = flow_model.infer(images[b], images[a])
                return f, r

            outputs = dict(drive(adapter, frames, flow_pair))
            refined = np.stack([outputs[i]["disparity"][0, 0].detach().cpu().numpy() for i in range(T)])
            raw = np.asarray(disparity[:T], np.float32)
            protocol = (covers > .5) & (np.asarray(validity[:T]) > 0)

            report = evaluate_argos_prediction(
                raw_disparity=raw[None], refined_disparity=refined[None], gt_disparity=gts[None],
                gt_valid=(covers > .5)[None], protocol_mask=protocol[None], config=MetricConfig(),
                alignment_by_horizon=alignment,
                sequence_ids=[sequence], frame_ids=list(ids))

            # methods[method][FAMILY][STAT][aggregation]; families present depend on whether
            # the alignment contract was accepted, so log them once for provenance.
            temporal = report.get("temporal", {}).get("disparity_px", {})
            for k in sorted(alignment):
                node = temporal.get(str(k), temporal.get(k, {})).get("methods", {})
                if k == sorted(alignment)[0]:
                    print(f"    families at k={k}: {sorted(node.get('raw', {}))}", flush=True)
                for method in ("raw", "refined"):
                    for family in ("TEPE_corr_px", "OPW_px"):
                        block = node.get(method, {})
                        # The pixel-pooled summary exposes "value"; the sequence-aware
                        # aggregate exposes "macro_sequence". Prefer the latter.
                        for source, key in ((block.get("aggregate", {}).get(family, {}), "macro_sequence"),
                                            (block.get(family, {}), "value")):
                            leaf = source.get("MAE", {}) if isinstance(source, dict) else {}
                            value = leaf.get(key) if isinstance(leaf, dict) else None
                            if value is None:
                                continue
                            rows.append({"split": args.split, "sequence": sequence, "backbone": backbone,
                                         "horizon": k, "method": method, "metric": family,
                                         "aggregation": key, "value": float(value),
                                         "support": leaf.get("support_count")})
            print(f"  {backbone}: {len(rows)} rows", flush=True)

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    import csv
    target = out / f"temporal_corrected_{args.split}.csv"
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]) if rows else
                                ["split", "sequence", "backbone", "horizon", "method", "metric", "value", "support"])
        writer.writeheader()
        writer.writerows(rows)
    (out / f"run_manifest_{args.split}.json").write_text(json.dumps({
        "project": "ARGOS v2", "generated_at": datetime.now(timezone.utc).isoformat(),
        "module": args.module, "module_provenance": adapter.describe(),
        "split": args.split, "sequences": list(sequences), "backbones": list(args.backbones),
        "horizons": list(HORIZONS),
        "alignment": "flow_t_to_tk[:, i] = SEA-RAFT infer(frame_i, frame_{i+k}); corr_valid = forward-backward consistency",
        "training_performed": False, "threshold_tuning_performed": False,
        "rows": len(rows),
    }, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "rows": len(rows), "csv": str(target)}, indent=2))


if __name__ == "__main__":
    main()
