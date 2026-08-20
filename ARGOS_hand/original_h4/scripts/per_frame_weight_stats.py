#!/usr/bin/env python3
"""Does the figure's anecdote generalise: is a high mean weight a frame-level warning sign?

Fig. 2 shows two frames of one sequence. On the median frame the head touches 19.2% of the
support, emits a mean effective weight of 0.46, and improves 82.8% of what it touched; on the
worst frame it touches 2.2%, emits 0.79, and improves 12.9%. Read across those two points the
slope is "more weight, more harm" -- which is the opposite sign of the per-pixel AUROC result
the paper reports. Two frames cannot settle that. This script drives the same module over the
whole sequence and writes the per-frame series, so the correlation can be read off instead of
inferred from an anecdote.

Definitions are taken from the figure code rather than restated, so the numbers here are
comparable to the caption:

  * support  = GT coverage AND raw validity (prediction-independent), exactly `dump_qualitative`.
  * w_t      = (fused - raw) / (memory - raw), undefined where |memory - raw| <= 1e-3. The
               recovered convex weight, exactly `dump_qualitative`.
  * de       = |fused - gt| - |raw - gt|, evaluated on support only, exactly
               `export_intervention_figure`. Negative means the module helped.
  * frames are ranked by the signed change in masked EPE, exactly `dump_qualitative`, so the
    "median" and "worst" rows named in the summary are the same two frames the figure shows.

Nothing is trained and no threshold is tuned.

Two data sources, one set of definitions. `--source bida` (default) reads the frozen NPZ
boundary, which only exists for D2. `--source definitive` reads the same frames the
definitive evaluator reads for a SCARED-C split, through its own loaders, so the held-out
D7 split can be asked the same question. The per-frame quantities above are unchanged by
the source; only where the arrays come from changes.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARGOS = ROOT.parents[1]
BIDA = ARGOS / "ARGOS_hand/external_comparison/results/bidastabilizer_raftstereo_robust/d2_full"
BIDA_SEQUENCE = "dataset_2_keyframe_4"
OUT = ROOT.parent / "results" / "per_frame_weight_stats"

FIELDS = ["frame", "frame_id", "support_fraction", "mean_weight_on_support", "mean_delta_e",
          "mean_abs_delta_e", "improved_fraction", "epe_raw", "epe_fused", "delta_epe_full"]


def _mean(values: np.ndarray) -> float:
    """nanmean that reports an empty/all-undefined slice as NaN instead of warning about it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmean(values)) if values.size else float("nan")


def frame_stats(raw, fused, memory, gt, gt_valid, raw_valid) -> dict:
    """Per-frame quantities on one frame. All masks are prediction-independent."""
    support = gt_valid & raw_valid
    error_raw, error_fused = np.abs(raw - gt), np.abs(fused - gt)

    delta = (error_fused - error_raw)[support]
    denominator = memory - raw
    defined = support & np.isfinite(denominator) & (np.abs(denominator) > 1e-3)
    weight = (fused - raw)[defined] / denominator[defined]

    return {
        "support_fraction": float(support.mean()),
        "mean_weight_on_support": _mean(weight),
        "mean_delta_e": _mean(delta),
        "mean_abs_delta_e": _mean(np.abs(delta)),
        "improved_fraction": _mean((delta < 0).astype(float)),
        "epe_raw": _mean(error_raw[support]),
        "epe_fused": _mean(error_fused[support]),
        # The figure's ranking quantity is masked EPE on `support`, i.e. exactly mean_delta_e.
        # This one widens the mask to every GT-valid pixel, so it also charges the module for
        # what it did where the raw prediction was invalid.
        "delta_epe_full": _mean((error_fused - error_raw)[gt_valid]),
    }


def correlate(x: np.ndarray, y: np.ndarray) -> dict:
    """Pearson and Spearman over the pairs where both series are finite."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"n": int(x.size), "pearson_r": None, "spearman_r": None}
    try:  # scipy is in the argos env; the fallback keeps this runnable where it is not
        from scipy.stats import pearsonr, spearmanr
        pearson, spearman = pearsonr(x, y)[0], spearmanr(x, y)[0]
    except ImportError:
        pearson = float(np.corrcoef(x, y)[0, 1])
        spearman = float(np.corrcoef(_rank(x), _rank(y))[0, 1])
    return {"n": int(x.size), "pearson_r": float(pearson), "spearman_r": float(spearman)}


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks, so tied values do not fake a monotone relation."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, float)
    ranks[order] = np.arange(values.size, dtype=float)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.zeros(unique.size)
    np.add.at(sums, inverse, ranks)
    return (sums / counts)[inverse]


def collect(sequence: str, module: str, device_name: str, max_frames: int | None) -> tuple[list[dict], dict]:
    """Drive the module over the sequence and return one row per frame, plus provenance."""
    import torch
    for path in (str(ROOT), str(ROOT / "scripts"), str(ARGOS / "ARGOS_FREEZED/src"),
                 str(ARGOS / "ARGOS-V2/scripts")):
        if path not in sys.path:
            sys.path.insert(0, path)
    from argos_freezed.alignment.sea_raft_adapter import SEARAFTFlowAdapter
    from model_design.comparison.run_comparison import drive, load_factory

    device = torch.device(device_name)
    adapter = load_factory(module)(device=device_name)
    flow_model = SEARAFTFlowAdapter(device=device)

    raw_npz = np.load(BIDA / sequence / "raw.npz", allow_pickle=False)
    evaluation = np.load(BIDA / sequence / "evaluation.npz", allow_pickle=False)
    T = len(raw_npz["frame_ids"])
    if max_frames:
        T = min(T, max_frames)

    # Each `npz[key]` inflates the whole array; hoisted so it happens once, not per frame.
    rgb_left, rgb_right = raw_npz["rgb_left"], raw_npz["rgb_right"]
    raw_disparity, raw_valid_all = raw_npz["raw_disparity"], raw_npz["raw_valid"]
    left = [torch.from_numpy(rgb_left[i:i + 1].copy()).float().to(device) for i in range(T)]
    right = [torch.from_numpy(rgb_right[i:i + 1].copy()).float().to(device) for i in range(T)]
    frames = [{"index": i,
               "raw": torch.from_numpy(raw_disparity[i:i + 1].copy()).float().to(device),
               "raw_valid": torch.from_numpy(raw_valid_all[i:i + 1].copy()).to(device),
               "rgb": left[i], "right_rgb": right[i]} for i in range(T)]

    def flow(current, past):
        a, b = current["index"], past["index"]
        return flow_model.infer(left[a], left[b]), flow_model.infer(left[b], left[a])

    outputs = dict(drive(adapter, frames, flow))

    raw = raw_disparity[:T, 0].astype(np.float64)
    gt = evaluation["gt_disparity"][:T, 0].astype(np.float64)
    gt_valid = evaluation["gt_valid"][:T, 0].astype(bool)
    raw_valid = raw_valid_all[:T, 0].astype(bool)

    def array(index: int, key: str):
        value = outputs[index].get(key)
        return None if value is None else value[0, 0].detach().cpu().numpy().astype(np.float64)

    rows = []
    for i in range(T):
        memory = array(i, "aligned_memory")
        rows.append({"frame": i, "frame_id": str(raw_npz["frame_ids"][i]),
                     **frame_stats(raw[i], array(i, "disparity"),
                                   np.full_like(raw[i], np.nan) if memory is None else memory,
                                   gt[i], gt_valid[i], raw_valid[i])})
    return rows, {"frames_driven": T, "module_provenance": adapter.describe()}


# --- the definitive evaluator's data source, for splits the frozen NPZ boundary never covered ---
#
# `run_comparison._scared` is the definitive path. Its loaders are imported here rather than
# re-parsed; its two private closures (`rgb`, `gt`) cannot be imported, so they are copied
# verbatim below and must stay in step with them.

def _definitive_syspath() -> None:
    """Same prepend order `run_comparison._scared` uses, so `argos_v2` resolves the same way."""
    sys.path[:0] = [str(ROOT / "scripts"), str(ARGOS / "ARGOS_FREEZED/src")]


def _rgb(path: Path) -> np.ndarray:
    """`_scared`'s `rgb` closure, minus the device transfer: [1, 3, 144, 180] float32."""
    import cv2
    from argos_v2.scared_c_data import read_rgb
    value = cv2.resize(read_rgb(path), (180, 144), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(value).transpose(2, 0, 1).astype(np.float32)[None]


def _gt(info, frame_id: str) -> tuple[np.ndarray, np.ndarray]:
    """`_scared`'s `gt` closure: GT resampled to the cache grid, with its coverage fraction."""
    import cv2
    from argos_v2.scared_c_data import load_frame_gt
    value, valid = load_frame_gt(info, frame_id)
    coverage = cv2.resize(valid.astype(np.float32), (180, 144), interpolation=cv2.INTER_AREA)
    numerator = cv2.resize(value * valid.astype(np.float32), (180, 144), interpolation=cv2.INTER_AREA)
    return numerator / np.maximum(coverage, 1e-6) * (180 / value.shape[1]), coverage


def cached_sequences(split: str, backbone: str) -> dict[str, int]:
    """Complete cached sequences for the split, with their frame counts, from the cache itself."""
    from argos_v2.cache_io import COMPLETE_FLAG, load_sequence_cache
    from argos_v2.paths import CACHE_DIR
    prefix = f"dataset_{split.rsplit('-d', 1)[-1]}_"
    return {path.name: int(len(load_sequence_cache(backbone, path.name)[2]))
            for path in sorted((CACHE_DIR / backbone).glob(f"{prefix}*")) if (path / COMPLETE_FLAG).exists()}


def resolve_definitive(split: str, backbone: str, sequence: str | None) -> tuple[str, dict[str, int]]:
    """Default to the longest cached sequence: the most per-frame points for the correlation."""
    _definitive_syspath()
    available = cached_sequences(split, backbone)
    if not available:
        raise SystemExit(f"no complete {backbone} cache for {split}")
    if sequence and sequence not in available:
        raise SystemExit(f"{sequence} is not a complete {backbone} cache; have {sorted(available)}")
    return sequence or max(available, key=lambda name: (available[name], name)), available


def dry_list(split: str, backbone: str, sequence: str | None) -> dict:
    """Resolve the split without touching the model: sequence list plus first-frame shapes."""
    from argos_v2.cache_io import load_sequence_cache
    from argos_v2.paths import CACHE_DIR
    from argos_v2.scared_c_data import load_sequence_info
    sequence, available = resolve_definitive(split, backbone, sequence)
    info = load_sequence_info(sequence)
    disparity, validity, cache_ids, metadata = load_sequence_cache(backbone, sequence)
    if [str(value) for value in cache_ids] != info.frame_ids:
        raise SystemExit(f"frame-ID mismatch: {backbone}/{sequence}")
    gt, coverage = _gt(info, info.frame_ids[0])

    def shape(value) -> list[int]:
        return [int(size) for size in np.asarray(value).shape]

    return {
        "split": split, "backbone": backbone, "available_sequences": available,
        "sequence": sequence, "selection_rule": "most frames among complete cached sequences",
        "frames": len(info.frame_ids), "first_frame_id": info.frame_ids[0],
        "disparity_cache": str(CACHE_DIR / backbone / sequence), "sequence_dir": str(info.seq_dir),
        "cache_completion_status": metadata.get("completion_status"),
        "first_frame_shapes": {
            "rgb_left": shape(_rgb(info.seq_dir / "left" / f"{info.frame_ids[0]}.png")),
            "rgb_right": shape(_rgb(info.seq_dir / "right" / f"{info.frame_ids[0]}.png")),
            "raw": shape(disparity[0]), "raw_valid": shape(validity[0]),
            "gt": shape(gt), "gt_coverage": shape(coverage),
        },
        "dtypes": {"raw": str(disparity.dtype), "raw_valid": str(validity.dtype), "gt": str(gt.dtype)},
    }


def collect_definitive(sequence: str, module: str, device_name: str, max_frames: int | None,
                       split: str, backbone: str) -> tuple[list[dict], dict]:
    """Same as `collect`, with the frames read the way the definitive evaluator reads them."""
    import torch
    for path in (str(ROOT), str(ARGOS / "ARGOS-V2/scripts")):
        if path not in sys.path:
            sys.path.insert(0, path)
    _definitive_syspath()  # last, so `argos_v2` resolves to ROOT/scripts as it does in `_scared`
    from argos_freezed.alignment.sea_raft_adapter import SEARAFTFlowAdapter
    from argos_v2.cache_io import load_sequence_cache
    from argos_v2.paths import CACHE_DIR
    from argos_v2.scared_c_data import load_sequence_info
    from model_design.comparison.run_comparison import drive, load_factory

    device = torch.device(device_name)
    adapter = load_factory(module)(device=device_name)
    flow_model = SEARAFTFlowAdapter(device=device)

    info = load_sequence_info(sequence)
    ids = info.frame_ids[:max_frames] if max_frames else info.frame_ids
    disparity, validity, cache_ids, metadata = load_sequence_cache(backbone, sequence)
    if [str(value) for value in cache_ids[:len(ids)]] != ids:
        raise RuntimeError(f"frame-ID mismatch: {backbone}/{sequence}")
    T = len(ids)

    left = [torch.from_numpy(_rgb(info.seq_dir / "left" / f"{value}.png")).to(device) for value in ids]
    right = [torch.from_numpy(_rgb(info.seq_dir / "right" / f"{value}.png")).to(device) for value in ids]
    gts, covers = zip(*(_gt(info, value) for value in ids))
    frames = [{"index": i,
               "raw": torch.from_numpy(np.asarray(disparity[i:i + 1], np.float32))[:, None].to(device),
               "raw_valid": torch.from_numpy(np.asarray(validity[i:i + 1]) > 0)[:, None].to(device),
               "rgb": left[i], "right_rgb": right[i]} for i in range(T)]

    def flow(current, past):
        a, b = current["index"], past["index"]
        return flow_model.infer(left[a], left[b]), flow_model.infer(left[b], left[a])

    outputs = dict(drive(adapter, frames, flow))

    raw = np.asarray(disparity[:T], np.float64)
    gt = np.asarray(gts, np.float64)
    gt_valid = np.asarray(covers) > .5
    raw_valid = np.asarray(validity[:T]) > 0

    def array(index: int, key: str):
        value = outputs[index].get(key)
        return None if value is None else value[0, 0].detach().cpu().numpy().astype(np.float64)

    rows = []
    for i in range(T):
        memory = array(i, "aligned_memory")
        rows.append({"frame": i, "frame_id": str(ids[i]),
                     **frame_stats(raw[i], array(i, "disparity"),
                                   np.full_like(raw[i], np.nan) if memory is None else memory,
                                   gt[i], gt_valid[i], raw_valid[i])})
    return rows, {
        "frames_driven": T, "module_provenance": adapter.describe(), "split": split,
        "input_boundary": str(info.seq_dir), "disparity_cache": str(CACHE_DIR / backbone / sequence),
        "cache_checkpoint": metadata.get("checkpoint"),
        "frame_source": "argos_v2.scared_c_data + argos_v2.cache_io, as run_comparison._scared",
        "reanchor_schedule": "owned by run_comparison.drive from adapter.horizon, identical to "
                             "the definitive evaluator and to the bida path",
        "support_note": "gt_valid AND raw_valid, as on the bida path; the definitive evaluator's "
                        "stricter protocol_mask is deliberately not applied here",
    }


def summarise(rows: list[dict]) -> dict:
    """Correlations over the per-frame series, plus the two frames the figure shows."""
    column = {field: np.array([row[field] for row in rows], float) for field in FIELDS[2:]}

    # Same ranking rule as `dump_qualitative`: signed change in masked EPE, worst last.
    delta = column["mean_delta_e"]
    order = np.argsort(np.where(np.isnan(delta), np.inf, delta))
    ranked = [int(i) for i in order if np.isfinite(delta[i])]
    figure_frames = {}
    if ranked:
        for label, i in (("median", ranked[len(ranked) // 2]), ("worst", ranked[-1])):
            figure_frames[label] = {k: rows[i][k] for k in FIELDS}

    return {
        "n_frames": len(rows),
        "n_frames_with_weight": int(np.isfinite(column["mean_weight_on_support"]).sum()),
        "correlations": {
            "weight_vs_mean_delta_e": correlate(column["mean_weight_on_support"], column["mean_delta_e"]),
            "weight_vs_delta_epe_full": correlate(column["mean_weight_on_support"], column["delta_epe_full"]),
            "support_vs_mean_delta_e": correlate(column["support_fraction"], column["mean_delta_e"]),
            "support_vs_delta_epe_full": correlate(column["support_fraction"], column["delta_epe_full"]),
        },
        "correlation_reading": "positive r means a higher value goes with more harm, since "
                               "delta_e and delta_epe are signed and negative is an improvement",
        "figure_frames": figure_frames,
    }


def demo() -> None:
    """Self-check: planted correlation signs are recovered, and the per-frame masking holds."""
    rng = np.random.default_rng(0)

    # One synthetic frame with hand-computable answers. Support is 6 of 12 pixels; the module
    # helps on 4 of them and hurts on 2, and the recovered weight is 0.5 wherever it is defined.
    raw = np.array([[10.0] * 12])
    memory = np.array([[12.0] * 12])
    fused = np.array([[11.0] * 12])            # w = (11-10)/(12-10) = 0.5 everywhere
    gt = np.array([[11.0] * 4 + [9.0] * 2 + [11.0] * 6])
    gt_valid = np.array([[True] * 8 + [False] * 4])
    raw_valid = np.array([[True] * 6 + [False] * 2 + [True] * 4])
    stats = frame_stats(raw, fused, memory, gt, gt_valid, raw_valid)
    assert abs(stats["support_fraction"] - 0.5) < 1e-12, stats
    assert abs(stats["mean_weight_on_support"] - 0.5) < 1e-12, stats
    assert abs(stats["improved_fraction"] - 4 / 6) < 1e-12, stats
    # 4 pixels improve by 1px, 2 worsen by 1px -> mean de = (-4 + 2) / 6
    assert abs(stats["mean_delta_e"] - (-2 / 6)) < 1e-12, stats
    assert abs(stats["mean_abs_delta_e"] - 1.0) < 1e-12, stats
    # widening to GT-valid adds 2 pixels where raw was invalid and the module still moved
    assert abs(stats["delta_epe_full"] - (-4 + 2 + 2 * -1) / 8) < 1e-12, stats

    # Planted signs. The figure's anecdote is the positive case: more weight, more harm.
    weight = rng.uniform(0.2, 0.9, 200)
    for sign in (+1.0, -1.0):
        harm = sign * 3.0 * weight + rng.normal(0, 0.05, 200)
        rows = [{"frame": i, "frame_id": str(i), "support_fraction": float(weight[i]),
                 "mean_weight_on_support": float(weight[i]), "mean_delta_e": float(harm[i]),
                 "mean_abs_delta_e": abs(float(harm[i])), "improved_fraction": 0.5,
                 "epe_raw": 1.0, "epe_fused": 1.0, "delta_epe_full": float(harm[i])}
                for i in range(200)]
        found = summarise(rows)["correlations"]["weight_vs_mean_delta_e"]
        assert np.sign(found["pearson_r"]) == sign, found
        assert np.sign(found["spearman_r"]) == sign, found
        assert abs(found["pearson_r"]) > 0.9, found

    # Ties must not fake a monotone relation, and a constant series has no correlation at all.
    assert np.allclose(_rank(np.array([5.0, 1.0, 5.0, 3.0])), [2.5, 0.0, 2.5, 1.0])
    assert correlate(np.ones(50), np.arange(50.0))["pearson_r"] is None

    print(json.dumps({"self_check": "PASS", "frame_stats": stats,
                      "planted_signs_recovered": ["+", "-"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=("bida", "definitive"), default="bida",
                        help="bida: the frozen D2 NPZ boundary. definitive: the loaders the "
                             "definitive evaluator uses, the only way to reach a D7 sequence")
    parser.add_argument("--split", default="scared-d7", help="definitive source only")
    parser.add_argument("--sequence", help=f"default {BIDA_SEQUENCE} for bida; for definitive, "
                        "the longest cached sequence of the split")
    parser.add_argument("--module", default="model_design.comparison.ablation_h4:factory_a2")
    parser.add_argument("--backbone", default="RAFT-Stereo", help="recorded in the manifest; on "
                        "the bida source the frozen NPZ boundary is already backbone-specific, "
                        "on the definitive source this selects the prediction cache")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-frames", type=int, help="limit the driven prefix, for smoke runs")
    parser.add_argument("--output", type=Path, help="default routes by module, like dump_qualitative")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--dry-list", action="store_true", help="definitive source only: resolve "
                        "the sequences and the first frame's shapes without running the model")
    args = parser.parse_args()

    if args.self_check:
        demo(); return
    if args.dry_list:
        if args.source != "definitive":
            raise SystemExit("--dry-list applies to --source definitive")
        print(json.dumps(dry_list(args.split, args.backbone, args.sequence), indent=2)); return

    # The 142-channel results already live at the bare path. A different head must not
    # land on them: this is how a stale table survived a recanonicalisation once.
    out = args.output or (OUT if "canonical_h4" in args.module else OUT / "a2")

    if args.source == "bida":
        sequence = args.sequence or BIDA_SEQUENCE
        rows, provenance = collect(sequence, args.module, args.device, args.max_frames)
        provenance |= {"split": "scared-d2", "input_boundary": str(BIDA / sequence)}
    else:
        sequence, available = resolve_definitive(args.split, args.backbone, args.sequence)
        rows, provenance = collect_definitive(sequence, args.module, args.device, args.max_frames,
                                              args.split, args.backbone)
        provenance["available_sequences"] = available
    summary = summarise(rows)

    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{sequence}_frames.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    (out / f"{sequence}_summary.json").write_text(json.dumps({
        "project": "ARGOS v2", "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "per-frame weight/harm series, to test whether the Fig. 2 anecdote generalises",
        "source": args.source, "sequence": sequence,
        "backbone": args.backbone, "module": args.module,
        "support": "GT coverage AND raw validity; prediction-independent",
        "weight_recovery": "w = (fused - raw) / (memory - raw), undefined where |memory - raw| <= 1e-3",
        "ranking": "signed change in masked EPE, identical to dump_qualitative's rule",
        "training_performed": False, "threshold_tuning_performed": False,
        **provenance, **summary,
    }, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "output": str(out), **summary}, indent=2))


if __name__ == "__main__":
    main()
