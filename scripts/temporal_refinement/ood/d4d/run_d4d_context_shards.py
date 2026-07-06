#!/usr/bin/env python3
"""D4D zero-shot: run S2M2-S on each keyframe anchor + its 3 causal predecessors,
build training-format 4-frame shards so the existing refiner harness runs unchanged.

Reuses the validated S2M2-S recipe (predict_s2m2_long_sequences.infer, variant S, width 512,
disparity in original image coords), D4D rectification (d4d_keyframe_gt), and the training
downsample (target_scale=0.25). GT disparity comes from the processed benchmark (Zivid).

Per anchor -> shard npz {raw_disp[4], gt_disp[4], valid_mask[4], delta[4]} with GT only at the
anchor frame (index 3 = t); context frames t-1..t-3 carry raw disparity but valid=0 (no GT).
Causal window [t, t-1, t-2, t-3], same session, timestamp-ordered, NO future frames.

Outputs: raw/s2m2_s_raw/<anchor>.npy (anchor raw disp, full res) + shards/<anchor>.npz + index.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path("/dtu/p1/leopam/ARGOS")
for p in ("scripts/temporal_refinement/ood/d4d", "scripts/temporal_refinement/data_prep",
          "scripts/temporal_refinement/eval_scripts"):
    sys.path.insert(0, str(ROOT / p))
from d4d_keyframe_gt import load_cam, rectify_maps, session_root  # noqa: E402
from predict_s2m2_long_sequences import build_model, infer  # noqa: E402
from generate_distillation_targets_selected_clips import target_hw, valid_masked_downsample_disparity  # noqa: E402

OUT = ROOT / "results/03_temporal_refinement/ood/d4d_s2m2_zero_shot"
TARGET_SCALE = 0.25
MIN_VALID_RATIO = 0.25
CTX = 4


def ts_of(name: str) -> float:
    a, b = Path(name).stem.split("_"); return float(f"{a}.{b}")


def session_frames(specimen: str, session: str):
    li = session_root(specimen) / session / "left_images"
    fs = sorted(li.glob("*.png"), key=lambda p: ts_of(p.name))
    return fs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path,
                    default=ROOT / "dataset/D4D/processed/keyframe_stereo_gt/manifests/valid_and_warning_manifest.csv")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N anchors")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    rows = list(csv.DictReader(args.manifest.open()))
    if args.limit:
        rows = rows[:args.limit]
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = build_model(device, "S")
    raw_dir = args.out / "raw" / "s2m2_s_raw"; raw_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.out / "shards"; shard_dir.mkdir(parents=True, exist_ok=True)

    map_cache: dict = {}       # session -> (lmapx,lmapy,rmapx,rmapy,W,H)
    disp_cache: dict = {}      # (session, frame_stem) -> full-res raw disp
    index_rows, skipped, ctx_rows = [], [], []

    def rect_maps(specimen, session):
        if session in map_cache:
            return map_cache[session]
        ci = session_root(specimen) / session / "camera_info"
        left, right = load_cam(ci / "left.yaml"), load_cam(ci / "right.yaml")
        lm = rectify_maps(left); rm = rectify_maps(right)
        map_cache[session] = (lm[0], lm[1], rm[0], rm[1], left["W"], left["H"])
        return map_cache[session]

    def s2m2_frame(specimen, session, stem):
        key = (session, stem)
        if key in disp_cache:
            return disp_cache[key]
        sr = session_root(specimen) / session
        lp = sr / "left_images" / f"{stem}.png"; rp = sr / "right_images" / f"{stem}.png"
        if not rp.exists():
            disp_cache[key] = None; return None
        lmapx, lmapy, rmapx, rmapy, W, H = rect_maps(specimen, session)
        l = cv2.remap(cv2.imread(str(lp)), lmapx, lmapy, cv2.INTER_LINEAR)
        r = cv2.remap(cv2.imread(str(rp)), rmapx, rmapy, cv2.INTER_LINEAR)
        l = cv2.cvtColor(l, cv2.COLOR_BGR2RGB); r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
        disp, _ms, _sx = infer(model, l, r, device, 512)
        disp_cache[key] = disp.astype(np.float32)
        if len(disp_cache) > 400:  # cap RAM
            disp_cache.pop(next(iter(disp_cache)))
        return disp_cache[key]

    for r in rows:
        aid = r["anchor_id"]; spec = r["specimen_id"]; sess = r["session_id"]
        if not (ROOT / r["right_rectified_path"]).exists():
            skipped.append({"anchor_id": aid, "reason": "missing_right_view"}); continue
        frames = session_frames(spec, sess)
        stems = [p.stem for p in frames]
        t_ts = float(r["stereo_timestamp"])
        # anchor stem = nearest frame to stereo_timestamp (== the GT-pipeline choice)
        ti = int(np.argmin([abs(ts_of(s + ".png") - t_ts) for s in stems]))
        win_idx = [max(0, ti - k) for k in range(CTX)]  # [t, t-1, t-2, t-3] clamped
        win_stems = [stems[i] for i in win_idx]
        # S2M2 on each context frame
        raws = []
        ok = True
        for st in win_stems:
            d = s2m2_frame(spec, sess, st)
            if d is None:
                ok = False; break
            raws.append(d)
        if not ok:
            skipped.append({"anchor_id": aid, "reason": "context_frame_missing_right"}); continue

        # GT disparity (anchor, full res) + valid
        gt_full = np.load(ROOT / r["gt_disparity_path"]).astype(np.float32)
        valid_full = cv2.imread(str(ROOT / r["valid_mask_path"]), cv2.IMREAD_GRAYSCALE) > 0
        H0, W0 = gt_full.shape
        out_h, out_w = target_hw((H0, W0), TARGET_SCALE)
        # downsample each raw to grid (valid = finite raw)
        raw_ds = []
        for d in raws:
            dr, _ = valid_masked_downsample_disparity(d, np.isfinite(d) & (d > 0), out_h, out_w, MIN_VALID_RATIO)
            raw_ds.append(dr.astype(np.float16))
        vfull = valid_full & np.isfinite(gt_full) & (gt_full > 0) & np.isfinite(raws[0])
        gt_ds, gvalid = valid_masked_downsample_disparity(gt_full, vfull, out_h, out_w, MIN_VALID_RATIO)
        raw0_ds, _ = valid_masked_downsample_disparity(raws[0], vfull, out_h, out_w, MIN_VALID_RATIO)
        valid_anchor = gvalid & np.isfinite(gt_ds) & np.isfinite(raw0_ds) & (gt_ds > 0)
        # shard arrays: GT/valid only at anchor (index 0 of window == current t; FullFrameDataset
        # builds window from raw_disp[offset..offset-3]; we store frames in TEMPORAL order 0..3 = t-3..t)
        # store temporal order oldest->newest so offset=CTX-1 = anchor
        raw_stack = np.stack(raw_ds[::-1])          # [t-3, t-2, t-1, t]
        gt_stack = np.zeros((CTX, out_h, out_w), np.float16)
        val_stack = np.zeros((CTX, out_h, out_w), np.uint8)
        delta = np.zeros((CTX, out_h, out_w), np.float16)
        gt_stack[CTX - 1] = gt_ds.astype(np.float16)
        val_stack[CTX - 1] = valid_anchor.astype(np.uint8)
        delta[CTX - 1] = np.where(valid_anchor, gt_ds - raw0_ds, 0).astype(np.float16)
        np.savez(shard_dir / f"{aid}.npz", raw_disp=raw_stack, gt_disp=gt_stack,
                 valid_mask=val_stack, delta_disp_gt_minus_raw=delta)
        np.save(raw_dir / f"{aid}.npy", raws[0])    # anchor raw (full res)

        index_rows.append({"sequence_id": aid, "frame_id": win_stems[0], "frame_index": CTX - 1,
                           "frame_offset": CTX - 1, "target_path": str((shard_dir / f"{aid}.npz").resolve()),
                           "target_h": out_h, "target_w": out_w, "dataset": "d4d",
                           "specimen": spec, "session": sess, "clip": r["clip_id"],
                           "anchor_type": r["anchor_type"], "quality": r["quality_status"],
                           "convention": r["convention"], "split": r["quality_status"]})
        ctx_rows.append({"anchor_id": aid, "context_stems": ";".join(win_stems),
                         "timestamps": ";".join(f"{ts_of(s+'.png'):.3f}" for s in win_stems),
                         "intervals_ms": ";".join(f"{(ts_of(win_stems[i]+'.png')-ts_of(win_stems[i+1]+'.png'))*1e3:.0f}" for i in range(CTX - 1)),
                         "padding": "clamp_start" if len(set(win_idx)) < CTX else "none"})

    def wcsv(path, rr):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rr:
            path.write_text(""); return
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rr[0].keys())); w.writeheader(); w.writerows(rr)

    wcsv(args.out / "d4d_index.csv", index_rows)
    wcsv(args.out / "skipped_anchors.csv", skipped)
    wcsv(args.out / "context_manifest.csv", ctx_rows)
    peak = float(torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else 0.0
    (args.out / "context_build_summary.json").write_text(json.dumps(
        {"evaluated": len(index_rows), "skipped": len(skipped), "unique_s2m2_frames": len(disp_cache),
         "peak_vram_mb": round(peak, 1), "device": str(device)}, indent=2) + "\n")
    print(json.dumps({"evaluated": len(index_rows), "skipped": len(skipped), "peak_vram_mb": round(peak, 1)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
