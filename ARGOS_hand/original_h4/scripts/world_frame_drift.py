#!/usr/bin/env python3
"""Accumulated reconstruction drift in a common world frame, using SCARED-C's own poses.

`evaluate_3d_consistency.py` measures frame-to-frame surface stability with both clouds of
a pair in their own camera frames, and says so: p2p there mixes true surface change with
rigid camera motion, which the ground-truth floor absorbs. It justified that restriction by
asserting SCARED-C ships no per-frame pose. It does -- correcting them is what the dataset
is for -- so the restriction is lifted here and the camera motion is removed rather than
absorbed.

The convention is not invented. `scripts/scared_c/build_corrected_temporal_gt.py` builds the
ground truth as

    rectified_t = R1 . pose_t . keyframe_point

so a rectified point at frame t returns to the shared keyframe frame through

    keyframe_point = pose_t^-1 . R1^T . rectified_t

which is what this applies. R1 is a rotation, so its inverse is its transpose.

What makes this checkable: SCARED-C's ground truth *is* one keyframe's structured-light
geometry propagated by those poses, so transforming ground truth back into the keyframe
frame must return the same cloud at every frame, to numerical precision. `--self-check`
asserts exactly that on real data before any prediction is scored. If the convention were
wrong -- a transposed rotation, an inverted pose, a millimetre/metre mix -- that assertion
fails, and it fails loudly rather than producing a plausible drift number.

The consequence for reading the output: the ground-truth floor is zero by construction, not
by measurement, so drift here is entirely prediction error. It also means real tissue
deformation is absent from the reference, and a prediction that tracks genuine deformation
is charged for it. This measures rigid-scene reconstruction consistency, which is what a
mapping or servoing stack accumulates; it is not a deformation benchmark.
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARGOS = ROOT.parents[1]
RAW = ARGOS / "dataset/SCARED-C/raw"
GRID_W, GRID_H = 180, 144
MIN_DISP_PX = 0.1


def _paths() -> None:
    for path in (str(ROOT), str(ROOT / "scripts"), str(ARGOS / "ARGOS_FREEZED/src"),
                 str(ARGOS / "ARGOS-V2/scripts"),
                 str(ARGOS / "ARGOS_FREEZED/experiments/02_massive_training/scripts")):
        if path not in sys.path:
            sys.path.insert(0, path)


def sequence_paths(sequence: str) -> tuple[Path, Path]:
    """`dataset_2_keyframe_3` -> its raw keyframe directory and pose archive."""
    dataset, keyframe = sequence.split("_keyframe_")
    kf_dir = RAW / dataset / f"keyframe_{keyframe}"
    archive = kf_dir / "data/frame_data.tar.gz"
    if not archive.is_file():
        raise FileNotFoundError(f"no pose archive for {sequence}: {archive}")
    return kf_dir, archive


def load_poses(sequence: str, frame_ids: list[str]) -> dict[str, np.ndarray]:
    """Per-frame 4x4 camera poses, keyed by the frame ids the evaluation uses.

    The archive names members `frame_data%06d.json` from a 1-based video frame index, which
    is the same numbering the curated `frame_id` strings carry.
    """
    _, archive = sequence_paths(sequence)
    wanted = {int(v) for v in frame_ids}
    poses: dict[str, np.ndarray] = {}
    with tarfile.open(archive) as tf:
        for member in tf:
            if not member.name.endswith(".json"):
                continue
            index = int("".join(c for c in Path(member.name).stem if c.isdigit()))
            if index not in wanted:
                continue
            payload = json.loads(tf.extractfile(member).read())
            poses[f"{index:06d}"] = np.asarray(payload["camera-pose"], dtype=np.float64).reshape(4, 4)
    missing = sorted(set(frame_ids) - set(poses))
    if missing:
        raise RuntimeError(f"{sequence}: {len(missing)} frames have no pose, first {missing[:3]}")
    return poses


def rectification_rotation(sequence: str, image_size: tuple[int, int]) -> np.ndarray:
    """R1 from the same stereoRectify call the ground-truth converter uses."""
    import cv2
    sys.path.insert(0, str(ARGOS))
    from scripts.scared_c.build_corrected_temporal_gt import load_calib
    kf_dir, _ = sequence_paths(sequence)
    calib = load_calib(kf_dir / "endoscope_calibration.yaml")
    h, w = image_size
    r1, _r2, _p1, _p2, _q, _roi1, _roi2 = cv2.stereoRectify(
        calib["M1"], calib["D1"], calib["M2"], calib["D2"], (w, h),
        calib["R"], calib["T"].reshape(3, 1), flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    return np.asarray(r1, dtype=np.float64)


def to_world(xyz: np.ndarray, pose: np.ndarray, r1: np.ndarray) -> np.ndarray:
    """Rectified-camera XYZ [3,H,W] in mm -> shared keyframe frame, inverting the GT path."""
    flat = xyz.reshape(3, -1)
    native = r1.T @ flat                                    # rectified -> frame-t native
    homogeneous = np.vstack([native, np.ones((1, native.shape[1]))])
    world = np.linalg.inv(pose) @ homogeneous               # frame-t native -> keyframe
    return (world[:3] / world[3:4]).reshape(xyz.shape)


def perturb_poses(poses: dict[str, np.ndarray], sigma_mm: float, sigma_deg: float,
                  rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Independent small rigid error on each frame's pose, expressed in its own camera frame.

    Independent rather than smoothly drifting, because a multi-view spread metric is most
    sensitive to uncorrelated jitter: a slow common drift moves every view of a point the
    same way and largely cancels in the spread, so drifting noise would flatter the result.
    This is the pessimistic model.

    The same perturbed set is then used for ground truth, raw and refined alike. That is the
    physical situation -- one wrong pose, three clouds transformed by it -- and it is also
    what makes the test meaningful: the reported quantity is an excess over the ground-truth
    floor, so pose error enters both terms and the question is what survives the difference.
    Drawing independent noise per cloud would measure something that never happens.
    """
    out = {}
    for key, pose in poses.items():
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = rng.normal(0.0, np.deg2rad(sigma_deg))
        cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        rotation = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)
        delta = np.eye(4)
        delta[:3, :3] = rotation
        delta[:3, 3] = rng.normal(0.0, sigma_mm, 3)
        out[key] = pose @ delta
    return out


def backproject(disp: np.ndarray, fx: float, baseline_mm: float, cx: float, cy: float) -> np.ndarray:
    h, w = disp.shape
    z = fx * baseline_mm / np.maximum(disp, MIN_DISP_PX)
    ys, xs = np.meshgrid(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij")
    return np.stack(((xs - cx) * z / fx, (ys - cy) * z / fx, z))


def accumulate_spread(clouds, masks, intrinsics, shape, min_views: int = 3):
    """Disagreement between frames about the same world point, which is what a map fuses.

    Every frame's prediction is projected into the shared keyframe pixel grid and its world
    Z is accumulated there. A predictor that is wrong but *consistently* wrong leaves a tight
    stack and a clean surface; one that jitters leaves a thick one, at the same mean error.
    That difference is invisible to per-frame disparity metrics and is exactly what
    accumulates in a TSDF or a pose graph.

    Welford, because storing every observation of every pixel over a thousand frames is
    gigabytes for a number that needs two accumulators.
    """
    count = np.zeros(shape, dtype=np.int32)
    mean = np.zeros(shape, dtype=np.float64)
    m2 = np.zeros(shape, dtype=np.float64)
    for cloud, mask in zip(clouds, masks):
        points = cloud[:, mask]
        pixel = intrinsics @ points
        u = np.rint(pixel[0] / pixel[2]).astype(int)
        v = np.rint(pixel[1] / pixel[2]).astype(int)
        inside = (u >= 0) & (u < shape[1]) & (v >= 0) & (v < shape[0]) & (points[2] > 0)
        u, v, z = u[inside], v[inside], points[2][inside]
        # Several rays of one frame can land on one keyframe pixel; keep the nearest, which
        # is the surface a z-buffer would keep, rather than averaging through it.
        order = np.argsort(-z)
        flat = v[order] * shape[1] + u[order]
        nearest = np.zeros(shape[0] * shape[1], dtype=np.float64)
        touched = np.zeros(shape[0] * shape[1], dtype=bool)
        nearest[flat] = z[order]
        touched[flat] = True
        idx = np.flatnonzero(touched)
        value = nearest[idx]
        count.flat[idx] += 1
        delta = value - mean.flat[idx]
        mean.flat[idx] += delta / count.flat[idx]
        m2.flat[idx] += delta * (value - mean.flat[idx])
    enough = count >= min_views
    variance = np.zeros(shape, dtype=np.float64)
    variance[enough] = m2[enough] / (count[enough] - 1)
    return np.sqrt(variance[enough]), int(enough.sum())


def self_check(sequence: str, frames: int) -> None:
    """Reproject transformed ground truth into the keyframe and compare against its own
    structured-light point map. Includes the discrimination test the first version lacked.

    The first attempt compared percentiles of the reconstructed extent across frames and
    passed at 1.7mm -- but so did applying no pose at all, and so did inverting the pose the
    wrong way. It had no power to reject anything, which is worse than having no check.

    This one has an absolute reference. The world frame *is* the keyframe camera frame, so a
    correctly transformed point projects through M1 onto the pixel whose depth the keyframe's
    own `left_depth_map.tiff` records. Deliberately wrong conventions are scored alongside
    the intended one and must fail, which is asserted rather than assumed.
    """
    import cv2
    import tifffile
    _paths()
    sys.path.insert(0, str(ARGOS))
    from argos_v2.scared_c_data import load_frame_gt, load_sequence_info
    from scripts.scared_c.build_corrected_temporal_gt import load_calib

    kf_dir, _ = sequence_paths(sequence)
    keyframe_points = tifffile.imread(kf_dir / "left_depth_map.tiff").astype(np.float64)
    intrinsics = load_calib(kf_dir / "endoscope_calibration.yaml")["M1"]
    info = load_sequence_info(sequence)
    ids = [info.frame_ids[i] for i in np.linspace(0, len(info.frame_ids) - 1, frames).astype(int)]
    native, _valid = load_frame_gt(info, ids[0])
    r1 = rectification_rotation(sequence, native.shape)
    poses = load_poses(sequence, ids)

    def median_offset(transform) -> tuple[float, int]:
        errors, count = [], 0
        for frame_id in ids:
            depth, valid = load_frame_gt(info, frame_id)
            height, width = depth.shape
            fx = info.fx * width / native.shape[1]
            xyz = backproject(depth, fx, info.baseline_mm, (width - 1) / 2, (height - 1) / 2)
            world = transform(xyz, poses[frame_id], r1)
            keep = valid & (depth > 0)
            points = world[:, keep]
            pixel = intrinsics @ points
            u = np.rint(pixel[0] / pixel[2]).astype(int)
            v = np.rint(pixel[1] / pixel[2]).astype(int)
            inside = (u >= 0) & (u < keyframe_points.shape[1]) & (v >= 0) & (v < keyframe_points.shape[0])
            reference = keyframe_points[v[inside], u[inside], 2]
            hit = reference > 0
            errors.append(np.abs(points[2][inside][hit] - reference[hit]))
            count += int(hit.sum())
        return float(np.median(np.concatenate(errors))), count

    def wrong_pose(xyz, pose, rotation):
        flat = rotation.T @ xyz.reshape(3, -1)
        moved = pose @ np.vstack([flat, np.ones((1, flat.shape[1]))])
        return (moved[:3] / moved[3:4]).reshape(xyz.shape)

    correct, n = median_offset(to_world)
    print(f"self-check {sequence}, {len(ids)} frames spanning the sequence, {n:,} matched points")
    print(f"  intended   inv(pose) . R1^T : median |dZ| = {correct:8.4f} mm")
    for label, transform in (("pose instead of inv(pose)", wrong_pose),
                             ("no pose at all", lambda xyz, pose, rotation: xyz)):
        value, _ = median_offset(transform)
        print(f"  wrong      {label:22s}: median |dZ| = {value:8.4f} mm")
        assert value > 20 * max(correct, 1e-3), (
            f"the check cannot tell the intended convention from '{label}' "
            f"({correct:.4f} vs {value:.4f} mm) and therefore validates nothing")
    assert correct < 0.5, f"intended convention is off by {correct:.4f} mm"
    print("PASS: the pose direction is validated against an absolute reference, and the "
          "check demonstrably rejects wrong ones")
    # R1 is a 0.33 degree rotation here, so transposing it moves a point by about 1 mm at
    # 100 mm depth and mostly in x and y. This test constrains the pose direction strongly
    # and the rectification direction only weakly; the latter is taken from the ground-truth
    # builder rather than chosen.


def score(sequence: str, backbone: str, frames: int, min_views: int,
          module: str, device_name: str, output: Path | None,
          noise_levels: list[tuple[float, float]] | None = None,
          noise_repeats: int = 5, noise_seed: int = 0) -> None:
    """Ground truth, raw and refined in the shared world frame, all on the module's grid.

    Everything is computed at 144x180 because that is where the module operates and where
    the cached raw predictions live. Scoring raw at native resolution and refined at the
    module's would compare two different resamplings and call the difference a result.
    """
    import cv2
    import tifffile
    import torch
    _paths()
    sys.path.insert(0, str(ARGOS))
    from argos_freezed.alignment.sea_raft_adapter import SEARAFTFlowAdapter
    from argos_v2.cache_io import load_sequence_cache
    from argos_v2.scared_c_data import load_frame_gt, load_sequence_info, read_rgb
    from model_design.comparison.run_comparison import drive, load_factory
    from scripts.scared_c.build_corrected_temporal_gt import load_calib

    kf_dir, _ = sequence_paths(sequence)
    intrinsics = load_calib(kf_dir / "endoscope_calibration.yaml")["M1"]
    keyframe_shape = tifffile.imread(kf_dir / "left_depth_map.tiff").shape[:2]
    info = load_sequence_info(sequence)
    ids = info.frame_ids[:frames] if frames else info.frame_ids
    poses = load_poses(sequence, ids)
    device = torch.device(device_name)
    adapter = load_factory(module)(device=device_name)
    flow_model = SEARAFTFlowAdapter(device=device)

    def rgb(path: Path):
        value = cv2.resize(read_rgb(path), (GRID_W, GRID_H), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(np.ascontiguousarray(value)).permute(2, 0, 1).float().to(device)[None]

    images = [rgb(info.seq_dir / "left" / f"{v}.png") for v in ids]
    right = [rgb(info.seq_dir / "right" / f"{v}.png") for v in ids]

    gt_grid, coverage = [], []
    native_shape = None
    for frame_id in ids:
        native, valid = load_frame_gt(info, frame_id)
        native_shape = native.shape
        cover = cv2.resize(valid.astype(np.float32), (GRID_W, GRID_H), interpolation=cv2.INTER_AREA)
        numerator = cv2.resize(native * valid.astype(np.float32), (GRID_W, GRID_H), interpolation=cv2.INTER_AREA)
        gt_grid.append(numerator / np.maximum(cover, 1e-6) * (GRID_W / native.shape[1]))
        coverage.append(cover)
    gt_grid, coverage = np.stack(gt_grid), np.stack(coverage)
    fx_grid = info.fx * GRID_W / native_shape[1]
    r1 = rectification_rotation(sequence, native_shape)
    cx, cy = (GRID_W - 1) / 2, (GRID_H - 1) / 2

    disparity, validity, cache_ids, _meta = load_sequence_cache(backbone, sequence)
    if [str(v) for v in cache_ids[:len(ids)]] != ids:
        raise RuntimeError(f"frame-ID mismatch: {backbone}/{sequence}")
    count = len(ids)
    frame_dicts = [{"index": i,
                    "raw": torch.from_numpy(np.asarray(disparity[i:i + 1], np.float32))[:, None].to(device),
                    "raw_valid": torch.from_numpy(np.asarray(validity[i:i + 1]) > 0)[:, None].to(device),
                    "rgb": images[i], "right_rgb": right[i]} for i in range(count)]

    def flow_pair(current, past):
        a, b = current["index"], past["index"]
        return flow_model.infer(images[a], images[b]), flow_model.infer(images[b], images[a])

    with torch.inference_mode():
        outputs = dict(drive(adapter, frame_dicts, flow_pair))
        refined = np.stack([outputs[i]["disparity"][0, 0].detach().float().cpu().numpy()
                            for i in range(count)])
    raw = np.asarray(disparity[:count], np.float64)
    raw_valid = np.asarray(validity[:count]) > 0

    # One prediction-independent support for all three, as everywhere else in the framework.
    support = (coverage > 0.5) & raw_valid & (gt_grid > 0)
    record = {"sequence": sequence, "backbone": backbone, "module": module,
              "frames": count, "min_views": min_views, "grid": [GRID_H, GRID_W],
              "metric": "per-keyframe-pixel standard deviation of world-frame Z, mm",
              "results": {}}
    print(f"{sequence} / {backbone} / {module}: {count} frames, >= {min_views} views")
    # Camera-frame geometry does not depend on the pose, so it is built once and re-used by
    # every perturbation draw below; only the world transform and the accumulation repeat.
    camera_frame = {name: [backproject(np.asarray(source[t], np.float64), fx_grid,
                                       info.baseline_mm, cx, cy) for t in range(count)]
                    for name, source in (("gt", gt_grid), ("raw", raw), ("refined", refined))}

    def spreads(pose_set: dict[str, np.ndarray]) -> dict[str, dict]:
        out = {}
        for name, xyzs in camera_frame.items():
            clouds = [to_world(xyzs[t], pose_set[ids[t]], r1) for t in range(count)]
            spread, pixels = accumulate_spread(clouds, list(support), intrinsics,
                                               keyframe_shape, min_views)
            out[name] = {"pixels": pixels, "median_mm": float(np.median(spread)),
                         "mean_mm": float(spread.mean()),
                         "p95_mm": float(np.percentile(spread, 95))}
        return out

    def excess_reduction(values: dict[str, dict]) -> float:
        floor = values["gt"]["median_mm"]
        raw_excess = values["raw"]["median_mm"] - floor
        return 100.0 * (raw_excess - (values["refined"]["median_mm"] - floor)) / raw_excess

    record["results"] = spreads(poses)
    for name, value in record["results"].items():
        print(f"  {name:8s} n={value['pixels']:8,}  median {value['median_mm']:8.4f} mm  "
              f"mean {value['mean_mm']:8.4f}  p95 {value['p95_mm']:8.4f}")
    # Per-frame accuracy on the same support, as a coherence check on the whole result.
    # If refinement were worse frame-by-frame as well, the refined branch would be broken
    # rather than the finding interesting, and the multi-view number would mean nothing.
    per_frame = {}
    for name, source in (("raw", raw), ("refined", refined)):
        error = np.abs(np.asarray(source, np.float64) - gt_grid)[support]
        per_frame[name] = {"disparity_mae_px": float(error.mean()),
                           "disparity_median_px": float(np.median(error))}
    record["per_frame"] = per_frame
    print(f"  per-frame disparity MAE on the same support: raw "
          f"{per_frame['raw']['disparity_mae_px']:.4f} -> refined "
          f"{per_frame['refined']['disparity_mae_px']:.4f} px "
          f"({100 * (per_frame['raw']['disparity_mae_px'] - per_frame['refined']['disparity_mae_px']) / per_frame['raw']['disparity_mae_px']:+.2f}%)")

    gt_floor = record["results"]["gt"]["median_mm"]
    for name in ("raw", "refined"):
        record["results"][name]["excess_over_gt_mm"] = record["results"][name]["median_mm"] - gt_floor
    excess_raw = record["results"]["raw"]["excess_over_gt_mm"]
    excess_ref = record["results"]["refined"]["excess_over_gt_mm"]
    record["excess_reduction_pct"] = 100.0 * (excess_raw - excess_ref) / excess_raw
    print(f"  excess over the ground-truth floor: raw {excess_raw:.4f} -> refined "
          f"{excess_ref:.4f} mm ({record['excess_reduction_pct']:+.1f}%)")

    if noise_levels:
        # Does the sign of this number survive pose error, and at what magnitude does the
        # measurement stop meaning anything? The ground-truth spread already answers half of
        # it empirically -- that cloud is one keyframe's structured light moved by these very
        # poses, so its floor IS the pose residual plus resampling -- and the sweep answers
        # the rest by moving the poses on purpose.
        baseline_sign = np.sign(record["excess_reduction_pct"])
        sweep = []
        for sigma_mm, sigma_deg in noise_levels:
            draws = []
            for repeat in range(noise_repeats):
                rng = np.random.default_rng((noise_seed, repeat, int(sigma_mm * 1e6), int(sigma_deg * 1e6)))
                values = spreads(perturb_poses(poses, sigma_mm, sigma_deg, rng))
                draws.append({"gt_median_mm": values["gt"]["median_mm"],
                              "raw_median_mm": values["raw"]["median_mm"],
                              "refined_median_mm": values["refined"]["median_mm"],
                              "excess_reduction_pct": excess_reduction(values)})
            reductions = [d["excess_reduction_pct"] for d in draws]
            floors = [d["gt_median_mm"] for d in draws]
            entry = {"sigma_translation_mm": sigma_mm, "sigma_rotation_deg": sigma_deg,
                     "repeats": noise_repeats,
                     "excess_reduction_mean_pct": float(np.mean(reductions)),
                     "excess_reduction_sd_pct": float(np.std(reductions, ddof=1)) if len(reductions) > 1 else 0.0,
                     "excess_reduction_min_pct": float(np.min(reductions)),
                     "excess_reduction_max_pct": float(np.max(reductions)),
                     "gt_floor_mean_mm": float(np.mean(floors)),
                     "sign_preserved_in_all_repeats": bool(all(np.sign(r) == baseline_sign for r in reductions)),
                     "draws": draws}
            sweep.append(entry)
            print(f"  pose noise sigma_t={sigma_mm:.3f}mm sigma_r={sigma_deg:.4f}deg: "
                  f"reduction {entry['excess_reduction_mean_pct']:+.1f}% "
                  f"+-{entry['excess_reduction_sd_pct']:.1f} "
                  f"[{entry['excess_reduction_min_pct']:+.1f},{entry['excess_reduction_max_pct']:+.1f}] "
                  f"gt floor {entry['gt_floor_mean_mm']:.4f}mm  "
                  f"sign {'HOLDS' if entry['sign_preserved_in_all_repeats'] else 'FLIPS'}")
        record["pose_noise_sweep"] = sweep
        record["pose_noise_seed"] = noise_seed
        surviving = [e for e in sweep if e["sign_preserved_in_all_repeats"]]
        record["largest_sigma_translation_mm_preserving_sign"] = (
            max(e["sigma_translation_mm"] for e in surviving) if surviving else None)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence", default="dataset_2_keyframe_2")
    parser.add_argument("--backbone", default="RAFT-Stereo")
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--min-views", type=int, default=3)
    parser.add_argument("--module", default="model_design.comparison.ablation_h4:factory_a2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--pose-noise", action="store_true",
                        help="sweep pose perturbation and report whether the sign of the "
                             "excess reduction survives it")
    parser.add_argument("--noise-repeats", type=int, default=5)
    parser.add_argument("--noise-seed", type=int, default=0)
    args = parser.parse_args()
    if args.self_check:
        self_check(args.sequence, min(args.frames, 8))
        return
    # Paired ladder: at the ~100 mm working distance of these sequences a rotation of
    # 0.01 deg displaces a point by ~0.017 mm, so each rung's rotation contributes error of
    # the same order as its translation rather than one term dominating the other.
    levels = [(0.01, 0.006), (0.05, 0.029), (0.10, 0.057), (0.50, 0.286), (1.00, 0.573)]
    score(args.sequence, args.backbone, args.frames, args.min_views, args.module,
          args.device, args.output,
          noise_levels=levels if args.pose_noise else None,
          noise_repeats=args.noise_repeats, noise_seed=args.noise_seed)


if __name__ == "__main__":
    main()
