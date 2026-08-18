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


def score(sequence: str, backbone: str, frames: int, min_views: int) -> None:
    """Ground truth and raw, in the shared world frame. Refined needs the module and a GPU."""
    import cv2
    import tifffile
    _paths()
    sys.path.insert(0, str(ARGOS))
    from argos_v2.cache_io import load_sequence_cache
    from argos_v2.scared_c_data import load_frame_gt, load_sequence_info
    from scripts.scared_c.build_corrected_temporal_gt import load_calib

    kf_dir, _ = sequence_paths(sequence)
    intrinsics = load_calib(kf_dir / "endoscope_calibration.yaml")["M1"]
    keyframe_shape = tifffile.imread(kf_dir / "left_depth_map.tiff").shape[:2]
    info = load_sequence_info(sequence)
    ids = info.frame_ids[:frames] if frames else info.frame_ids
    poses = load_poses(sequence, ids)
    native, _valid = load_frame_gt(info, ids[0])
    r1 = rectification_rotation(sequence, native.shape)

    disparity, valid_mask, cached_ids, _meta = load_sequence_cache(backbone, sequence)
    index = {str(v): i for i, v in enumerate(np.asarray(cached_ids).astype(str))}

    series = {"gt": ([], []), "raw": ([], [])}
    for frame_id in ids:
        depth, valid = load_frame_gt(info, frame_id)
        height, width = depth.shape
        fx = info.fx * width / native.shape[1]
        cx, cy = (width - 1) / 2, (height - 1) / 2
        series["gt"][0].append(to_world(backproject(depth, fx, info.baseline_mm, cx, cy),
                                        poses[frame_id], r1))
        series["gt"][1].append(valid & (depth > 0))

        row = index.get(frame_id)
        if row is None:
            raise RuntimeError(f"{backbone} has no cached prediction for {sequence}/{frame_id}")
        raw = np.asarray(disparity[row], dtype=np.float64)
        raw_valid = np.asarray(valid_mask[row]).astype(bool)
        if raw.shape != depth.shape:
            scale = width / raw.shape[1]
            raw = cv2.resize(raw, (width, height), interpolation=cv2.INTER_NEAREST) * scale
            raw_valid = cv2.resize(raw_valid.astype(np.uint8), (width, height),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
        series["raw"][0].append(to_world(backproject(raw, fx, info.baseline_mm, cx, cy),
                                         poses[frame_id], r1))
        series["raw"][1].append(raw_valid & (raw > 0))

    print(f"{sequence} / {backbone}: {len(ids)} frames, world-frame disagreement per keyframe "
          f"pixel over >= {min_views} views")
    for name, (clouds, masks) in series.items():
        spread, pixels = accumulate_spread(clouds, masks, intrinsics, keyframe_shape, min_views)
        print(f"  {name:6s} n={pixels:8,}  median {np.median(spread):8.3f} mm  "
              f"mean {spread.mean():8.3f}  p95 {np.percentile(spread, 95):8.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence", default="dataset_2_keyframe_2")
    parser.add_argument("--backbone", default="RAFT-Stereo")
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--min-views", type=int, default=3)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check(args.sequence, min(args.frames, 8))
        return
    score(args.sequence, args.backbone, args.frames, args.min_views)


if __name__ == "__main__":
    main()
