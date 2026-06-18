import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_calibration(path):
    calib = json.loads(path.read_text())
    p1 = np.array(calib["P1"]["data"], dtype=np.float32).reshape(3, 4)
    p2 = np.array(calib["P2"]["data"], dtype=np.float32).reshape(3, 4)
    return {
        "fx": float(p1[0, 0]),
        "fy": float(p1[1, 1]),
        "cx_left": float(p1[0, 2]),
        "cy_left": float(p1[1, 2]),
        "cx_right": float(p2[0, 2]),
        "cy_right": float(p2[1, 2]),
        "baseline_mm": float(abs(p2[0, 3] / p2[0, 0])),
    }


def copy_png(src, dst):
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read {src}")
    cv2.imwrite(str(dst), img)


def convert(root, out_root, reference):
    root = Path(root)
    out_root = Path(out_root)
    count = 0
    for exp in ["Experiment_1", "Experiment_2"]:
        exp_dir = root / exp
        ref_dir = exp_dir / reference
        for left_path in sorted((exp_dir / "Left_rectified").glob("*.png")):
            stem = left_path.stem
            right_path = exp_dir / "Right_rectified" / f"{stem}.png"
            disp_path = ref_dir / "Disparity" / f"{stem}.png"
            depth_path = ref_dir / "DepthL" / f"{stem}.png"
            calib_path = exp_dir / "Rectified_calibration" / f"{stem}.json"
            if not (right_path.exists() and disp_path.exists() and depth_path.exists() and calib_path.exists()):
                continue

            sample_id = f"{exp}_{stem}"
            split = "honest_train" if exp == "Experiment_1" else "honest_test"
            sample_dir = out_root / "servct" / split / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            copy_png(left_path, sample_dir / "left.png")
            copy_png(right_path, sample_dir / "right.png")
            disp = cv2.imread(str(disp_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
            mask = (disp > 0) & (depth > 0)
            np.save(sample_dir / "disp_gt.npy", disp)
            np.save(sample_dir / "depth_gt_mm.npy", depth)
            np.save(sample_dir / "valid_mask.npy", mask.astype(np.uint8))

            left_img = cv2.imread(str(left_path), cv2.IMREAD_UNCHANGED)
            calib = load_calibration(calib_path)
            calib.update({"width": int(left_img.shape[1]), "height": int(left_img.shape[0])})
            (sample_dir / "calib.json").write_text(json.dumps(calib, indent=2))
            metadata = {
                "dataset": "SERV-CT",
                "split": split,
                "sequence": exp,
                "frame": stem,
                "reference_type": reference,
                "left_path_original": str(left_path),
                "right_path_original": str(right_path),
                "has_disparity_gt": True,
                "has_depth_gt": True,
                "units": "depth_mm_disparity_px",
            }
            (sample_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--servct_root", default="../../external/frame_stereo_repos/Fast-FoundationStereo/data/surgical_stereo/servct/SERV-CT")
    parser.add_argument("--out_root", default="../../dataset")
    parser.add_argument("--reference", choices=["Reference_CT", "Reference_RGB"], default="Reference_CT")
    args = parser.parse_args()
    count = convert(args.servct_root, args.out_root, args.reference)
    print(json.dumps({"converted_samples": count, "out_root": args.out_root}, indent=2))


if __name__ == "__main__":
    main()
