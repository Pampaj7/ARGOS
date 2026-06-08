import argparse
import csv
from pathlib import Path


def collect(root, datasets):
    rows = []
    root = Path(root)
    for dataset_id in datasets:
        exp_dir = root / dataset_id
        ref_dir = exp_dir / "Reference_SCARED"
        for left_path in sorted((exp_dir / "Left_rectified").glob("*.png")):
            stem = left_path.stem
            right_path = exp_dir / "Right_rectified" / f"{stem}.png"
            depth_path = ref_dir / "DepthL_float32" / f"{stem}.npy"
            disp_path = ref_dir / "Disparity_float32" / f"{stem}.npy"
            mask_path = ref_dir / "ValidMask" / f"{stem}.npy"
            calib_path = exp_dir / "Rectified_calibration" / f"{stem}.json"
            if all(p.exists() for p in [right_path, depth_path, disp_path, mask_path, calib_path]):
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "frame_id": stem,
                        "left_path": str(left_path),
                        "right_path": str(right_path),
                        "depth_float32_path": str(depth_path),
                        "disparity_float32_path": str(disp_path),
                        "valid_mask_path": str(mask_path),
                        "calibration_path": str(calib_path),
                    }
                )
    return rows


def write(path, rows):
    keys = [
        "dataset_id",
        "frame_id",
        "left_path",
        "right_path",
        "depth_float32_path",
        "disparity_float32_path",
        "valid_mask_path",
        "calibration_path",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="stereo/Fast-FoundationStereo/data/surgical_stereo/scared_keyframes")
    parser.add_argument("--val_csv", default="results/scared_clean_keyframes_val_dataset7.csv")
    parser.add_argument("--test_csv", default="results/scared_clean_keyframes_test_dataset8_9.csv")
    args = parser.parse_args()
    val = collect(args.root, ["dataset_7"])
    test = collect(args.root, ["dataset_8", "dataset_9"])
    Path(args.val_csv).parent.mkdir(parents=True, exist_ok=True)
    write(args.val_csv, val)
    write(args.test_csv, test)
    print({"val": len(val), "test": len(test)})


if __name__ == "__main__":
    main()
