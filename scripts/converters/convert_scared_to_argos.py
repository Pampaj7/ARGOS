import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scared_root", default="../../external/frame_stereo_repos/Fast-FoundationStereo/data/surgical_stereo/scared")
    parser.add_argument("--out_root", default="../../dataset")
    args = parser.parse_args()

    scared_root = Path(args.scared_root)
    available_archives = sorted(p.name for p in scared_root.glob("*.zip"))
    status = {
        "status": "pending_full_dataset_layout_inspection",
        "scared_root": str(scared_root),
        "out_root": args.out_root,
        "available_archives": available_archives,
        "note": "Implement after full SCARED download/extraction confirms image, depth, pose, and calibration layout.",
    }
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
