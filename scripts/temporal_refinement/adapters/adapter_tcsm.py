import os
import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description="Adapter for Temporally Consistent Stereo Matching (TCSM)")
    parser.add_argument("--data-dir", required=True, help="Directory containing the sequence")
    parser.add_argument("--out-dir", required=True, help="Directory to save predicted depth maps")
    parser.add_argument("--checkpoint", required=True, help="Path to TCSM pre-trained weights")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Path to the TCSM repository
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../external/video_stereo_repos/Temporally-Consistent-Stereo-Matching"))
    inference_script = os.path.join(repo_dir, "evaluate_stereo.py")
    
    # Example command based on their test script format
    cmd = [
        "python", inference_script,
        "--restore_ckpt", args.checkpoint,
        "--dataset", "custom",
        "--datapath", args.data_dir,
        "--output_dir", args.out_dir
    ]
    
    print(f"Running TCSM inference: {' '.join(cmd)}")
    # subprocess.run(cmd, check=True, cwd=repo_dir)

if __name__ == "__main__":
    main()
