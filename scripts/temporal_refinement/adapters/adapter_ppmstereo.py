import os
import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description="Adapter for Pick-and-Play Memory Stereo (PPMStereo)")
    parser.add_argument("--data-dir", required=True, help="Directory containing the sequence")
    parser.add_argument("--out-dir", required=True, help="Directory to save predicted depth maps")
    parser.add_argument("--checkpoint", required=True, help="Path to PPMStereo pre-trained weights")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Path to the PPMStereo repository
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../external/video_stereo_repos/PPMStereo_real"))
    
    cmd = [
        "python", os.path.join(repo_dir, "test.py"),  # Modify based on their actual script name
        "--loadckpt", args.checkpoint,
        "--datapath", args.data_dir,
        "--outdir", args.out_dir
    ]
    
    print(f"Running PPMStereo inference: {' '.join(cmd)}")
    # subprocess.run(cmd, check=True, cwd=repo_dir)

if __name__ == "__main__":
    main()
