import os
import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description="Adapter for Consistent Online Dynamic Depth (CODD)")
    parser.add_argument("--img-dir", required=True, help="Directory containing left images")
    parser.add_argument("--r-img-dir", required=True, help="Directory containing right images")
    parser.add_argument("--out-dir", required=True, help="Directory to save predicted depth maps")
    parser.add_argument("--checkpoint", required=True, help="Path to CODD pre-trained weights")
    parser.add_argument("--config", required=True, help="Path to CODD config file")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Path to the original CODD repository
    codd_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../external/video_stereo_repos/codd"))
    inference_script = os.path.join(codd_dir, "inference.py")
    
    # Construct the command to run CODD's inference
    # Note: CODD requires an MMCV environment. We use subprocess to call it.
    cmd = [
        "python", inference_script,
        args.config,
        args.checkpoint,
        "--img-dir", args.img_dir,
        "--r-img-dir", args.r_img_dir,
        "--show-dir", args.out_dir,
        "--num-workers", "4",
        "--eval"
    ]
    
    print(f"Running CODD inference: {' '.join(cmd)}")
    # subprocess.run(cmd, check=True, cwd=codd_dir)
    print("WARNING: Make sure you run this within the CODD mmcv conda environment!")

if __name__ == "__main__":
    main()
