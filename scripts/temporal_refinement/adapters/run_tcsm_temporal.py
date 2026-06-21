import os
import sys
import torch
import numpy as np
import argparse
import pandas as pd
import cv2
import time
import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    return parser.parse_args()

def main():
    args = get_args()
    seq_dir = os.path.abspath(args.sequence_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    # Load metadata
    meta = pd.read_csv(os.path.join(seq_dir, "metadata.csv"))
    
    # TC-Stereo setup
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../external/video_stereo_repos/Temporally-Consistent-Stereo-Matching"))
    sys.path.insert(0, repo_dir)
    sys.path.insert(0, os.path.join(repo_dir, "core"))
    from tc_stereo import TCStereo
    from utils.utils import InputPadder
    
    model_args = type('Args', (), {
        'hidden_dims': [128]*3,
        'shared_backbone': True,
        'corr_levels': 4,
        'corr_radius': 4,
        'n_downsample': 2,
        'context_norm': "none",
        'slow_fast_gru': False,
        'n_gru_layers': 3,
        'temporal': True,
        'mixed_precision': False,
        'init_thres': 0.5,
    })()
    model = TCStereo(model_args)
    state = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(state['model'], strict=True)
    model = model.cuda().eval()
    
    params = {}
    previous_T = None
    flow_q = None
    net_list = None
    fmap1 = None
    baseline = torch.tensor(0.54).float().cuda()[None] # default kitti baseline?
    
    runtimes = []
    
    argos_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    
    with torch.no_grad():
        for i, row in meta.iterrows():
            left_path = os.path.join(argos_root, row['left_path'])
            right_path = os.path.join(argos_root, row['right_path'])
            calib_path = os.path.join(argos_root, row['calibration_path'])
            
            img1 = torch.from_numpy(cv2.imread(left_path)).permute(2, 0, 1).unsqueeze(0).cuda().float()
            img2 = torch.from_numpy(cv2.imread(right_path)).permute(2, 0, 1).unsqueeze(0).cuda().float()
            
            K_raw = torch.eye(3, dtype=torch.float32, device='cuda')[None]
            
            padder = InputPadder(img1.shape, divis_by=32)
            imgs, K = padder.pad(img1, img2, K=K_raw)
            img1, img2 = imgs
            
            pose = torch.eye(4, dtype=torch.float32, device='cuda')[None]
            
            params = {
                'K': K,
                'T': pose,
                'previous_T': previous_T,
                'last_disp': flow_q,
                'last_net_list': net_list,
                'fmap1': fmap1,
                'baseline': baseline
            }
            
            torch.cuda.synchronize()
            t0 = time.time()
            testing_output = model(img1, img2, iters=32, test_mode=True, params=params if (flow_q is not None) else None)
            torch.cuda.synchronize()
            runtimes.append((time.time() - t0)*1000)
            
            disp_pr = -testing_output['flow']
            flow_q = testing_output['flow_q']
            net_list = testing_output['net_list']
            fmap1 = testing_output['fmap1']
            previous_T = pose
            
            disp_pr, K = padder.unpad(disp_pr, K=K)
            disp_pr = disp_pr.squeeze(0).squeeze(0).cpu().numpy()
            
            np.save(os.path.join(out_dir, f"{row['frame_id']}.npy"), disp_pr)

    metadata = {
        "method": "TC-Stereo-IdentityPose",
        "kind": "video_stereo",
        "frames": len(meta),
        "avg_runtime_ms": float(np.mean(runtimes)) if runtimes else 0,
        "notes": "Ran temporally using Identity poses (due to lack of robot kinematics in SCARED)."
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == '__main__':
    main()
