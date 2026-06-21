import os
import sys
import torch
import numpy as np
import argparse
import pandas as pd
import cv2
import time
import torch.nn.functional as F
import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--chunk-size", type=int, default=20)
    return parser.parse_args()

def main():
    args = get_args()
    seq_dir = os.path.abspath(args.sequence_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    # Load metadata
    meta = pd.read_csv(os.path.join(seq_dir, "metadata.csv"))
    
    # PPMStereo setup
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../external/video_stereo_repos/PPMStereo"))
    sys.path.insert(0, repo_dir)
    
    # Mock PyTorch3D to avoid massive compilation dependencies!
    import types
    class DummyConfigurable:
        pass
    def dummy_get_default_args(cls):
        return {}
    def dummy_get_origin(x):
        return getattr(x, "__origin__", None)
    def dummy_get_args(x):
        return getattr(x, "__args__", ())
        
    pytorch3d = types.ModuleType("pytorch3d")
    implicitron = types.ModuleType("pytorch3d.implicitron")
    tools = types.ModuleType("pytorch3d.implicitron.tools")
    config = types.ModuleType("pytorch3d.implicitron.tools.config")
    config.Configurable = DummyConfigurable
    config.get_default_args = dummy_get_default_args
    config.get_default_args_field = lambda x: {}
    
    pytorch3d.implicitron = implicitron
    implicitron.tools = tools
    tools.config = config
    
    sys.modules["pytorch3d"] = pytorch3d
    sys.modules["pytorch3d.implicitron"] = implicitron
    sys.modules["pytorch3d.implicitron.tools"] = tools
    sys.modules["pytorch3d.implicitron.tools.config"] = config
    
    common = types.ModuleType("pytorch3d.common")
    datatypes = types.ModuleType("pytorch3d.common.datatypes")
    datatypes.get_args = dummy_get_args
    datatypes.get_origin = dummy_get_origin
    common.datatypes = datatypes
    pytorch3d.common = common
    sys.modules["pytorch3d.common"] = common
    sys.modules["pytorch3d.common.datatypes"] = datatypes

    from models.core.ppmstereo import PPMStereo
    
    model = PPMStereo(
        mixed_precision=False, # We usually evaluate in fp32
        num_frames=5,
        attention_type="self_stereo_temporal_update_time_update_space",
        use_3d_update_block=True,
        different_update_blocks=True,
    )
    
    state = torch.load(args.checkpoint, map_location='cpu')
    if "model" in state:
        state = state["model"]
    elif "state_dict" in state:
        state = state["state_dict"]
        
    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[7:]
        if k.startswith("model."):
            k = k[6:]
        new_state[k] = v
        
    model.load_state_dict(new_state, strict=False)
    model = model.cuda().eval()
    
    runtimes = []
    resize_hw = (384, 640)
    
    argos_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    
    with torch.no_grad():
        frames_list = meta.to_dict('records')
        num_frames = len(frames_list)
        chunk_size = args.chunk_size
        
        for start_idx in range(0, num_frames, chunk_size):
            end_idx = min(start_idx + chunk_size, num_frames)
            chunk_frames = frames_list[start_idx:end_idx]
            
            lefts = []
            rights = []
            orig_hw = None
            
            for row in chunk_frames:
                left_path = os.path.join(argos_root, row['left_path'])
                right_path = os.path.join(argos_root, row['right_path'])
                
                img1 = cv2.imread(left_path)
                img2 = cv2.imread(right_path)
                orig_hw = img1.shape[:2]
                
                img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
                img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
                
                img1 = F.interpolate(img1[None], size=resize_hw, mode="bilinear", align_corners=True)[0]
                img2 = F.interpolate(img2[None], size=resize_hw, mode="bilinear", align_corners=True)[0]
                
                lefts.append(img1.cuda())
                rights.append(img2.cuda())
                
            video_left = torch.stack(lefts, dim=0).unsqueeze(0)  # Add B=1
            video_right = torch.stack(rights, dim=0).unsqueeze(0) # Add B=1
            # Wait, demo.py did not add B=1, but models usually expect B=1.
            # Let's check if model fails with B=1 or expects it.
            # actually let's just pass what demo.py did
            video_left_demo = torch.stack(lefts, dim=0)
            video_right_demo = torch.stack(rights, dim=0)
            
            batch_dict = {"stereo_video": torch.stack([video_left_demo, video_right_demo], dim=1)}
            
            torch.cuda.synchronize()
            t0 = time.time()
            predictions = model.forward_batch_test(batch_dict, kernel_size=20, iters=20)
            torch.cuda.synchronize()
            runtimes.append((time.time() - t0)*1000 / len(chunk_frames))
            
            disparities = predictions["disparity"][:, :1].clone().data.abs() # shape: [T, 1, H, W] or [B, T, ...]
            # Remove B dim if it was returned, but usually it matches input. Let's just reshape to [T, 1, H, W]
            disparities = disparities.view(len(chunk_frames), 1, resize_hw[0], resize_hw[1])
            
            scale_w = orig_hw[1] / resize_hw[1]
            
            for i, row in enumerate(chunk_frames):
                disp_pr = disparities[i, 0].cpu().numpy()
                disp_pr = cv2.resize(disp_pr, (orig_hw[1], orig_hw[0]), interpolation=cv2.INTER_LINEAR) * scale_w
                np.save(os.path.join(out_dir, f"{row['frame_id']}.npy"), disp_pr)

    metadata = {
        "method": "PPMStereo",
        "kind": "video_stereo",
        "frames": len(meta),
        "avg_runtime_ms": float(np.mean(runtimes)) if runtimes else 0,
        "notes": f"Ran temporally in chunks of {args.chunk_size}."
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == '__main__':
    main()
