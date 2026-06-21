import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_disp(path):
    if path.endswith('.npy'):
        return np.load(path)
    elif path.endswith('.npz'):
        data = np.load(path)
        keys = list(data.keys())
        if 'disparity' in keys: return data['disparity']
        return data[keys[0]]
    elif path.endswith('.tiff') or path.endswith('.tif'):
        import tifffile
        return tifffile.imread(path)
    else:
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)

def colorize(disp, vmax=None):
    if vmax is None:
        vmax = np.percentile(disp[disp > 0], 95) if np.sum(disp > 0) > 0 else 1.0
    vmax = max(vmax, 1e-5)
    disp_norm = np.clip(disp / vmax, 0, 1)
    cmap = plt.get_cmap('magma')
    color = (cmap(disp_norm)[..., :3] * 255).astype(np.uint8)
    color[disp <= 0] = 0
    return color

def main():
    # Directories
    s2m2_dir = '/dtu/p1/leopam/ARGOS/results/02_video_stereo/all_methods_fair_eval/predictions/S2M2-L_736'
    sav_dir = '/dtu/p1/leopam/ARGOS/results/02_video_stereo/all_methods_fair_eval/predictions/StereoAnyVideo_384x640'
    gt_dir = '/dtu/p1/leopam/ARGOS/dataset/SCARED/curated/temporal_gt/test_dataset_9_keyframe_3/gt/Disparity_float32'
    # TODO: Add refined model predictions directory here when available
    # refined_dir = '/dtu/p1/leopam/ARGOS/results/02_video_stereo/all_methods_fair_eval/predictions/CODD'
    
    out_path = '/dtu/p1/leopam/ARGOS/presentation/argos_progress/videos/comparison_disparity.mp4'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    s2m2_files = sorted(glob.glob(os.path.join(s2m2_dir, '*.npy')))
    sav_files = sorted(glob.glob(os.path.join(sav_dir, '*.npy')))
    
    # GT might be TIFF or NPY
    gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.tiff')) + glob.glob(os.path.join(gt_dir, '*.tif')) + glob.glob(os.path.join(gt_dir, '*.npy')))

    num_frames = min(len(s2m2_files), len(sav_files))
    if num_frames == 0:
        print("No frames found to render!")
        return

    print(f"Rendering {num_frames} frames to {out_path}...")
    writer = None
    vmax = 100.0 # Standard disparity max

    for i in tqdm(range(num_frames)):
        s2m2 = load_disp(s2m2_files[i])
        sav = load_disp(sav_files[i])
        
        if i < len(gt_files):
            gt = load_disp(gt_files[i])
        else:
            gt = np.zeros_like(s2m2)
            
        c_s2m2 = colorize(s2m2, vmax)
        c_sav = colorize(sav, vmax)
        c_gt = colorize(gt, vmax)
        
        cv2.putText(c_s2m2, 'S2M2-L Raw', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(c_sav, 'StereoAnyVideo', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(c_gt, 'Ground Truth', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Concatenate horizontally
        frame = np.concatenate([c_s2m2, c_sav, c_gt], axis=1)
        
        if writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_path, fourcc, 10.0, (w, h))
            
        # Convert RGB to BGR for OpenCV
        writer.write(frame[..., ::-1])
        
    if writer is not None:
        writer.release()
    print(f"Done! Saved video to {out_path}")

if __name__ == '__main__':
    main()
