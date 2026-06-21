import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def get_cmap(disp, vmax):
    norm = disp / vmax
    norm = np.clip(norm, 0, 1)
    cmap = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cmap

def get_diff_cmap(diff, vmax=5.0):
    norm = diff / vmax
    norm = np.clip(norm, 0, 1)
    cmap = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    return cmap

def main():
    rgb_dir = Path("/dtu/p1/leopam/ARGOS/dataset/SCARED/curated/consecutive32/left")
    pred_dir = Path("/dtu/p1/leopam/ARGOS/results/02_video_stereo/stereoanyvideo_temporal_eval/consecutive32/predictions")
    
    s2m2_dir = pred_dir / "S2M2-S_512"
    sav_dir = pred_dir / "StereoAnyVideo_384x640"
    
    # Load frames
    frames = []
    for i in range(32):
        fid = f"{i:06d}"
        rgb = cv2.cvtColor(cv2.imread(str(rgb_dir / f"{i+1:06d}.png")), cv2.COLOR_BGR2RGB)
        s2m2 = np.load(s2m2_dir / f"{fid}.npy")
        sav = np.load(sav_dir / f"{fid}.npy")
        frames.append({'id': fid, 'rgb': rgb, 's2m2': s2m2, 'sav': sav})
        
    # Find sequence with highest flickering in S2M2
    max_diff_idx = 1
    max_diff = 0
    for i in range(1, 28): # leave space for 5 frames
        # compute diff for current frame
        diff = np.mean(np.abs(frames[i]['s2m2'] - frames[i-1]['s2m2']))
        if diff > max_diff:
            max_diff = diff
            max_diff_idx = i
            
    # Actually, let's just pick a window that has visually striking differences
    # We can pre-calculate the sequence of differences and pick the 5-frame window with the highest sum
    diffs = [np.mean(np.abs(frames[i]['s2m2'] - frames[i-1]['s2m2'])) for i in range(1, 32)]
    window_sums = [sum(diffs[i:i+4]) for i in range(len(diffs)-4)]
    best_start = np.argmax(window_sums) + 1 # +1 because diffs is shifted
    
    print(f"Selecting 5-frame sequence starting at {best_start} to {best_start+4}")
    
    selected_frames = frames[best_start:best_start+5]
    prev_frames = frames[best_start-1:best_start+4]
    
    # Render grid
    fig, axes = plt.subplots(5, 5, figsize=(20, 15))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    # VMAX for disparity
    all_disp = np.concatenate([f['s2m2'].flatten() for f in frames] + [f['sav'].flatten() for f in frames])
    vmax_disp = np.percentile(all_disp, 99)
    vmax_diff = 10.0 # 10 pixels of diff
    
    for c in range(5):
        f = selected_frames[c]
        p = prev_frames[c]
        
        # RGB
        ax = axes[0, c]
        ax.imshow(f['rgb'])
        ax.axis('off')
        if c == 0:
            ax.set_title("RGB", fontsize=18, fontweight='bold', loc='left', pad=10)
        ax.text(0.5, 1.05, f"t+{c}", transform=ax.transAxes, ha="center", fontsize=16, fontweight='bold')
        
        # S2M2 Disparity
        ax = axes[1, c]
        ax.imshow(get_cmap(f['s2m2'], vmax_disp)[..., ::-1])
        ax.axis('off')
        if c == 0:
            ax.text(-0.1, 0.5, "S2M2-S\nDisparity", transform=ax.transAxes, va='center', ha='right', fontsize=18, fontweight='bold')
            
        # S2M2 Diff
        ax = axes[2, c]
        s2m2_diff = np.abs(f['s2m2'] - p['s2m2'])
        ax.imshow(get_diff_cmap(s2m2_diff, vmax_diff)[..., ::-1])
        ax.axis('off')
        if c == 0:
            ax.text(-0.1, 0.5, "S2M2-S\nTemporal Diff", transform=ax.transAxes, va='center', ha='right', fontsize=18, fontweight='bold')
            
        # SAV Disparity
        ax = axes[3, c]
        ax.imshow(get_cmap(f['sav'], vmax_disp)[..., ::-1])
        ax.axis('off')
        if c == 0:
            ax.text(-0.1, 0.5, "StereoAnyVideo\nDisparity", transform=ax.transAxes, va='center', ha='right', fontsize=18, fontweight='bold')
            
        # SAV Diff
        ax = axes[4, c]
        sav_diff = np.abs(f['sav'] - p['sav'])
        ax.imshow(get_diff_cmap(sav_diff, vmax_diff)[..., ::-1])
        ax.axis('off')
        if c == 0:
            ax.text(-0.1, 0.5, "StereoAnyVideo\nTemporal Diff", transform=ax.transAxes, va='center', ha='right', fontsize=18, fontweight='bold')

    out_path = Path("/dtu/p1/leopam/ARGOS/presentation/argos_progress/images/flicker_comparison_grid.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), bbox_inches='tight', dpi=150, facecolor='white')
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
