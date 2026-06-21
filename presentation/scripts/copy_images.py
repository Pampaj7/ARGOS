import os
import glob
import shutil

def main():
    src_dirs = [
        '/dtu/p1/leopam/ARGOS/results/03_temporal_refinement/training/adaptive_motion_fusion/exp_200ep_2gpu/reference_images',
        '/dtu/p1/leopam/ARGOS/results/03_temporal_refinement/playground/gt_short_race_v1/reference_images'
    ]
    dst_dir = '/dtu/p1/leopam/ARGOS/presentation/argos_progress/images'

    os.makedirs(dst_dir, exist_ok=True)
    count = 0
    for src_dir in src_dirs:
        png_files = glob.glob(os.path.join(src_dir, '*.png'))
        for f in png_files:
            shutil.copy(f, dst_dir)
            count += 1
    print(f"Copied {count} files to {dst_dir}")

if __name__ == '__main__':
    main()
