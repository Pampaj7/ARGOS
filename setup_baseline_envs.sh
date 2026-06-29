#!/bin/bash
set -e

echo "=== Setup CODD ==="
conda create --name codd python=3.8 -y
source activate codd || conda activate codd
pip install scipy pyyaml terminaltables natsort
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1121/download.html
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html
pip install mmsegmentation
pip install git+https://github.com/princeton-vl/lietorch.git
echo "CODD setup complete!"

echo "=== Setup TCSM ==="
conda create --name tcsm python=3.10 -y
source activate tcsm || conda activate tcsm
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
pip install cupy_cuda117==10.6.0 imageio==2.31.1 matplotlib==3.7.1 numpy==1.24.3 opencv_python==4.7.0.72 Pillow==9.4.0 psutil==5.9.5 pykitti==0.3.1 scipy==1.10.1 scikit-image==0.21.0 tqdm==4.65.0 wandb==0.15.10
echo "TCSM setup complete!"

echo "=== Setup PPMStereo ==="
conda env create -f external/video_stereo_repos/PPMStereo/environment.yml
echo "PPMStereo setup complete!"

echo "All baseline environments are ready!"
