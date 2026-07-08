#!/bin/bash
#BSUB -J fs_matrix
#BSUB -q p1
#BSUB -gpu "num=1:mode=shared"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -W 4:00
#BSUB -o logs/fs_matrix_%J.out
#BSUB -e logs/fs_matrix_%J.err
source /dtu/p1/leopam/ARGOS/.miniconda/etc/profile.d/conda.sh
conda activate argos
cd /dtu/p1/leopam/ARGOS
export PYTHONPATH="$(pwd)"
python scripts/temporal_refinement/adaptation/run_d4d_few_shot_matrix.py
