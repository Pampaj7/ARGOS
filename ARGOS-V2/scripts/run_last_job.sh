#!/bin/bash
cd /dtu/p1/leopam/ARGOS/ARGOS-V2
source /dtu/p1/leopam/ARGOS/.miniconda/etc/profile.d/conda.sh && conda activate argos
ESUB_BYPASS=1 ESUB_QUIET=1 bsub -q p1i -app h100app -gpu "num=1:mode=shared" -n 4 -I \
  python3 scripts/run_backbone_cache.py --backbone StereoAnywhere --sequence dataset_7_keyframe_4 --force \
  > /dtu/p1/leopam/ARGOS/ARGOS-V2/results/full_run/last_job.log 2>&1
echo "EXIT=$?" >> /dtu/p1/leopam/ARGOS/ARGOS-V2/results/full_run/last_job.log
