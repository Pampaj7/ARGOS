#!/usr/bin/env python3
"""Download SCARED-C (Han et al., arXiv:2605.16628) from HuggingFace into dataset/SCARED-C/raw/.

SCARED-C keeps the same on-disk layout as SCARED but replaces data/frame_data.tar.gz with
COLMAP-corrected per-frame poses + reprojected depth, covering datasets 1, 2, 3, 6, 7
(datasets 4/5 excluded upstream for calibration issues, same as vanilla SCARED).

Deliberately skips two file kinds per keyframe to avoid ~95GB of redundant download:
- data/scene_points.tar.gz: the OLD kinematics-propagated points, superseded by
  data/frame_data.tar.gz in this release; not needed since we never use kinematics GT.
- data/rgb_frames.tar.gz: pre-extracted RGB frames; we extract the same frames ourselves
  from data/rgb.mp4 (using frame_log.json to pick co-registered frames), matching the
  SCARED temporal_sequences convention already used elsewhere in ARGOS.
Also skips dataset_6/keyframe_1/data/old_tars/ (leftover artifact on the HF repo, not
referenced by the dataset card).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.argos_paths import DATASET_DIR

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "juseonghan/SCARED-C"
OUT = DATASET_DIR / "SCARED-C/raw"
SKIP_BASENAMES = {"scene_points.tar.gz", "rgb_frames.tar.gz"}


def should_skip(rel_path: str) -> bool:
    if "old_tars" in rel_path:
        return True
    return rel_path.rsplit("/", 1)[-1] in SKIP_BASENAMES


def main() -> int:
    api = HfApi()
    info = api.dataset_info(REPO_ID, files_metadata=True)
    files = [f.rfilename for f in info.siblings if not should_skip(f.rfilename)]
    total = len(files)
    print(f"downloading {total} files (skipping scene_points.tar.gz, rgb_frames.tar.gz, old_tars/)")

    OUT.mkdir(parents=True, exist_ok=True)
    for i, rel_path in enumerate(files, 1):
        hf_hub_download(
            repo_id=REPO_ID,
            filename=rel_path,
            repo_type="dataset",
            local_dir=str(OUT),
        )
        print(f"[{i}/{total}] {rel_path}")

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
