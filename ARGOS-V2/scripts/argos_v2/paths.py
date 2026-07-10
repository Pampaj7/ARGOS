from pathlib import Path

V2_ROOT = Path(__file__).resolve().parents[2]
ARGOS_ROOT = V2_ROOT.parent

DATASET_DIR = ARGOS_ROOT / "dataset"
FRAME_STEREO_REPOS_DIR = ARGOS_ROOT / "external/frame_stereo_repos"

SCARED_C_SEQ_DIR = DATASET_DIR / "SCARED-C/curated/geometric_gt/corrected_temporal_gt"
QUALITY_GATE_CSV = DATASET_DIR / "SCARED-C/curated/manifests/quality_gate.csv"

CACHE_DIR = V2_ROOT / "cache_scaredc_backbones"
RESULTS_DIR = V2_ROOT / "results"

CACHE_WIDTH = 180
CACHE_HEIGHT = 144
