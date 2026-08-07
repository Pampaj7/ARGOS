"""Frozen ARGOS v2 geometry-v1 constants."""
from pathlib import Path

PROJECT_NAME = "ARGOS v2"
VERSION = "geometry_v1"
ANCHOR_AGES = (1, 2, 4, 8)
FEATURE_CHANNELS = 17
PARAMETER_COUNT = 60_739
PROBABILITY_THRESHOLD = 0.9
UTILITY_THRESHOLD_PX = 0.1
FB_ABSOLUTE_THRESHOLD = 0.5
FB_RELATIVE_THRESHOLD = 0.01
VALID_SAMPLE_THRESHOLD = 0.999
CHECKPOINT_SHA256 = "40526a32ef6e9a62a3ea2b59e6751a60c441b8190f9b96522e3b12b35895d5cd"
SEA_RAFT_SHA256 = "1a21575ed6ca2c6945fb8e25c4169d241cf59ee5d12b8802c01c965206268cac"
H4_SHA256 = "99c5745c164fd4903b8aa8acf8f57efccacd9cdcd0d4ed4305cd10609324d725"
ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "checkpoints/raw_multi_anchor_best_validation.pt"
SEA_RAFT_CHECKPOINT = Path("/dtu/p1/leopam/ARGOS/external/bidavideo/third_party/SEA-RAFT/models/Tartan-C-T-TSKH-spring540x960-S.pth")
EXTERNAL_ROOT = Path("/dtu/p1/leopam/ARGOS/external")
