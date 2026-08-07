import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_checkpoint_hashes():
    expected = "40526a32ef6e9a62a3ea2b59e6751a60c441b8190f9b96522e3b12b35895d5cd"
    source = "/dtu/p1/leopam/ARGOS/ARGOS-V2/results/raw_multi_anchor_temporal_refiner/soft_fusion/checkpoints/best_validation.pt"
    assert digest(source) == expected == digest(ROOT / "checkpoints/raw_multi_anchor_best_validation.pt")
    external = json.loads((ROOT / "external_checkpoints/MANIFEST.json").read_text())["checkpoints"][0]
    assert digest(external["path"]) == external["sha256"] == "1a21575ed6ca2c6945fb8e25c4169d241cf59ee5d12b8802c01c965206268cac"
    state = __import__("torch").load(ROOT / "checkpoints/raw_multi_anchor_best_validation.pt", map_location="cpu", weights_only=False)
    assert state["ages"] == [1, 2, 4, 8]
