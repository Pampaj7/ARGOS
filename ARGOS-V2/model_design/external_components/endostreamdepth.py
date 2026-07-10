"""EndoStreamDepth temporal-block/loss wrapper for component probes.

Original repository: external/EndoStreamDepth
Commit: 5abe89d9c0e09f64fdc5276d21bb5a34aa815cc6
Original source paths:
- endostreamdepth/mamba.py: MambaModel, InferenceParams, forward_single_frame
- endostreamdepth/xlstm_block.py: xLSTMModel, start_new_sequence, forward_single_frame
- endostreamdepth/model.py: dpt_features_to_mamba integration
- endostreamdepth/util/loss.py: temporal_consistency_loss, GradientEdgeLoss

Status:
- actual temporal blocks are DPT/xLSTM/Mamba dependency-coupled;
- the actual temporal loss is importable and executable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
ENDO_ROOT = ROOT / "external/EndoStreamDepth"


def _with_endo_path():
    sys.path.insert(0, str(ENDO_ROOT))


def _drop_endo_path():
    try:
        sys.path.remove(str(ENDO_ROOT))
    except ValueError:
        pass


def import_status() -> list[dict]:
    rows: list[dict] = []
    _with_endo_path()
    try:
        try:
            from endostreamdepth.mamba import MambaModel  # type: ignore
            rows.append({"component": "MambaModel import", "status": "pass", "reason": ""})
            try:
                _ = MambaModel(dpt_dim=64, mamba_type="add", num_mamba_layers=1, batch_size=1)
                rows.append({"component": "MambaModel instantiate", "status": "pass", "reason": ""})
            except Exception as exc:
                rows.append({"component": "MambaModel instantiate", "status": "blocked", "reason": f"{type(exc).__name__}: {exc}"})
        except Exception as exc:
            rows.append({"component": "MambaModel import", "status": "blocked", "reason": f"{type(exc).__name__}: {exc}"})
        try:
            from endostreamdepth.xlstm_block import xLSTMModel  # type: ignore  # noqa: F401
            rows.append({"component": "xLSTMModel import", "status": "pass", "reason": ""})
        except Exception as exc:
            rows.append({"component": "xLSTMModel import", "status": "blocked", "reason": f"{type(exc).__name__}: {exc}"})
        try:
            from endostreamdepth.util.loss import temporal_consistency_loss  # type: ignore  # noqa: F401
            rows.append({"component": "temporal_consistency_loss import", "status": "pass", "reason": ""})
        except Exception as exc:
            rows.append({"component": "temporal_consistency_loss import", "status": "blocked", "reason": f"{type(exc).__name__}: {exc}"})
    finally:
        _drop_endo_path()
    return rows


def temporal_consistency_loss_actual(pred_temporal: torch.Tensor, valid_temporal: torch.Tensor) -> torch.Tensor:
    _with_endo_path()
    try:
        from endostreamdepth.util.loss import temporal_consistency_loss  # type: ignore
        return temporal_consistency_loss(pred_temporal, valid_temporal)
    finally:
        _drop_endo_path()


if __name__ == "__main__":
    for row in import_status():
        print(row)

