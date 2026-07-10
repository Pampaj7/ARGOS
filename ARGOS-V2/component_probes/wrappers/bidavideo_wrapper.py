"""Compatibility shim; reusable logic lives in model_design.external_components."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model_design.external_components.bidavideo import *  # noqa: F401,F403,E402

