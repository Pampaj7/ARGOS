#!/usr/bin/env python3
"""Evaluate Adaptive Safe-Step Proposal Refiner."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "models"))
sys.path.insert(0, str(SCRIPT_DIR / "eval_scripts"))

import evaluate_counterfactual_proposal_verifier_refiner as evaluator
from adaptive_safe_step_proposal_refiner import adaptive_safe_step_proposal_refiner


if __name__ == "__main__":
    evaluator.counterfactual_proposal_verifier_refiner = adaptive_safe_step_proposal_refiner
    if "--output-root" not in sys.argv:
        sys.argv += ["--output-root", "results/03_temporal_refinement/training/adaptive_safe_step_proposal_refiner"]
    raise SystemExit(evaluator.main())
