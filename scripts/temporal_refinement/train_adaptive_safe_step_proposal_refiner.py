#!/usr/bin/env python3
"""Train Adaptive Safe-Step Proposal Refiner.

Ponytail wrapper: reuse the CPV trainer/loss/dataloaders, swap only the model factory.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "models"))

import train_counterfactual_proposal_verifier_refiner as trainer
from adaptive_safe_step_proposal_refiner import adaptive_safe_step_proposal_refiner


DEFAULT_OUTPUT = Path("results/03_temporal_refinement/training/adaptive_safe_step_proposal_refiner")


def output_root_from_argv() -> Path:
    if "--output-root" in sys.argv:
        return Path(sys.argv[sys.argv.index("--output-root") + 1])
    return DEFAULT_OUTPUT


def main() -> int:
    trainer.counterfactual_proposal_verifier_refiner = adaptive_safe_step_proposal_refiner
    trainer.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    code = trainer.main()
    out = output_root_from_argv()
    if code == 0:
        (out / "architecture_design.md").write_text(
            "# Adaptive Safe-Step Proposal Refiner\n\n"
            "Mechanism: MPC proposal generation is preserved, but the final correction is decomposed as "
            "`small_residual + alpha_safe * large_residual`. The safe-step head predicts a spatial alpha "
            "and new-Bad3 risk from counterfactual proposal context. Training reuses the CPV counterfactual "
            "loss and staged freeze/unfreeze schedule.\n"
        )
        if (out / "README.md").exists():
            txt = (out / "README.md").read_text()
            (out / "README.md").write_text(txt.replace("Counterfactual Proposal Verifier Refiner", "Adaptive Safe-Step Proposal Refiner"))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
