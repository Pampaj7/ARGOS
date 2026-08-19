#!/usr/bin/env python3
"""Grid-based temporal change error for the methods we compare against.

The review's objection was that the paper builds a temporal evaluation framework and then
reports none of it for the competitors, which is fair: the head-to-head tables carry EPE,
bad-pixel rates and RMSE only. Nothing prevents the temporal measures -- DTCE needs the
prediction stack, the target and a support, and no pose, flow or extra model -- so the gap
was that the comparison scripts call a flat scorer instead of the framework.

DTCE is |dPrediction - dTarget| over a lag, on a support neither method influences. It is
the column where the asymmetry between us and a bidirectional stabiliser pays: a method
that sees the future should win temporal smoothness, so a causal method that stays close
is the strongest form of the result, and one that loses says something the paper's own
metric section already warns about. Either outcome belongs in the table.

Reported at every lag the framework declares, not only the one the paper quotes, because
the horizon analysis found that DTCE can change sign between lags and a single lag chosen
after the fact would be exactly the reader trap the paper complains about elsewhere.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def dtce(stacks: Mapping[str, np.ndarray], target: np.ndarray, support: np.ndarray,
         lags: tuple[int, ...] = (1, 2, 4, 8)) -> dict:
    """`{method: {lag: {'DTCE_px': v, 'reduction_pct': r}}}` for one sequence.

    `stacks` must contain 'raw'; every other entry is scored as a refinement of it, so the
    reduction is against the same frozen prediction each method was given.
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from model_design.metrics.unified_metrics import MetricConfig, compute_temporal_metrics

    if "raw" not in stacks:
        raise ValueError("the raw stack is the reference every reduction is taken against")
    config = MetricConfig(temporal_horizons=tuple(lags))
    frames = target.shape[0]
    # The framework wants B,T,H,W; one sequence is one batch element.
    def batched(a):
        return np.asarray(a, dtype=np.float64)[None]

    out: dict[str, dict] = {}
    for name, stack in stacks.items():
        if name == "raw":
            continue
        report = compute_temporal_metrics(
            batched(stacks["raw"]), batched(target), batched(support),
            np.ones((1, frames) + target.shape[1:], bool), config,
            unit="px", refined=batched(stack))
        for lag in lags:
            if str(lag) not in report:
                continue
            methods = report[str(lag)]["methods"]
            raw_value = methods["raw"]["DTCE_grid_px"]["MAE"]["value"]
            value = methods["refined"]["DTCE_grid_px"]["MAE"]["value"]
            out.setdefault("raw", {})[lag] = {"DTCE_px": raw_value}
            out.setdefault(name, {})[lag] = {
                "DTCE_px": value,
                "reduction_pct": 100.0 * (raw_value - value) / raw_value if raw_value else None}
    return out


def demo() -> None:
    """A stack that drifts must score worse than one that does not, and raw must be exact."""
    rng = np.random.default_rng(0)
    frames, height, width = 12, 8, 10
    target = np.cumsum(rng.normal(0, 0.3, (frames, height, width)), axis=0) + 20.0
    support = np.ones((frames, height, width), bool)
    noisy = target + rng.normal(0, 0.05, target.shape)
    drifting = target + np.arange(frames)[:, None, None] * 0.4
    got = dtce({"raw": noisy, "perfect": target, "drifting": drifting}, target, support, lags=(1, 2))
    # DTCE is |dPrediction - dTarget|, so a prediction that tracks the target exactly scores
    # zero however wrong it is in absolute terms -- the metric is about change, not accuracy.
    assert got["perfect"][1]["DTCE_px"] == 0.0, "a prediction equal to the target must score zero"
    assert got["drifting"][1]["DTCE_px"] > got["raw"][1]["DTCE_px"], \
        "a linearly drifting prediction must score worse than the noisy raw it replaces"
    assert got["drifting"][2]["DTCE_px"] > got["drifting"][1]["DTCE_px"], \
        "drift must grow with the lag"
    assert got["perfect"][1]["reduction_pct"] == 100.0, "a zero-DTCE method removes all of it"
    print("demo OK:", {k: round(v[1]["DTCE_px"], 4) for k, v in got.items()})


if __name__ == "__main__":
    demo()
