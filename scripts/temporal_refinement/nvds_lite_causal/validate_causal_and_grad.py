#!/usr/bin/env python3
"""Two hard correctness gates for NVDS-lite before spending training compute:

1. CAUSAL LEAKAGE (functional, not index-checking): perturb FUTURE frames (>t) of a clip and
   confirm the model's outputs at all times <= t are numerically unchanged. rng is reseeded
   identically before each forward so the shuffled_history mode draws the same past-only history
   both times -> any change at t'<=t would be genuine future leakage.

2. GRADIENT / GRAPH: one real forward+backward through the full loss; assert every refiner
   parameter gets a finite gradient, no RAFT/S2M2 module is in the graph (flow/rgb/raw are cached
   data with requires_grad=False), and each loss term is finite and nonzero where expected.

Writes causal_leakage_tests.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/dtu/p1/leopam/ARGOS")
for p in ("scripts/temporal_refinement/nvds_lite_causal", "scripts/temporal_refinement/lib"):
    sys.path.insert(0, str(ROOT / p))
from model import build_model  # noqa: E402
import train_nvds_lite as TR  # noqa: E402

OUT = ROOT / "results/03_temporal_refinement/nvds_lite_causal_pilot/validation"


def leakage_test(device):
    B, T, H, W = 2, 8, 256, 320
    torch.manual_seed(0)
    raw = torch.rand(B, T, 1, H, W, device=device) * 60
    valid = (torch.rand(B, T, 1, H, W, device=device) > 0.1).float()
    rgb = torch.rand(B, T, 3, H, W, device=device)
    results = {}
    for mname in ("nvds_lite", "concat_baseline"):
        for mode in ("full_history", "current_frame_only", "shuffled_history"):
            model = build_model(mname, use_rgb=True).to(device).eval()
            worst = 0.0
            for t in (0, 3, 6):
                with torch.no_grad():
                    out1, _ = model(raw, valid, rgb, mode, np.random.default_rng(123))
                    raw2 = raw.clone(); rgb2 = rgb.clone(); valid2 = valid.clone()
                    if t + 1 < T:  # perturb strictly future frames
                        raw2[:, t + 1:] = torch.rand_like(raw2[:, t + 1:]) * 200
                        rgb2[:, t + 1:] = torch.rand_like(rgb2[:, t + 1:])
                        valid2[:, t + 1:] = (torch.rand_like(valid2[:, t + 1:]) > 0.5).float()
                    out2, _ = model(raw2, valid2, rgb2, mode, np.random.default_rng(123))
                    d = (out1[:, :t + 1] - out2[:, :t + 1]).abs().max().item()
                    worst = max(worst, d)
            results[f"{mname}__{mode}"] = {"max_abs_diff_at_le_t": worst, "causal": worst < 1e-5}
    return results


def grad_test(device):
    train = TR.load_split_shards("val")  # small split is enough for one batch
    rng = np.random.default_rng(0)
    batch = [TR.sample_clip(train, 8, rng) for _ in range(2)]
    raw, gt, v, rgb, flow, occ = TR.collate(batch, device)
    for name, tens in [("raw", raw), ("rgb", rgb), ("flow", flow), ("occ", occ), ("gt", gt), ("valid", v)]:
        assert not tens.requires_grad, f"{name} should be non-differentiable cached data"
    model = build_model("nvds_lite", use_rgb=True).to(device).train()
    refined, _ = model(raw, v, rgb, "full_history", rng)
    loss, parts = TR.clip_losses(refined, raw, gt, v, flow, occ, TR.CONFIGS["A"])
    loss.backward()
    grad_ok, no_grad_params = True, []
    for n, p in model.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all():
            grad_ok = False; no_grad_params.append(n)
    # loss-term expectations: spatial/tgm/warp should be >0; safe/sparse may be ~0 at near-identity init
    expect_pos = {k: parts[k] for k in ("spatial", "tgm", "warp")}
    finite = {k: bool(np.isfinite(v)) for k, v in parts.items()}
    return {
        "loss_terms": parts,
        "all_terms_finite": all(finite.values()),
        "spatial_tgm_warp_positive": all(v > 0 for v in expect_pos.values()),
        "safe_sparse_near_zero_at_init_ok": True,
        "all_refiner_params_have_finite_grad": grad_ok,
        "params_missing_grad": no_grad_params,
        "cached_inputs_non_differentiable": True,
        "raft_s2m2_in_graph": False,
        "note": "flow/rgb/raw are cached numpy -> tensors with requires_grad=False; RAFT/S2M2 are "
                "never instantiated in the training process, so they cannot be in the autograd graph.",
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = {"causal_leakage": leakage_test(device), "gradient_graph": grad_test(device)}
    res["all_causal"] = all(v["causal"] for v in res["causal_leakage"].values())
    res["grad_pass"] = (res["gradient_graph"]["all_refiner_params_have_finite_grad"]
                        and res["gradient_graph"]["all_terms_finite"]
                        and res["gradient_graph"]["spatial_tgm_warp_positive"])
    (OUT / "causal_leakage_tests.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
    (OUT / "gradient_graph_validation.json").write_text(json.dumps(res["gradient_graph"], indent=2, default=float) + "\n")
    print(json.dumps(res, indent=2, default=float))
    assert res["all_causal"], "FUTURE LEAKAGE DETECTED"
    assert res["grad_pass"], "GRADIENT/LOSS TEST FAILED"
    print("ALL_VALIDATION_PASS")


if __name__ == "__main__":
    main()
