#!/usr/bin/env python3
"""Regression test for the GT-resize bug found while validating the ARGOS-V2 pilot cache.

Synthetic setup: a native-resolution disparity field that is exactly 50px inside a circular
"valid" region and 0 (invalid) outside it — mimicking SCARED-C's own invalid=0-fill convention
— plus a "prediction" that matches the true disparity almost exactly everywhere inside the
valid region. A correct evaluation pipeline must report a tiny error near 0px. The old naive
resize (nearest-neighbor validity mask over an INTER_AREA-blended disparity field) is expected
to report a large, wrong error instead, because cache cells straddling the valid/invalid
boundary get marked "valid" (single nearest source pixel) while their disparity value is
blended with neighboring 0s.

Run: python3 scripts/test_gt_resize_regression.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from argos_v2.metrics import resize_gt_to_cache_naive, resize_gt_to_cache_corrected

NATIVE_H, NATIVE_W = 1024, 1280
TRUE_DISP = 120.0  # SCARED-C native disparities range up to ~200px; a small constant like 50
                    # understates how much a 0-filled invalid neighbor can drag a blended average


def make_synthetic():
    """SCARED-C's real GT coverage is speckled (many small valid/invalid blobs), not one
    smooth blob — that's what gives the naive resize so much boundary to contaminate.
    Reproduce that here via blurred-noise thresholding instead of a single circle.
    """
    import cv2
    rng = np.random.default_rng(0)
    noise = rng.normal(size=(NATIVE_H, NATIVE_W)).astype(np.float32)
    smoothed = cv2.GaussianBlur(noise, (0, 0), sigmaX=9)  # small blob scale -> lots of boundary per area
    valid = smoothed > np.median(smoothed)  # ~50% coverage, speckled, blob scale ~ a few tens of px

    gt_disp = np.where(valid, TRUE_DISP, 0.0).astype(np.float32)
    pred_disp = np.where(valid, TRUE_DISP + rng.normal(0, 0.05, size=gt_disp.shape), TRUE_DISP).astype(np.float32)
    return gt_disp, valid, pred_disp


def main() -> int:
    gt_disp, gt_valid, pred_native = make_synthetic()

    # native-resolution direct comparison: confirms the synthetic data itself is sane
    native_mask = gt_valid
    native_err = np.abs(pred_native[native_mask] - gt_disp[native_mask]).mean()
    assert native_err < 0.2, f"native-res synthetic error should be ~0.05px, got {native_err:.3f}px"

    # resize the prediction the same way the real cache builder does (already-validated logic)
    from argos_v2.cache_io import resize_pred_to_cache
    pred_cache, pred_valid_cache, _fp32 = resize_pred_to_cache(pred_native, native_w=NATIVE_W)

    naive_gt_small, naive_valid_small = resize_gt_to_cache_naive(gt_disp, gt_valid, NATIVE_W)
    naive_mask = naive_valid_small & (pred_valid_cache > 0)
    naive_err = np.abs(pred_cache.astype(np.float32)[naive_mask] - naive_gt_small[naive_mask]).mean()

    fixed_gt_small, fixed_valid_small = resize_gt_to_cache_corrected(gt_disp, gt_valid, NATIVE_W)
    fixed_mask = fixed_valid_small & (pred_valid_cache > 0)
    fixed_err = np.abs(pred_cache.astype(np.float32)[fixed_mask] - fixed_gt_small[fixed_mask]).mean()

    print(f"native-res error (sanity):     {native_err:.4f}px")
    print(f"naive resize error (broken):   {naive_err:.4f}px  (n={naive_mask.sum()} cache px)")
    print(f"corrected resize error (fixed):{fixed_err:.4f}px  (n={fixed_mask.sum()} cache px)")

    assert naive_err > 1.0, (
        f"expected the OLD naive resize to inflate error strongly (>1px, vs ~0.05px injected "
        f"prediction noise) on this synthetic boundary case, got {naive_err:.4f}px — if this "
        f"fails, the bug may have regressed back to 'fixed' by accident and this test's "
        f"premise needs re-checking"
    )
    assert fixed_err < 0.2, f"expected the CORRECTED resize to stay plausible (<0.2px), got {fixed_err:.4f}px"
    assert naive_err > fixed_err * 10, (
        f"corrected resize should be far more accurate than the naive one "
        f"(naive={naive_err:.4f}px, fixed={fixed_err:.4f}px, ratio={naive_err / fixed_err:.1f}x)"
    )

    print("PASS: naive resize reproduces the inflation bug; corrected resize stays plausible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
