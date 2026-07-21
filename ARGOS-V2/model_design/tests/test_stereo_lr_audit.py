import numpy as np

from scripts.run_stereo_lr_consistency_audit import auroc


def test_lrc_audit_auroc_is_one_for_perfect_ordering_and_half_for_ties():
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([False, False, True, True])) == 1.0
    assert auroc(np.ones(4), np.array([False, True, False, True])) == 0.5


def test_lrc_audit_auroc_requires_both_classes():
    assert auroc(np.array([.1, .2]), np.array([True, True])) is None
    assert auroc(np.array([.1, .2]), np.array([False, False])) is None
