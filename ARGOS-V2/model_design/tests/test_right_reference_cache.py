import numpy as np

from scripts.run_backbone_cache import cache_key, right_reference_flip_swap


def test_right_reference_namespace_never_aliases_canonical_cache():
    assert cache_key("S2M2-S", smoke=False, right_reference=False) == "S2M2-S"
    assert cache_key("S2M2-S", smoke=False, right_reference=True) == "_rightref_S2M2-S"
    assert cache_key("S2M2-S", smoke=True, right_reference=True) == "_smoke__rightref_S2M2-S"


def test_right_reference_flip_swap_is_exact_and_contiguous():
    left = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
    right = (left + 70).astype(np.uint8)
    reference, counterpart = right_reference_flip_swap(left, right)
    assert np.array_equal(reference, right[:, ::-1])
    assert np.array_equal(counterpart, left[:, ::-1])
    assert reference.flags.c_contiguous and counterpart.flags.c_contiguous
