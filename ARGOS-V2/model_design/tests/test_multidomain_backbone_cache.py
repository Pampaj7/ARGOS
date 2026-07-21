"""Contract tests for the separate OOD frozen-backbone cache builder."""
from __future__ import annotations

import numpy as np

from scripts.build_multidomain_backbone_cache import (
    CACHE_HEIGHT, CACHE_WIDTH, d4d_records, resize_to_cache, serv_records,
)


def test_d4d_records_are_unique_true_stereo_and_causal():
    records = d4d_records(limit_anchors=1)
    assert len(records) == 4
    assert [record.temporal_order for record in records] == [0, 1, 2, 3]
    assert len({record.frame_id for record in records}) == len(records)
    assert all(record.rectified for record in records)
    assert all(record.left_path != record.right_path for record in records)


def test_serv_records_preserve_sequence_order_and_true_stereo():
    records = serv_records(limit_sequences=1)
    assert len(records) >= 2
    assert [record.temporal_order for record in records] == list(range(len(records)))
    assert len({record.sequence_id for record in records}) == 1
    assert all(not record.rectified for record in records)
    assert all(record.left_path != record.right_path for record in records)


def test_resize_contract_scales_disparity_and_preserves_validity():
    native = np.full((10, 20), 4.0, np.float32)
    cached, valid = resize_to_cache(native, native_width=20)
    assert cached.shape == (CACHE_HEIGHT, CACHE_WIDTH)
    assert valid.shape == cached.shape
    assert cached.dtype == np.float16 and valid.dtype == np.uint8
    # 4 native pixels at width 20 become 4 * (180 / 20) cache-grid pixels.
    assert np.allclose(cached, 36.0)
    assert valid.all()


def test_resize_marks_nonpositive_and_nonfinite_predictions_invalid():
    native = np.full((10, 20), 4.0, np.float32)
    native[0, 0], native[1, 1] = 0.0, np.nan
    _, valid = resize_to_cache(native, native_width=20)
    assert valid.min() == 0
    assert valid.max() == 1
