"""NPZ/JSON-only boundary for external temporal-stereo methods."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

INPUT_KEYS = ("rgb_left", "rgb_right", "raw_disparity", "raw_valid", "frame_ids")


def _digest(values: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in INPUT_KEYS:
        value = values[name]
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def rgb_input_sha256(values: dict[str, np.ndarray]) -> str:
    """Hash the immutable RGB/frame snapshot independently of predictions."""
    digest = hashlib.sha256()
    for name in ("rgb_left", "rgb_right", "frame_ids"):
        value = values[name]
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _load_npz(data: bytes) -> np.lib.npyio.NpzFile:
    return np.load(io.BytesIO(data), allow_pickle=False)


def validate_input(values: dict[str, np.ndarray]) -> None:
    if set(values) != set(INPUT_KEYS):
        raise ValueError(f"input keys must be {INPUT_KEYS}")
    left, right, disparity, valid, ids = (values[key] for key in INPUT_KEYS)
    if left.dtype != np.float32 or right.dtype != np.float32 or disparity.dtype != np.float32 or valid.dtype != np.bool_:
        raise ValueError("RGB/disparity must be float32 and raw_valid must be bool")
    if left.ndim != 4 or left.shape[1] != 3 or left.shape != right.shape:
        raise ValueError("RGB arrays must be matching [T,3,H,W]")
    if disparity.shape != (left.shape[0], 1, left.shape[2], left.shape[3]) or valid.shape != disparity.shape:
        raise ValueError("disparity/valid must be [T,1,H,W] on the RGB grid")
    if ids.ndim != 1 or ids.shape[0] != left.shape[0] or ids.dtype.kind not in "US":
        raise ValueError("frame_ids must be a string [T] array")
    if len(set(ids.tolist())) != len(ids) or any(not str(item) for item in ids):
        raise ValueError("frame_ids must be non-empty and unique")
    if not all(np.isfinite(item).all() for item in (left, right, disparity)):
        raise ValueError("arrays must be finite")
    if np.any(disparity[valid] <= 0):
        raise ValueError("valid raw_disparity must be positive-left")


def read_input(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with _load_npz(path.read_bytes()) as loaded:
        values = {key: loaded[key] for key in INPUT_KEYS}
    validate_input(values)
    metadata = json.loads(path.with_suffix(".json").read_bytes())
    if metadata.get("input_sha256") != _digest(values) or metadata.get("frame_ids") != values["frame_ids"].tolist():
        raise ValueError("input JSON hash/frame IDs do not match NPZ")
    if "rgb_input_sha256" in metadata and metadata["rgb_input_sha256"] != rgb_input_sha256(values):
        raise ValueError("input JSON RGB snapshot hash does not match NPZ")
    return values, metadata


def write_input(path: Path, values: dict[str, np.ndarray], metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    validate_input(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)
    info = dict(metadata or {}) | {"input_sha256": _digest(values), "rgb_input_sha256": rgb_input_sha256(values),
                                   "frame_ids": values["frame_ids"].tolist(), "contract": "external-comparison-v1"}
    path.with_suffix(".json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    return info


def validate_output(prediction: np.ndarray, values: dict[str, np.ndarray], frame_ids: list[str]) -> None:
    if prediction.dtype != np.float32 or prediction.shape != values["raw_disparity"].shape:
        raise ValueError("prediction must be float32 [T,1,H,W] on the input grid")
    if not np.isfinite(prediction).all() or np.any(prediction[values["raw_valid"]] <= 0):
        raise ValueError("valid predictions must be finite positive-left")
    if frame_ids != values["frame_ids"].tolist():
        raise ValueError("output frame_ids do not exactly match input")


def write_output(path: Path, prediction: np.ndarray, values: dict[str, np.ndarray], input_meta: dict[str, Any], method: str,
                 metadata: dict[str, Any] | None = None) -> None:
    validate_output(prediction, values, values["frame_ids"].tolist())
    np.savez_compressed(path, disparity=prediction, frame_ids=values["frame_ids"])
    info = dict(metadata or {}) | {"contract": "external-comparison-v1", "method": method,
                                   "input_sha256": input_meta["input_sha256"], "source_input_sha256": input_meta["input_sha256"],
                                   "source_rgb_input_sha256": input_meta.get("rgb_input_sha256", rgb_input_sha256(values)),
                                   "frame_ids": values["frame_ids"].tolist()}
    path.with_suffix(".json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")


def read_output_snapshot(path: Path, values: dict[str, np.ndarray], input_meta: dict[str, Any], method: str) -> tuple[np.ndarray, str]:
    data = path.read_bytes()
    with _load_npz(data) as loaded:
        if set(loaded.files) != {"disparity", "frame_ids"}:
            raise ValueError("output NPZ must contain only disparity and frame_ids")
        prediction, frame_ids = loaded["disparity"], loaded["frame_ids"]
    metadata = json.loads(path.with_suffix(".json").read_bytes())
    validate_output(prediction, values, frame_ids.tolist())
    if (metadata.get("contract") != "external-comparison-v1" or metadata.get("method") != method
            or metadata.get("input_sha256") != input_meta["input_sha256"]
            or metadata.get("source_input_sha256") != input_meta["input_sha256"]
            or metadata.get("source_rgb_input_sha256") != input_meta.get("rgb_input_sha256", rgb_input_sha256(values))):
        raise ValueError("output metadata does not match the validated bridge input/method")
    return prediction, hashlib.sha256(data).hexdigest()


def read_output(path: Path, values: dict[str, np.ndarray], input_meta: dict[str, Any], method: str) -> np.ndarray:
    return read_output_snapshot(path, values, input_meta, method)[0]


def positive_left_to_bidastabilizer(disparity: np.ndarray) -> np.ndarray:
    """BiDAStabilizer negates its input internally; retain its signed convention."""
    return -disparity


def bidastabilizer_to_positive_left(disparity: np.ndarray) -> np.ndarray:
    return -disparity


def resize_disparity(disparity: np.ndarray, height: int, width: int) -> np.ndarray:
    """Nearest resize with the required horizontal pixel-disparity scale."""
    if disparity.ndim != 4 or height < 1 or width < 1:
        raise ValueError("expected [T,1,H,W] and positive output dimensions")
    _, _, old_height, old_width = disparity.shape
    y = np.minimum((np.arange(height) * old_height // height), old_height - 1)
    x = np.minimum((np.arange(width) * old_width // width), old_width - 1)
    return (disparity[:, :, y[:, None], x] * (width / old_width)).astype(np.float32, copy=False)
