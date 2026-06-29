from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any



REPO_ROOT = Path(__file__).resolve().parents[3]


MANIFEST_COLUMNS = [
    "pair_index",
    "prev_frame_id",
    "cur_frame_id",
    "forward_flow_path",
    "backward_flow_path",
    "forward_confidence_path",
    "backward_confidence_path",
    "occlusion_path",
    "forward_runtime_ms",
    "backward_runtime_ms",
    "exception_type",
    "exception_message",
]

for _artifact in [
    "forward_flow",
    "backward_flow",
    "forward_confidence",
    "backward_confidence",
    "occlusion",
]:
    MANIFEST_COLUMNS.extend(
        [
            f"{_artifact}_exists",
            f"{_artifact}_nonempty",
            f"{_artifact}_shape",
            f"{_artifact}_dtype",
            f"{_artifact}_finite",
            f"{_artifact}_valid",
            f"{_artifact}_size_mb",
            f"{_artifact}_error",
        ]
    )

MANIFEST_COLUMNS.extend(["status", "error"])


@dataclass(frozen=True)
class FlowFrame:
    frame_id: str
    left_path: Path
    height: int
    width: int


@dataclass(frozen=True)
class FlowPairPaths:
    forward_flow: Path
    backward_flow: Path
    forward_confidence: Path
    backward_confidence: Path
    occlusion: Path


def resolve_manifest_path(sequence_dir: Path, value: str | None, fallback: Path) -> Path:
    if not value:
        return fallback
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    rooted = REPO_ROOT / path
    if rooted.exists():
        return rooted
    return sequence_dir / path


def load_temporal_gt_frames(sequence_dir: Path) -> list[FlowFrame]:
    metadata_csv = sequence_dir / "metadata.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Missing sequence metadata: {metadata_csv}")

    rows: list[dict[str, str]]
    with metadata_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows found in {metadata_csv}")

    frames: list[FlowFrame] = []
    expected_ids: list[int] = []
    first_shape: tuple[int, int] | None = None
    for row in rows:
        frame_id = row.get("frame_id") or row.get("id")
        if frame_id is None:
            raise RuntimeError("metadata.csv must contain frame_id or id")
        expected_ids.append(int(frame_id))
        left_path = resolve_manifest_path(
            sequence_dir,
            row.get("left_path"),
            sequence_dir / "left" / f"{frame_id}.png",
        )
        if not left_path.exists():
            raise FileNotFoundError(f"Missing left RGB frame for {frame_id}: {left_path}")
        import cv2

        image = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read left RGB frame for {frame_id}: {left_path}")
        height, width = image.shape[:2]
        if first_shape is None:
            first_shape = (height, width)
        elif (height, width) != first_shape:
            raise RuntimeError(
                f"Inconsistent image size at frame {frame_id}: {(height, width)} != {first_shape}"
            )
        frames.append(FlowFrame(frame_id=frame_id, left_path=left_path, height=height, width=width))

    sorted_ids = sorted(expected_ids)
    if expected_ids != sorted_ids:
        raise RuntimeError("Frame ids in metadata.csv are not sorted")
    if sorted_ids != list(range(sorted_ids[0], sorted_ids[0] + len(sorted_ids))):
        raise RuntimeError(f"Frame ids are not continuous: first={sorted_ids[0]} count={len(sorted_ids)}")

    return frames


def pair_stem(prev_id: str, cur_id: str) -> str:
    return f"{prev_id}_to_{cur_id}"


def paths_for_pair(output_dir: Path, prev_id: str, cur_id: str) -> FlowPairPaths:
    forward = pair_stem(prev_id, cur_id)
    backward = pair_stem(cur_id, prev_id)
    return FlowPairPaths(
        forward_flow=output_dir / "forward_flow" / f"{forward}.npy",
        backward_flow=output_dir / "backward_flow" / f"{backward}.npy",
        forward_confidence=output_dir / "forward_confidence" / f"{forward}.npy",
        backward_confidence=output_dir / "backward_confidence" / f"{backward}.npy",
        occlusion=output_dir / "occlusion" / f"{forward}.npy",
    )


def create_flow_cache_dirs(output_dir: Path) -> None:
    for name in [
        "forward_flow",
        "backward_flow",
        "forward_confidence",
        "backward_confidence",
        "occlusion",
    ]:
        (output_dir / name).mkdir(parents=True, exist_ok=True)


def save_npy(path: Path, array: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np

    np.save(path, array)


def array_validation(path: Path, expected_shape: tuple[int, ...], expected_kinds: set[str]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "nonempty": False,
        "shape": "",
        "dtype": "",
        "finite": False,
        "valid": False,
        "size_mb": 0.0,
        "error": "",
    }
    if not path.exists():
        out["error"] = "missing"
        return out
    out["nonempty"] = path.stat().st_size > 0
    out["size_mb"] = path.stat().st_size / (1024.0 * 1024.0)
    if not out["nonempty"]:
        out["error"] = "empty"
        return out
    try:
        import numpy as np

        arr = np.load(path, allow_pickle=False)
    except Exception as exc:  # pragma: no cover - defensive validation path
        out["error"] = f"load_failed:{exc}"
        return out
    out["shape"] = "x".join(str(x) for x in arr.shape)
    out["dtype"] = str(arr.dtype)
    kind = arr.dtype.kind
    finite = bool(np.isfinite(arr).all()) if kind in {"f", "i", "u", "b"} else False
    out["finite"] = finite
    shape_ok = tuple(arr.shape) == expected_shape
    dtype_ok = kind in expected_kinds
    out["valid"] = bool(out["nonempty"] and shape_ok and dtype_ok and finite)
    if not shape_ok:
        out["error"] = f"shape_mismatch:expected={expected_shape}"
    elif not dtype_ok:
        out["error"] = f"dtype_kind_mismatch:expected={sorted(expected_kinds)}"
    elif not finite:
        out["error"] = "nonfinite_values"
    return out


def validate_pair_outputs(paths: FlowPairPaths, height: int, width: int) -> dict[str, dict[str, Any]]:
    return {
        "forward_flow": array_validation(paths.forward_flow, (height, width, 2), {"f"}),
        "backward_flow": array_validation(paths.backward_flow, (height, width, 2), {"f"}),
        "forward_confidence": array_validation(paths.forward_confidence, (height, width), {"f"}),
        "backward_confidence": array_validation(paths.backward_confidence, (height, width), {"f"}),
        "occlusion": array_validation(paths.occlusion, (height, width), {"b", "f"}),
    }


def flatten_validation(prefix: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_exists": result["exists"],
        f"{prefix}_nonempty": result["nonempty"],
        f"{prefix}_shape": result["shape"],
        f"{prefix}_dtype": result["dtype"],
        f"{prefix}_finite": result["finite"],
        f"{prefix}_valid": result["valid"],
        f"{prefix}_size_mb": f"{float(result['size_mb']):.6f}",
        f"{prefix}_error": result["error"],
    }


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in MANIFEST_COLUMNS})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

