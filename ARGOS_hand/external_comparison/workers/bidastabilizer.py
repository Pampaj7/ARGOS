"""Thin, strict BiDAStabilizer adapter around the pinned official sources.

The original checkpoints are trusted only while converting them offline.  The
runtime consumes the resulting tensor-only state dictionaries with
``weights_only=True``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARGOS = ROOT.parent
UPSTREAM = ROOT / "upstreams" / "bidavideo"
CHECKPOINTS = ROOT / "checkpoints"
DERIVED = CHECKPOINTS / "derived"
METHOD = "bidastabilizer_bidirectional_offline"
DERIVED_NAMES = {
    "raftstereo": "bidavideo_raftstereo_robust.state_dict.pth",
    "stabilizer": "bidavideo_raftstereo_stabilizer_robust.state_dict.pth",
    "sea_raft": "bidavideo_sea_raft.state_dict.pth",
}

sys.path.insert(0, str(ROOT))
from bridge import read_input, write_input, write_output  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _key_contract(state: Mapping[str, Any]) -> dict[str, Any]:
    keys = []
    for key, value in sorted(state.items()):
        if not hasattr(value, "dtype") or not hasattr(value, "shape"):
            raise ValueError(f"non-tensor state item: {key}")
        keys.append({"key": key, "dtype": str(value.dtype), "shape": list(value.shape)})
    encoded = json.dumps(keys, sort_keys=True, separators=(",", ":")).encode()
    return {"tensor_count": len(keys), "keys_sha256": hashlib.sha256(encoded).hexdigest()}


def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - exercised by CPU-only shell preflight
        raise RuntimeError("BiDAStabilizer requires the pinned torch interpreter") from error
    return torch


def _state(value: Any, torch: Any) -> dict[str, Any]:
    if isinstance(value, Mapping) and isinstance(value.get("model"), Mapping):
        value = value["model"]
    if not isinstance(value, Mapping) or not value:
        raise ValueError("checkpoint does not contain a model state dictionary")
    state = dict(value)
    if not all(isinstance(key, str) and isinstance(item, torch.Tensor) for key, item in state.items()):
        raise ValueError("checkpoint model contains non-tensor state")
    return state


def _lock() -> tuple[dict[str, Any], dict[str, Any]]:
    data = json.loads((ROOT / "checkpoints.lock.json").read_text())
    item = data["checkpoints"]["bidastabilizer_raftstereo_robust"]
    if item.get("status") != "READY":
        raise RuntimeError(f"BiDAStabilizer is blocked: {item.get('reason')}")
    return data, item


def _originals(item: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    found = {Path(entry["path"]).name: entry for entry in item["artifacts"]}
    required = {"raftstereo_robust.pth", "raftstereo_stabilizer_robust.pth", "Tartan-C-T-TSKH-spring540x960-S.pth"}
    if set(found) != required:
        raise ValueError("BiDA original checkpoint allowlist is incomplete")
    for entry in found.values():
        path = ROOT / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"] or _sha256(path) != entry["sha256"]:
            raise ValueError(f"original checkpoint hash mismatch: {path}")
    return found


def _atomic_torch_save(torch: Any, value: Mapping[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(value), temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def convert() -> dict[str, Any]:
    """Create reproducible tensor-only derivatives from verified pinned originals."""
    torch = _torch(); lock, item = _lock(); originals = _originals(item)
    source = {
        "raftstereo": originals["raftstereo_robust.pth"],
        "stabilizer": originals["raftstereo_stabilizer_robust.pth"],
        "sea_raft": originals["Tartan-C-T-TSKH-spring540x960-S.pth"],
    }
    derived: dict[str, Any] = {}
    states: dict[str, dict[str, Any]] = {}
    for name, entry in source.items():
        original = ROOT / entry["path"]
        # This is the sole trusted, offline deserialization boundary.
        state = _state(torch.load(original, map_location="cpu", weights_only=False), torch)
        target = DERIVED / DERIVED_NAMES[name]
        _atomic_torch_save(torch, state, target)
        states[name] = state
        derived[name] = {"path": str(target.relative_to(ROOT)), "bytes": target.stat().st_size,
                         "sha256": _sha256(target), "source_path": entry["path"],
                         "source_bytes": entry["bytes"], "source_sha256": entry["sha256"],
                         "key_contract": _key_contract(state)}
    embedded = {key.removeprefix("raft.model."): value for key, value in states["stabilizer"].items() if key.startswith("raft.model.")}
    if set(embedded) != set(states["sea_raft"]):
        raise ValueError("stabilizer embedded SEA-RAFT state has an unexpected key contract")
    equal = all(torch.equal(embedded[key], states["sea_raft"][key]) for key in embedded)
    derived["stabilizer"]["embedded_sea_raft"] = {
        "tensor_count": len(embedded), "keys_sha256": _key_contract(embedded)["keys_sha256"],
        "matches_standalone": equal,
        "runtime_order": "load standalone SEA-RAFT then strict-load stabilizer; embedded raft.model.* intentionally overwrites standalone initialization",
    }
    item["derived"] = derived
    item["runtime_load"] = {"weights_only": True, "strict": True, "device": "cuda:0 logical device when CUDA_VISIBLE_DEVICES is set"}
    temporary = ROOT / ".checkpoints.lock.json.tmp"
    temporary.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, ROOT / "checkpoints.lock.json")
    return derived


def _derived(item: Mapping[str, Any]) -> dict[str, Path]:
    values = item.get("derived")
    if not isinstance(values, Mapping) or set(values) != set(DERIVED_NAMES):
        raise RuntimeError("derived BiDA checkpoints are missing; run --convert first")
    output: dict[str, Path] = {}
    for name, value in values.items():
        if not isinstance(value, Mapping):
            raise ValueError("invalid derived checkpoint lock")
        path = ROOT / str(value["path"])
        if not path.is_file() or path.stat().st_size != value["bytes"] or _sha256(path) != value["sha256"]:
            raise ValueError(f"derived checkpoint hash mismatch: {path}")
        output[name] = path
    return output


def _load_safe(torch: Any, path: Path) -> dict[str, Any]:
    return _state(torch.load(path, map_location="cpu", weights_only=True), torch)


def _install_sources(torch: Any, sea_state: Mapping[str, Any]) -> tuple[Any, Any]:
    """Use pinned sources directly, replacing only their Configurable SEA wrapper."""
    source_paths = [str(ROOT / "upstreams"), str(UPSTREAM / "third_party" / "RAFT-Stereo"), str(UPSTREAM / "third_party" / "SEA-RAFT")]
    sys.path[:0] = [path for path in source_paths if path not in sys.path]
    import importlib
    sea_core = importlib.import_module("bidavideo.third_party.SEA-RAFT.core.raft")
    sea_utils = importlib.import_module("bidavideo.third_party.SEA-RAFT.core.utils.utils")

    class SEARAFTModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            args = SimpleNamespace(use_var=True, var_min=0, var_max=10, pretrain="resnet18", initial_dim=64,
                                   block_dims=[64, 128, 256], radius=4, dim=128, num_blocks=2, iters=4)
            self.model = sea_core.RAFT(args)
            self.model.load_state_dict(dict(sea_state), strict=True)

        def forward_fullres(self, image1: Any, image2: Any) -> Any:
            padder = sea_utils.InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            flow = self.model(image1, image2, iters=4, test_mode=True)["flow"][-1]
            return padder.unpad(flow)

    module = types.ModuleType("bidavideo.models.sea_raft_model")
    module.SEARAFTModel = SEARAFTModel
    sys.modules[module.__name__] = module
    raft_core = importlib.import_module("bidavideo.third_party.RAFT-Stereo.core.raft_stereo")
    raft_utils = importlib.import_module("bidavideo.third_party.RAFT-Stereo.core.utils.utils")
    stabilizer_core = importlib.import_module("bidavideo.models.core.bidastabilizer")
    return (raft_core.RAFTStereo, raft_utils.InputPadder, stabilizer_core.BiDAStabilizer)


def _install_configurable_shim(torch: Any) -> None:
    """Permit importing the unchanged official wrappers without pytorch3d."""
    if "pytorch3d.implicitron.tools.config" in sys.modules:
        return
    config = types.ModuleType("pytorch3d.implicitron.tools.config")

    class Configurable:
        def __init__(self) -> None:
            torch.nn.Module.__init__(self)

    config.Configurable = Configurable
    for name in ("pytorch3d", "pytorch3d.implicitron", "pytorch3d.implicitron.tools"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules[config.__name__] = config


def _official_instance(cls: Any, torch: Any) -> Any:
    """Instantiate Configurable wrappers exactly through their official post-init."""
    value = cls.__new__(cls)
    torch.nn.Module.__init__(value)
    value.__post_init__()
    return value


def _official_reference(values: Mapping[str, np.ndarray], device_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Run untouched official RAFTStereoModel + BiDAStabilizer on derived states."""
    torch = _torch(); device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("official BiDA reference requires CUDA")
    _, item = _lock(); paths = _derived(item)
    source_paths = [str(ROOT / "upstreams"), str(UPSTREAM / "third_party" / "RAFT-Stereo"), str(UPSTREAM / "third_party" / "SEA-RAFT")]
    sys.path[:0] = [path for path in source_paths if path not in sys.path]
    _install_configurable_shim(torch)
    import importlib
    raft_module = importlib.import_module("bidavideo.models.raft_stereo_model")
    importlib.import_module("bidavideo.models.sea_raft_model")
    stabilizer_module = importlib.reload(importlib.import_module("bidavideo.models.core.bidastabilizer"))
    original_load = torch.load

    def derived_load(path: Any, *args: Any, **kwargs: Any) -> Any:
        name = Path(path).name
        if name == "raftstereo_robust.pth":
            return {"model": _load_safe(torch, paths["raftstereo"])}
        if name == "Tartan-C-T-TSKH-spring540x960-S.pth":
            return _load_safe(torch, paths["sea_raft"])
        return original_load(path, *args, **kwargs)

    torch.load = derived_load
    try:
        raft = _official_instance(raft_module.RAFTStereoModel, torch).to(device).eval()
        stabilizer = stabilizer_module.BiDAStabilizer().to(device).eval()
        stabilizer.load_state_dict(_load_safe(torch, paths["stabilizer"]), strict=True)
        left = torch.from_numpy(np.ascontiguousarray(values["rgb_left"])).to(device)
        right = torch.from_numpy(np.ascontiguousarray(values["rgb_right"])).to(device)
        batch = {"stereo_video": torch.stack((left, right), dim=1)}
        with torch.inference_mode():
            raw = raft(batch, iters=32)["disparity"]
            refined = raft.forward_stabilizer(batch, stabilizer, iters=32)["disparity"]
    finally:
        torch.load = original_load
    return (raw.detach().cpu().numpy().astype(np.float32, copy=False),
            refined.detach().cpu().numpy().astype(np.float32, copy=False))


def equivalence(values: Mapping[str, np.ndarray], device_name: str) -> dict[str, Any]:
    """Prove the independent thin worker matches the unchanged official wrappers."""
    raw, refined, _ = infer(values, device_name)
    official_raw, official_refined = _official_reference(values, device_name)
    result: dict[str, Any] = {"method": METHOD, "frames": int(raw.shape[0]), "device": device_name,
                              "rtol": 1e-5, "atol": 1e-5, "worker_code_sha256": _sha256(Path(__file__))}
    for name, actual, expected in (("raw", raw, official_raw), ("refined", refined, official_refined)):
        delta = np.abs(actual - expected)
        result[name] = {"allclose": bool(np.allclose(actual, expected, rtol=result["rtol"], atol=result["atol"])),
                        "max_abs": float(delta.max()), "worker_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
                        "official_sha256": hashlib.sha256(expected.tobytes()).hexdigest()}
    if not all(result[name]["allclose"] for name in ("raw", "refined")):
        raise ValueError("BiDA worker does not match the official reference")
    return result


def _models(device: Any) -> tuple[Any, Any, Any]:
    torch = _torch(); _, item = _lock(); _originals(item); paths = _derived(item)
    raft_class, padder_class, stabilizer_class = _install_sources(torch, _load_safe(torch, paths["sea_raft"]))
    args = SimpleNamespace(hidden_dims=[128] * 3, corr_implementation="reg", shared_backbone=False, corr_levels=4,
                           corr_radius=4, n_downsample=2, slow_fast_gru=False, n_gru_layers=3,
                           mixed_precision=False, context_norm="batch")
    raft = raft_class(args).to(device).eval()
    raft.load_state_dict(_load_safe(torch, paths["raftstereo"]), strict=True)
    stabilizer = stabilizer_class().to(device).eval()
    stabilizer.load_state_dict(_load_safe(torch, paths["stabilizer"]), strict=True)
    return raft, stabilizer, padder_class


def infer(values: Mapping[str, np.ndarray], device_name: str) -> tuple[np.ndarray, np.ndarray, dict[str, int | None]]:
    torch = _torch(); device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    raft, stabilizer, padder_class = _models(device)
    left = torch.from_numpy(np.ascontiguousarray(values["rgb_left"])).to(device)
    right = torch.from_numpy(np.ascontiguousarray(values["rgb_right"])).to(device)
    signed = []
    with torch.inference_mode():
        for image_left, image_right in zip(left, right):
            image_left, image_right = image_left[None], image_right[None]
            padder = padder_class(image_left.shape, divis_by=32)
            image_left, image_right = padder.pad(image_left, image_right)
            _, flow = raft(image_left, image_right, iters=32, test_mode=True)
            signed.append(padder.unpad(flow))
        signed_value = torch.stack(signed, dim=0).squeeze(1)
        refined_signed = stabilizer.forward_batch(left, signed_value, kernel_size=50)
        # Upstream returns [T,1,1,H,W] for overlapping kernel windows and
        # [T,1,H,W] when the whole sequence fits one window.
        if refined_signed.ndim == 5:
            refined_signed = refined_signed.squeeze(1)
    raw = signed_value.abs().detach().cpu().numpy().astype(np.float32, copy=False)
    refined = refined_signed.abs().detach().cpu().numpy().astype(np.float32, copy=False)
    if raw.shape != refined.shape or raw.ndim != 4 or raw.shape[1] != 1 or not np.isfinite(raw).all() or not np.isfinite(refined).all():
        raise ValueError("official BiDA inference returned invalid tensors")
    memory = {"cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None,
              "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()) if device.type == "cuda" else None}
    return raw, refined, memory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path); parser.add_argument("--output", type=Path)
    parser.add_argument("--raw-output", type=Path, help="write the RAFT-robust bridge input used by the stabilizer")
    parser.add_argument("--protocol", type=Path); parser.add_argument("--checkpoints", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--purpose", default="SMOKE_DIAGNOSTIC", choices=("SMOKE_DIAGNOSTIC", "D2_FULL_DIAGNOSTIC"))
    parser.add_argument("--convert", action="store_true", help="offline conversion of verified original checkpoint bytes")
    parser.add_argument("--equivalence", type=Path, help="write official-wrapper equivalence JSON; does not write predictions")
    args = parser.parse_args()
    if args.convert:
        print(json.dumps(convert(), indent=2, sort_keys=True)); return
    if args.equivalence:
        if not args.input:
            parser.error("--equivalence requires --input")
        if args.equivalence.exists():
            raise FileExistsError("refusing to overwrite equivalence artifact")
        values, input_meta = read_input(args.input)
        result = equivalence(values, args.device)
        _, checkpoint = _lock()
        result |= {"source_input_sha256": input_meta["input_sha256"], "source_rgb_input_sha256": input_meta.get("rgb_input_sha256"),
                   "frame_ids": values["frame_ids"].tolist(), "upstream_commit": json.loads((ROOT / "upstreams.lock.json").read_text())["upstreams"]["bidavideo"]["commit"],
                   "checkpoints": {name: {key: checkpoint["derived"][name][key] for key in ("sha256", "source_sha256")} for name in DERIVED_NAMES}}
        args.equivalence.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        return
    if not all((args.input, args.output, args.raw_output, args.protocol, args.checkpoints)):
        parser.error("--input --output --raw-output --protocol and --checkpoints are required for inference")
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("method") != METHOD or protocol.get("online_or_h4") or not protocol.get("future_frames_required"):
        raise ValueError("invalid BiDA noncausal protocol")
    if any(path.exists() or path.with_suffix(".json").exists() for path in (args.output, args.raw_output)):
        raise FileExistsError("refusing to overwrite BiDA output")
    values, input_meta = read_input(args.input)
    raw, refined, memory = infer(values, args.device)
    raw_values = dict(values) | {"raw_disparity": raw, "raw_valid": np.isfinite(raw) & (raw > 0)}
    torch = _torch()
    upstream = json.loads((ROOT / "upstreams.lock.json").read_text())["upstreams"]["bidavideo"]
    _, checkpoint = _lock()
    provenance = {"purpose": args.purpose, "publication": "TEST_ONLY", "method": METHOD,
                  "protocol_sha256": _sha256(args.protocol), "upstream": {"name": "bidavideo", "commit": upstream["commit"]},
                  "checkpoint": {"id": "bidastabilizer_raftstereo_robust", "sha256": checkpoint["sha256"],
                                 "original_artifacts": [{key: entry[key] for key in ("path", "sha256")} for entry in checkpoint["artifacts"]],
                                 "derived_artifacts": [{"path": checkpoint["derived"][name]["path"], "sha256": checkpoint["derived"][name]["sha256"]} for name in DERIVED_NAMES]},
                  "worker": {"artifact_id": "workers/bidastabilizer.py", "sha256": _sha256(Path(__file__))},
                  "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda, "device": str(args.device),
                              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""), **memory}}
    raw_meta = write_input(args.raw_output, raw_values, {"raw_method": "RAFTStereo robust, 32 iterations, corr_implementation=reg",
                                                          "upstream_source_input_sha256": input_meta["input_sha256"],
                                                          "upstream_source_rgb_input_sha256": input_meta.get("rgb_input_sha256"), **provenance})
    write_output(args.output, refined, raw_values, raw_meta, METHOD, provenance)


if __name__ == "__main__":
    main()
