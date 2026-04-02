from __future__ import annotations

import json
import os
from typing import Any, Mapping
import tempfile

import numpy as np


_META_KEY = "__rasptorch_meta__"
_FORMAT = "rasptorch-checkpoint-v1"


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _state_dict_to_numpy(state_dict: Any) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if not isinstance(state_dict, dict):
        return out

    for k, v in state_dict.items():
        name = str(k)
        try:
            if hasattr(v, "detach") and hasattr(v, "cpu") and hasattr(v, "numpy"):
                arr = np.asarray(v.detach().cpu().numpy(), dtype=np.float32)
            else:
                arr = np.asarray(v, dtype=np.float32)
            out[name] = np.ascontiguousarray(arr)
        except Exception:
            continue
    return out


def _resolve_checkpoint_path(path: str) -> str:
    """Resolve a user-supplied checkpoint path to a safe absolute path.

    Checkpoints are stored under a dedicated directory inside the system
    temporary directory. Absolute paths that do not already reside under this
    directory are interpreted relative to it, and any attempt to escape the
    directory via '..' segments is rejected.
    """
    raw = str(path or "").strip()
    if not raw:
        raise ValueError("Checkpoint path must not be empty")

    # Base directory for checkpoint files.
    base_dir = os.path.join(tempfile.gettempdir(), "rasptorch_checkpoints")
    os.makedirs(base_dir, exist_ok=True)
    base_abs = os.path.abspath(base_dir)

    candidate = raw
    if os.path.isabs(candidate):
        # If the absolute path is already under base_abs, keep it as-is.
        abs_candidate = os.path.abspath(os.path.normpath(candidate))
        try:
            common = os.path.commonpath([base_abs, abs_candidate])
        except ValueError:
            # Different drives on Windows, etc.
            common = ""
        if common == base_abs:
            final_path = abs_candidate
        else:
            # Treat an external absolute path as relative to base_dir.
            candidate = candidate.lstrip(os.sep)
            joined = os.path.join(base_abs, candidate)
            final_path = os.path.abspath(os.path.normpath(joined))
    else:
        joined = os.path.join(base_abs, candidate)
        final_path = os.path.abspath(os.path.normpath(joined))

    # Ensure the final path stays within base_abs.
    try:
        common = os.path.commonpath([base_abs, final_path])
    except ValueError:
        raise ValueError("Invalid checkpoint path")
    if common != base_abs:
        raise ValueError("Checkpoint path escapes allowed directory")

    return final_path


def save_checkpoint(path: str, payload: Mapping[str, Any]) -> str:
    """Save a checkpoint without requiring torch.

    For `.pt`/`.pth`, this writes a NumPy compressed archive that can be loaded
    back by `load_checkpoint` with `allow_pickle=False`.
    """

    safe_path = _resolve_checkpoint_path(path)

    state_dict = _state_dict_to_numpy(payload.get("state_dict", {}))
    payload_meta = {k: _json_safe(v) for k, v in dict(payload).items() if k != "state_dict"}
    state_keys = sorted(state_dict.keys())

    meta = {
        "format": _FORMAT,
        "payload": payload_meta,
        "state_keys": state_keys,
    }

    archive_items: dict[str, Any] = {_META_KEY: np.array(json.dumps(meta), dtype=np.str_)}
    for i, key in enumerate(state_keys):
        archive_items[f"arr_{i}"] = state_dict[key]

    with open(safe_path, "wb") as f:
        np.savez_compressed(f, **archive_items)
    return "rasptorch-npz"


def load_checkpoint(path: str) -> dict[str, Any]:
    """Load a checkpoint saved by `save_checkpoint`.

    Raises ValueError if the file is not a rasptorch npz checkpoint.
    """

    safe_path = _resolve_checkpoint_path(path)

    try:
        archive = np.load(safe_path, allow_pickle=False)
    except Exception as e:
        raise ValueError(f"Not a rasptorch checkpoint archive: {e}") from e

    with archive:
        if _META_KEY not in archive.files:
            raise ValueError("Missing checkpoint metadata")

        meta_raw = archive[_META_KEY]
        meta_text = str(meta_raw.tolist())
        try:
            meta = json.loads(meta_text)
        except Exception as e:
            raise ValueError(f"Invalid checkpoint metadata: {e}") from e

        if meta.get("format") != _FORMAT:
            raise ValueError("Unsupported checkpoint format")

        payload = dict(meta.get("payload", {}))
        state_keys = list(meta.get("state_keys", []))
        state_dict: dict[str, np.ndarray] = {}
        for i, key in enumerate(state_keys):
            arr_key = f"arr_{i}"
            if arr_key not in archive.files:
                continue
            state_dict[str(key)] = np.asarray(archive[arr_key], dtype=np.float32)

    payload["state_dict"] = state_dict
    payload["_checkpoint_path"] = safe_path
    return payload


def _safe_torch_load_legacy(path: str) -> dict[str, Any]:
    """Safely load a legacy torch checkpoint using weights_only=True."""
    try:
        import torch  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Converting legacy torch checkpoints requires torch (import failed: {e})") from e

    ext = os.path.splitext(str(path))[1].lower()
    if ext not in {".pt", ".pth"}:
        raise ValueError(f"Legacy torch checkpoint must end with .pt or .pth: {path}")

    try:
        data = torch.load(path, map_location="cpu", weights_only=True)  # type: ignore[call-arg]
    except TypeError as e:
        raise RuntimeError(
            "Legacy conversion requires a torch version that supports weights_only=True"
        ) from e
    except Exception as e:
        msg = str(e)
        if "Weights only load failed" in msg or "weights_only" in msg:
            raise RuntimeError(
                "Checkpoint cannot be loaded safely with weights_only=True; refusing conversion"
            ) from e
        raise RuntimeError(f"Failed to load legacy torch checkpoint: {e}") from e

    if not isinstance(data, dict):
        raise ValueError("Legacy torch checkpoint must be a dict-like payload")
    return data


def convert_legacy_torch_checkpoint(src_path: str, dst_path: str) -> dict[str, Any]:
    """Convert a legacy torch `.pt`/`.pth` checkpoint to rasptorch format."""
    if not str(dst_path).strip():
        raise ValueError("Destination path must not be empty")

    src = os.path.abspath(src_path)
    dst = os.path.abspath(dst_path)
    ext = os.path.splitext(dst)[1].lower()
    if ext not in {".pt", ".pth"}:
        raise ValueError("Destination must end with .pt or .pth")

    raw = _safe_torch_load_legacy(src)

    # Some legacy checkpoints are just a state_dict mapping name -> tensor.
    if "state_dict" in raw and isinstance(raw.get("state_dict"), dict):
        payload = dict(raw)
    else:
        payload = {"state_dict": raw}

    save_checkpoint(dst, payload)

    sd = _state_dict_to_numpy(payload.get("state_dict", {}))
    return {
        "status": "success",
        "source_path": src,
        "output_path": dst,
        "format": "rasptorch-npz",
        "num_tensors": len(sd),
    }