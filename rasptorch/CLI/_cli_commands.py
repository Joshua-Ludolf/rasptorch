"""Compatibility shim for CLI commands.

This module re-exports from the implementation module.
"""

from __future__ import annotations


from typing import Any, Dict, Tuple, List, Optional
import numpy as np
import rasptorch
import os
import tempfile
import uuid


class _GRUSequence(rasptorch.nn.Module):
    """Wrap a GRU to return only the full output sequence (Tensor)."""

    def __init__(self, gru: Any) -> None:
        super().__init__()
        self.gru = gru

    def forward(self, x):
        if len(getattr(x, "shape", ())) == 2:
            x = x.view(x.shape[0], 1, x.shape[1])
        output, _hidden = self.gru(x)
        return output


class _GRULastHidden(rasptorch.nn.Module):
    """Wrap a GRU to return a 2D (batch, hidden) Tensor for composition."""

    def __init__(self, gru: Any) -> None:
        super().__init__()
        self.gru = gru

    def forward(self, x):
        if len(getattr(x, "shape", ())) == 2:
            x = x.view(x.shape[0], 1, x.shape[1])
        _output, hidden = self.gru(x)
        # hidden: (1, batch, hidden)
        return hidden[0]


class _SelfAttention(rasptorch.nn.Module):
    """Wrap MultiheadAttention as self-attention for Sequential composition."""

    def __init__(self, mha: Any) -> None:
        super().__init__()
        self.mha = mha

    def forward(self, x):
        # rasptorch.nn.MultiheadAttention expects [B,T,E]. For convenience in the CLI
        # we allow [B,E] and treat it as a length-1 sequence.
        squeeze_seq = False
        if len(getattr(x, "shape", ())) == 2:
            squeeze_seq = True
            x = x.view(x.shape[0], 1, x.shape[1])

        out = self.mha(x, x, x, need_weights=False)
        if isinstance(out, tuple):
            out = out[0]
        if squeeze_seq:
            return out[:, 0, :]
        return out


# Global session directory for model state persistence
_SESSION_DIR = os.path.join(tempfile.gettempdir(), "rasptorch_cli_session")
if not os.path.exists(_SESSION_DIR):
    os.makedirs(_SESSION_DIR)


class TensorCommands:
    """Tensor creation and manipulation commands."""

    @staticmethod
    def create_random(shape: Tuple[int, ...], dtype: str = "float32", device: str = "cpu") -> Dict[str, Any]:
        """Create random tensor."""
        shape_tuple = shape if isinstance(shape, tuple) else tuple(shape)
        data = np.random.randn(*shape_tuple).astype(dtype)
        tensor = rasptorch.Tensor(data, device=device)
        return {
            "status": "success",
            "tensor_id": str(id(tensor)),
            "shape": list(tensor.shape),
            "dtype": str(tensor.data.dtype),
            "device": device,
        }

    @staticmethod
    def create_zeros(shape: Tuple[int, ...], dtype: str = "float32", device: str = "cpu") -> Dict[str, Any]:
        """Create zero tensor."""
        shape_tuple = shape if isinstance(shape, tuple) else tuple(shape)
        data = np.zeros(shape_tuple, dtype=dtype)
        tensor = rasptorch.Tensor(data, device=device)
        return {
            "status": "success",
            "tensor_id": str(id(tensor)),
            "shape": list(tensor.shape),
            "dtype": str(tensor.data.dtype),
            "device": device,
        }

    @staticmethod
    def create_ones(shape: Tuple[int, ...], dtype: str = "float32", device: str = "cpu") -> Dict[str, Any]:
        """Create ones tensor."""
        shape_tuple = shape if isinstance(shape, tuple) else tuple(shape)
        data = np.ones(shape_tuple, dtype=dtype)
        tensor = rasptorch.Tensor(data, device=device)
        return {
            "status": "success",
            "tensor_id": str(id(tensor)),
            "shape": list(tensor.shape),
            "dtype": str(tensor.data.dtype),
            "device": device,
        }


class ModelCommands:
    """Model creation and training commands."""

    def __init__(self):
        self.models: Dict[str, Any] = self._load_session_state()
        self.optimizers: Dict[str, Any] = {}
        self.lora_adapters: Dict[str, Any] = {}

    def _get_session_file(self, model_id: str) -> str:
        """Get session file path for model."""
        return os.path.join(_SESSION_DIR, f"model_{model_id}.pkl")

    def _load_session_state(self) -> Dict[str, Any]:
        """Load persisted models from session directory."""
        import pickle
        models = {}
        if os.path.exists(_SESSION_DIR):
            for fname in os.listdir(_SESSION_DIR):
                if fname.startswith("model_") and fname.endswith(".pkl"):
                    try:
                        fpath = os.path.join(_SESSION_DIR, fname)
                        with open(fpath, "rb") as f:
                            data = pickle.load(f)
                            # Extract model ID from filename instead of relying on pickled value
                            extracted_id = fname.replace("model_", "").replace(".pkl", "")
                            # Reconstruct model data with consistent ID
                            models[extracted_id] = {
                                "model_id": extracted_id,
                                "type": data.get("type", "Unknown"),
                                "config": data.get("config", {}),
                                "state_dict": data.get("state_dict", {}),
                            }
                    except Exception:
                        pass
        return models

    def _save_session_model(self, model_id: str, model_data: Dict[str, Any]) -> None:
        """Save model to session storage."""
        import pickle
        try:
            # Save without the model object to avoid pickling issues
            save_data = {
                "model_id": model_id,
                "type": model_data.get("type"),
                "config": model_data.get("config", {}),
                "state_dict": model_data.get("model").state_dict() if model_data.get("model") else {},
            }
            self.models[model_id] = model_data
            with open(self._get_session_file(model_id), "wb") as f:
                pickle.dump(save_data, f)
        except Exception:
            # Silently fail, session isn't critical
            pass

    def _state_dict_to_numpy(self, state_dict: Any) -> Dict[str, np.ndarray]:
        """Convert a state_dict to CPU numpy arrays."""
        out: Dict[str, np.ndarray] = {}
        if not isinstance(state_dict, dict):
            return out

        for k, v in state_dict.items():
            try:
                # torch Tensor
                if hasattr(v, "detach") and hasattr(v, "cpu") and hasattr(v, "numpy"):
                    out[str(k)] = np.asarray(v.detach().cpu().numpy(), dtype=np.float32)
                else:
                    out[str(k)] = np.asarray(v, dtype=np.float32)
            except Exception:
                continue
        return out

    def _apply_state_dict(self, model: Any, state_dict: Any) -> Dict[str, Any]:
        """Best-effort state_dict application onto a rasptorch.nn.Module.

        rasptorch modules do not currently expose load_state_dict, so we map by
        parameter name as returned by Module.named_parameters().
        """
        sd = self._state_dict_to_numpy(state_dict)
        if not sd:
            return {"applied": 0, "missing": 0}

        applied = 0
        missing = 0
        try:
            named_params = list(model.named_parameters()) if hasattr(model, "named_parameters") else []
        except Exception:
            named_params = []

        def _set_param_by_path(root: Any, dotted: str, new_param: Any) -> bool:
            """Set a Parameter on a module by its dotted name."""
            parts = str(dotted).split(".")
            if not parts:
                return False
            cur = root
            for i, part in enumerate(parts[:-1]):
                if part.isdigit():
                    idx = int(part)
                    if isinstance(cur, (list, tuple)):
                        cur = cur[idx]
                    else:
                        return False
                else:
                    nxt = getattr(cur, part, None)
                    if nxt is None:
                        return False
                    cur = nxt

            last = parts[-1]
            if last.isdigit():
                idx = int(last)
                if isinstance(cur, list):
                    cur[idx] = new_param
                    return True
                return False

            setattr(cur, last, new_param)
            return True

        for name, param in named_params:
            key = str(name)
            if key not in sd:
                missing += 1
                continue
            try:
                arr = np.asarray(sd[key], dtype=np.float32)
                if hasattr(param, "shape") and tuple(getattr(param, "shape")) != tuple(arr.shape):
                    # Skip incompatible shapes.
                    continue
                if getattr(param, "device", "cpu") == "gpu":
                    # Replace GPU param with a CPU param holding the loaded weights.
                    try:
                        param_cpu = param.to("cpu")
                        param_cpu.data = arr.copy()
                        if not _set_param_by_path(model, key, param_cpu):
                            continue
                    except Exception:
                        continue
                else:
                    param.data = arr.copy()
                applied += 1
            except Exception:
                continue

        return {"applied": applied, "missing": missing}

    def _ensure_model(self, model_id: str) -> Any:
        """Ensure `self.models[model_id]['model']` exists and has weights applied."""
        model_data = self.models.get(model_id)
        if not model_data:
            return None
        model = model_data.get("model")
        if model is None:
            model_type = model_data.get("type", "Unknown")
            config = model_data.get("config", {})
            # Combined models need to be reconstructed from their component models.
            if model_type == "Combined" and isinstance(config, dict) and str(config.get("combine")) == "sequential":
                model_a_id = str(config.get("model_a_id", ""))
                model_b_id = str(config.get("model_b_id", ""))
                if not model_a_id or not model_b_id:
                    # Missing linkage data; cannot reconstruct.
                    return None
                # Prefer reconstructing from existing session models.
                model_a = self._ensure_model(model_a_id) if model_a_id in self.models else None
                model_b = self._ensure_model(model_b_id) if model_b_id in self.models else None

                # If components are missing from session, try embedded snapshots.
                if model_a is None or model_b is None:
                    try:
                        a_snap = config.get("model_a_snapshot") or {}
                        b_snap = config.get("model_b_snapshot") or {}
                        if model_a is None and isinstance(a_snap, dict) and a_snap.get("type"):
                            ma = self._reconstruct_model(str(a_snap.get("type")), a_snap.get("config") or {})
                            if ma is not None and a_snap.get("state_dict"):
                                self._apply_state_dict(ma, a_snap.get("state_dict"))
                            model_a = ma
                        if model_b is None and isinstance(b_snap, dict) and b_snap.get("type"):
                            mb = self._reconstruct_model(str(b_snap.get("type")), b_snap.get("config") or {})
                            if mb is not None and b_snap.get("state_dict"):
                                self._apply_state_dict(mb, b_snap.get("state_dict"))
                            model_b = mb
                    except Exception:
                        pass

                if model_a is None or model_b is None:
                    return None

                combined_layers = self._flatten_layers(model_a) + self._flatten_layers(model_b)
                if not combined_layers:
                    return None

                normalized_layers: List[Any] = []
                for layer in combined_layers:
                    if hasattr(rasptorch.nn, "GRU") and isinstance(layer, rasptorch.nn.GRU):
                        normalized_layers.append(_GRULastHidden(layer))
                    elif hasattr(rasptorch.nn, "MultiheadAttention") and isinstance(layer, rasptorch.nn.MultiheadAttention):
                        normalized_layers.append(_SelfAttention(layer))
                    else:
                        normalized_layers.append(layer)

                model = rasptorch.nn.Sequential(*normalized_layers)
            else:
                model = self._reconstruct_model(model_type, config)
            if model is None:
                return None
            # Apply saved weights if present.
            if model_data.get("state_dict"):
                self._apply_state_dict(model, model_data.get("state_dict"))
            model_data["model"] = model
        return model

    def _reconstruct_model(self, model_type: str, config: Dict[str, Any]) -> Any:
        """Reconstruct model from configuration."""
        def _act_layer(name: str) -> Optional[Any]:
            return self._make_activation_layer(name)

        if model_type == "MLP":
            layer_sizes = config.get("layer_sizes", [64, 32, 2])
            activation = config.get("activation", "relu")
            activations = config.get("activations")
            layers = []
            for i in range(len(layer_sizes) - 1):
                layers.append(rasptorch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                if i < len(layer_sizes) - 2:
                    act_name = None
                    if isinstance(activations, list) and i < len(activations):
                        act_name = str(activations[i])
                    else:
                        act_name = str(activation)
                    act = _act_layer(act_name)
                    if act is not None:
                        layers.append(act)
            return rasptorch.nn.Sequential(*layers)
        elif model_type == "Linear":
            input_size = config.get("input_size", 10)
            hidden_sizes = config.get("hidden_sizes", [32])
            output_size = config.get("output_size", 2)
            activation = config.get("activation", "relu")
            activations = config.get("activations")
            layers = []
            prev_size = input_size
            for idx, hidden_size in enumerate(hidden_sizes):
                layers.append(rasptorch.nn.Linear(prev_size, hidden_size))
                act_name = None
                if isinstance(activations, list) and idx < len(activations):
                    act_name = str(activations[idx])
                else:
                    act_name = str(activation)
                act = _act_layer(act_name)
                if act is not None:
                    layers.append(act)
                prev_size = hidden_size
            layers.append(rasptorch.nn.Linear(prev_size, output_size))
            return rasptorch.nn.Sequential(*layers)
        elif model_type == "CNN":
            in_channels = config.get("in_channels", 3)
            out_channels = config.get("out_channels", [32, 64])
            activation = config.get("activation", "relu")
            activations = config.get("activations")
            layers = []
            in_ch = in_channels
            for i, out_ch in enumerate(out_channels):
                layers.append(rasptorch.nn.Linear(in_ch, out_ch))
                if i < len(out_channels) - 1:
                    act_name = None
                    if isinstance(activations, list) and i < len(activations):
                        act_name = str(activations[i])
                    else:
                        act_name = str(activation)
                    act = _act_layer(act_name)
                    if act is not None:
                        layers.append(act)
                in_ch = out_ch
            return rasptorch.nn.Sequential(*layers)
        elif model_type == "GRU":
            input_size = config.get("input_size", 128)
            hidden_size = config.get("hidden_size", 256)
            num_layers = config.get("num_layers", 1)
            if hasattr(rasptorch.nn, "GRU"):
                if int(num_layers) <= 1:
                    return rasptorch.nn.GRU(input_size, hidden_size, batch_first=True)

                layers: List[Any] = []
                layers.append(_GRUSequence(rasptorch.nn.GRU(input_size, hidden_size, batch_first=True)))
                for _ in range(int(num_layers) - 2):
                    layers.append(_GRUSequence(rasptorch.nn.GRU(hidden_size, hidden_size, batch_first=True)))
                layers.append(_GRULastHidden(rasptorch.nn.GRU(hidden_size, hidden_size, batch_first=True)))
                return rasptorch.nn.Sequential(*layers)

            layers = []
            for i in range(num_layers):
                in_size = input_size if i == 0 else hidden_size
                layers.append(rasptorch.nn.Linear(in_size, hidden_size * 3))
            return rasptorch.nn.Sequential(*layers)
        else:
            return None

    def _normalize_activation_name(self, activation: str) -> str:
        name = str(activation).strip().lower().replace("-", "_")
        if name in {"", "none", "null", "no", "identity"}:
            return "none"
        if name in {"leaky", "leakyrelu"}:
            return "leaky_relu"
        if name in {"swish"}:
            return "silu"
        return name

    def _make_activation_layer(self, activation: str) -> Optional[Any]:
        name = self._normalize_activation_name(activation)
        if name == "none":
            return None
        if name == "relu":
            return rasptorch.nn.ReLU()
        if name == "gelu":
            return rasptorch.nn.GELU()
        if name == "tanh":
            return rasptorch.nn.Tanh()
        if name == "sigmoid":
            return rasptorch.nn.Sigmoid()
        if name == "silu":
            return rasptorch.nn.SiLU()
        if name == "leaky_relu":
            return rasptorch.nn.LeakyReLU()
        if name == "elu":
            return rasptorch.nn.ELU()
        raise ValueError(f"Unknown activation: {activation}")

    def _flatten_layers(self, model: Any) -> List[Any]:
        """Best-effort flattening of Sequential-like modules into a list of layers."""
        if model is None:
            return []

        # rasptorch.nn.Sequential stores layers in `.layers`.
        layers = getattr(model, "layers", None)
        if isinstance(layers, list):
            flat: List[Any] = []
            for layer in layers:
                flat.extend(self._flatten_layers(layer))
            return flat

        return [model]

    def _infer_linear_in_out(self, model: Any) -> Tuple[Optional[int], Optional[int]]:
        """Infer (in_features, out_features) by scanning Linear layers, if possible."""
        flat = self._flatten_layers(model)

        first_in: Optional[int] = None
        last_out: Optional[int] = None

        for layer in flat:
            try:
                if isinstance(layer, rasptorch.nn.Linear):
                    # Linear.weight: [out, in]
                    if first_in is None:
                        first_in = int(layer.weight.shape[1])
                    last_out = int(layer.weight.shape[0])
            except Exception:
                continue

        return first_in, last_out

    def _infer_io_spec(self, model_id: str) -> Dict[str, Any]:
        """Best-effort input/output spec inference for a model.

        Returns a dict:
          - type: model type string
          - in_features/out_features: int|None (feature dimension for rank-2 inputs/outputs)
          - in_ranks: set of allowed input ranks (e.g. {2} or {2,3})
          - out_rank: int (currently always 2 for CLI-composed models)

        This is used for combine-time compatibility checks.
        """
        model_data = self.models.get(model_id) or {}
        model_type = str(model_data.get("type", "Unknown"))
        config = model_data.get("config") or {}

        in_ranks = {2}
        out_rank = 2
        in_features: Optional[int] = None
        out_features: Optional[int] = None

        try:
            if model_type == "MLP":
                layer_sizes = config.get("layer_sizes")
                if isinstance(layer_sizes, list) and len(layer_sizes) >= 2:
                    in_features = int(layer_sizes[0])
                    out_features = int(layer_sizes[-1])
            elif model_type == "Linear":
                if "input_size" in config:
                    in_features = int(config.get("input_size"))
                if "output_size" in config:
                    out_features = int(config.get("output_size"))
            elif model_type == "CNN":
                # CLI "CNN" is currently an MLP-like stack; treat channels as features.
                if "in_channels" in config:
                    in_features = int(config.get("in_channels"))
                out_ch = config.get("out_channels")
                if isinstance(out_ch, list) and out_ch:
                    out_features = int(out_ch[-1])
            elif model_type == "GRU":
                # Wrapper allows 2D by treating as (B,1,E), so accept ranks {2,3}.
                in_ranks = {2, 3}
                if "input_size" in config:
                    in_features = int(config.get("input_size"))
                if "hidden_size" in config:
                    out_features = int(config.get("hidden_size"))
            elif model_type == "Transformer":
                # Transformer as built here: Linear(vocab_size->d_model) ... -> d_model
                if "vocab_size" in config:
                    in_features = int(config.get("vocab_size"))
                if "d_model" in config:
                    out_features = int(config.get("d_model"))
            elif model_type == "Combined":
                # Prefer stored metadata if present.
                if "input_size" in config:
                    in_features = int(config.get("input_size"))
                if "output_size" in config:
                    out_features = int(config.get("output_size"))
        except Exception:
            # Fall through to inference.
            pass

        # Fallback: scan Linear layers if we still don't know.
        if in_features is None or out_features is None:
            try:
                model = self._ensure_model(model_id)
                lin_in, lin_out = self._infer_linear_in_out(model)
                if in_features is None and lin_in is not None:
                    in_features = int(lin_in)
                if out_features is None and lin_out is not None:
                    out_features = int(lin_out)
            except Exception:
                pass

        return {
            "type": model_type,
            "in_features": in_features,
            "out_features": out_features,
            "in_ranks": set(in_ranks),
            "out_rank": int(out_rank),
        }

    def _validate_combine_compatibility(self, model_a_id: str, model_b_id: str) -> Tuple[bool, str, Dict[str, Any], Dict[str, Any]]:
        """Validate that model A's output can feed model B's input.

        Returns (ok, reason, a_spec, b_spec).
        """
        a_spec = self._infer_io_spec(model_a_id)
        b_spec = self._infer_io_spec(model_b_id)

        a_type = a_spec.get("type", "Unknown")
        b_type = b_spec.get("type", "Unknown")

        a_out_rank = int(a_spec.get("out_rank", 2))
        b_in_ranks = b_spec.get("in_ranks") or {2}

        if a_out_rank not in b_in_ranks:
            return (
                False,
                f"Incompatible tensor rank: {a_type} outputs rank {a_out_rank} but {b_type} expects rank {sorted(b_in_ranks)}",
                a_spec,
                b_spec,
            )

        a_out = a_spec.get("out_features")
        b_in = b_spec.get("in_features")

        if a_out is None:
            return (
                False,
                f"Cannot verify compatibility: {a_type} output size is unknown (missing config/Linear layers)",
                a_spec,
                b_spec,
            )
        if b_in is None:
            return (
                False,
                f"Cannot verify compatibility: {b_type} input size is unknown (missing config/Linear layers)",
                a_spec,
                b_spec,
            )

        if int(a_out) != int(b_in):
            return (
                False,
                f"Incompatible feature size: {a_type} outputs {int(a_out)} but {b_type} expects {int(b_in)}",
                a_spec,
                b_spec,
            )

        return True, "", a_spec, b_spec

    def combine_models(self, model_a_id: str, model_b_id: str) -> Dict[str, Any]:
        """Combine two existing models sequentially (A then B) into a new model."""
        if model_a_id not in self.models:
            return {"error": f"Model {model_a_id} not found"}
        if model_b_id not in self.models:
            return {"error": f"Model {model_b_id} not found"}

        try:
            model_a_data = self.models[model_a_id]
            model_b_data = self.models[model_b_id]

            model_a = self._ensure_model(model_a_id)
            if model_a is None:
                return {"error": f"Cannot reconstruct {model_a_data.get('type', 'Unknown')} model"}
            model_b = self._ensure_model(model_b_id)
            if model_b is None:
                return {"error": f"Cannot reconstruct {model_b_data.get('type', 'Unknown')} model"}

            ok, reason, a_spec, b_spec = self._validate_combine_compatibility(model_a_id, model_b_id)
            if not ok:
                return {
                    "error": (
                        f"Cannot combine {model_a_data.get('type', 'Unknown')}[{model_a_id[:8]}] -> "
                        f"{model_b_data.get('type', 'Unknown')}[{model_b_id[:8]}]: {reason}"
                    )
                }

            combined_layers = self._flatten_layers(model_a) + self._flatten_layers(model_b)
            if not combined_layers:
                return {"error": "Failed to combine models (no layers found)"}

            normalized_layers: List[Any] = []
            for layer in combined_layers:
                if hasattr(rasptorch.nn, "GRU") and isinstance(layer, rasptorch.nn.GRU):
                    normalized_layers.append(_GRULastHidden(layer))
                elif hasattr(rasptorch.nn, "MultiheadAttention") and isinstance(layer, rasptorch.nn.MultiheadAttention):
                    normalized_layers.append(_SelfAttention(layer))
                else:
                    normalized_layers.append(layer)

            combined_model = rasptorch.nn.Sequential(*normalized_layers)
            combined_id = str(uuid.uuid4())[:8]

            combined_config: Dict[str, Any] = {
                "combine": "sequential",
                "model_a_id": model_a_id,
                "model_b_id": model_b_id,
                "model_a_type": model_a_data.get("type", "Unknown"),
                "model_b_type": model_b_data.get("type", "Unknown"),
            }
            # Embed minimal snapshots of components so Combined can be reconstructed
            # even if individual session files are missing later.
            try:
                combined_config["model_a_snapshot"] = {
                    "type": model_a_data.get("type", "Unknown"),
                    "config": model_a_data.get("config", {}),
                    "state_dict": model_a_data.get("model").state_dict() if model_a_data.get("model") else model_a_data.get("state_dict", {}),
                }
                combined_config["model_b_snapshot"] = {
                    "type": model_b_data.get("type", "Unknown"),
                    "config": model_b_data.get("config", {}),
                    "state_dict": model_b_data.get("model").state_dict() if model_b_data.get("model") else model_b_data.get("state_dict", {}),
                }
            except Exception:
                pass
            # Persist inferred IO sizes for downstream training / info.
            if a_spec.get("in_features") is not None:
                combined_config["input_size"] = int(a_spec["in_features"])
            if b_spec.get("out_features") is not None:
                combined_config["output_size"] = int(b_spec["out_features"])

            model_data = {
                "model_id": combined_id,
                "model": combined_model,
                "type": "Combined",
                "config": combined_config,
            }
            self._save_session_model(combined_id, model_data)

            return {
                "status": "success",
                "model_id": combined_id,
                "type": "Combined",
                "message": "Models combined sequentially",
                "combined_from": {
                    "model_a_id": model_a_id,
                    "model_b_id": model_b_id,
                    "model_a_type": model_a_data.get("type", "Unknown"),
                    "model_b_type": model_b_data.get("type", "Unknown"),
                },
                "architecture": {
                    "combine": "sequential",
                    "input_size": a_spec.get("in_features"),
                    "output_size": b_spec.get("out_features"),
                    "num_layers": len(normalized_layers),
                },
            }
        except Exception as e:
            return {"error": str(e)}

    def create_linear_model(
        self,
        input_size: int,
        hidden_sizes: list,
        output_size: int,
        activation: str = "relu",
        activations: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a linear model."""
        try:
            if isinstance(activations, list) and len(activations) != len(hidden_sizes):
                return {
                    "error": f"activations must have {len(hidden_sizes)} value(s) (one per hidden layer)"
                }
            layers = []
            prev_size = input_size
            for idx, hidden_size in enumerate(hidden_sizes):
                layers.append(rasptorch.nn.Linear(prev_size, hidden_size))
                act_name = activations[idx] if isinstance(activations, list) and idx < len(activations) else activation
                act = self._make_activation_layer(act_name)
                if act is not None:
                    layers.append(act)
                prev_size = hidden_size
            layers.append(rasptorch.nn.Linear(prev_size, output_size))
            model = rasptorch.nn.Sequential(*layers)
            model_id = str(uuid.uuid4())[:8]
            
            model_data = {
                "model_id": model_id,
                "model": model,
                "type": "Linear",
                "config": {
                    "input_size": input_size,
                    "hidden_sizes": hidden_sizes,
                    "output_size": output_size,
                    "activation": self._normalize_activation_name(activation),
                    **({"activations": [self._normalize_activation_name(a) for a in activations]} if isinstance(activations, list) else {}),
                },
            }
            self._save_session_model(model_id, model_data)
            
            return {
                "status": "success",
                "model_id": model_id,
                "architecture": {
                    "input_size": input_size,
                    "hidden_sizes": hidden_sizes,
                    "output_size": output_size,
                    "num_layers": len(layers),
                }
            }
        except Exception as e:
            return {"error": str(e)}

    def create_mlp(
        self,
        layer_sizes: List[int],
        activation: str = "relu",
        activations: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a Multi-Layer Perceptron."""
        if len(layer_sizes) < 2:
            return {"error": "MLP requires at least 2 layers"}
        try:
            expected = max(len(layer_sizes) - 2, 0)
            if isinstance(activations, list) and len(activations) != expected:
                return {
                    "error": f"activations must have {expected} value(s) (one per hidden transition)"
                }
            layers = []
            for i in range(len(layer_sizes) - 1):
                layers.append(rasptorch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                if i < len(layer_sizes) - 2:
                    act_name = activations[i] if isinstance(activations, list) and i < len(activations) else activation
                    act = self._make_activation_layer(act_name)
                    if act is not None:
                        layers.append(act)
            model = rasptorch.nn.Sequential(*layers)
            model_id = str(uuid.uuid4())[:8]
            
            model_data = {
                "model_id": model_id,
                "model": model,
                "type": "MLP",
                "config": {
                    "layer_sizes": layer_sizes,
                    "activation": self._normalize_activation_name(activation),
                    **({"activations": [self._normalize_activation_name(a) for a in activations]} if isinstance(activations, list) else {}),
                },
            }
            self._save_session_model(model_id, model_data)
            
            return {
                "status": "success",
                "model_id": model_id,
                "type": "MLP",
                "architecture": {
                    "layers": len(layer_sizes),
                    "layer_sizes": layer_sizes,
                }
            }
        except Exception as e:
            return {"error": str(e)}

    def create_cnn(
        self,
        in_channels: int,
        out_channels: List[int],
        kernel_sizes: List[int] = None,
        activation: str = "relu",
        activations: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a CNN."""
        if not out_channels:
            return {"error": "CNN requires output channels"}
        try:
            expected = max(len(out_channels) - 1, 0)
            if isinstance(activations, list) and len(activations) != expected:
                return {
                    "error": f"activations must have {expected} value(s) (one per transition)"
                }
            if kernel_sizes is None:
                kernel_sizes = [3] * len(out_channels)
            layers = []
            in_ch = in_channels
            for i, out_ch in enumerate(out_channels):
                layers.append(rasptorch.nn.Linear(in_ch, out_ch))
                if i < len(out_channels) - 1:
                    act_name = activations[i] if isinstance(activations, list) and i < len(activations) else activation
                    act = self._make_activation_layer(act_name)
                    if act is not None:
                        layers.append(act)
                in_ch = out_ch
            model = rasptorch.nn.Sequential(*layers)
            model_id = str(uuid.uuid4())[:8]
            
            model_data = {
                "model_id": model_id,
                "model": model,
                "type": "CNN",
                "config": {
                    "in_channels": in_channels,
                    "out_channels": out_channels,
                    "kernel_sizes": kernel_sizes,
                    "activation": self._normalize_activation_name(activation),
                    **({"activations": [self._normalize_activation_name(a) for a in activations]} if isinstance(activations, list) else {}),
                },
            }
            self._save_session_model(model_id, model_data)
            
            return {
                "status": "success",
                "model_id": model_id,
                "type": "CNN",
                "architecture": {
                    "in_channels": in_channels,
                    "out_channels": out_channels,
                    "num_layers": len(out_channels),
                }
            }
        except Exception as e:
            return {"error": str(e)}

    def create_gru(self, input_size: int, hidden_size: int, num_layers: int = 1) -> Dict[str, Any]:
        """Create a GRU model."""
        try:
            if hasattr(rasptorch.nn, "GRU"):
                if int(num_layers) <= 1:
                    model = rasptorch.nn.GRU(input_size, hidden_size, batch_first=True)
                else:
                    layers: List[Any] = []
                    layers.append(_GRUSequence(rasptorch.nn.GRU(input_size, hidden_size, batch_first=True)))
                    for _ in range(int(num_layers) - 2):
                        layers.append(_GRUSequence(rasptorch.nn.GRU(hidden_size, hidden_size, batch_first=True)))
                    layers.append(_GRULastHidden(rasptorch.nn.GRU(hidden_size, hidden_size, batch_first=True)))
                    model = rasptorch.nn.Sequential(*layers)
            else:
                layers = []
                for i in range(num_layers):
                    in_size = input_size if i == 0 else hidden_size
                    layers.append(rasptorch.nn.Linear(in_size, hidden_size * 3))
                model = rasptorch.nn.Sequential(*layers)
            model_id = str(uuid.uuid4())[:8]
            
            model_data = {
                "model_id": model_id,
                "model": model,
                "type": "GRU",
                "config": {
                    "input_size": input_size,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                },
            }
            self._save_session_model(model_id, model_data)
            
            return {
                "status": "success",
                "model_id": model_id,
                "type": "GRU",
                "architecture": {
                    "input_size": input_size,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                }
            }
        except Exception as e:
            return {"error": str(e)}

    def create_transformer(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int) -> Dict[str, Any]:
        """Create a Transformer model."""
        try:
            if not hasattr(rasptorch.nn, "MultiheadAttention"):
                return {"error": "Transformer requires MultiheadAttention support"}
            layers = [rasptorch.nn.Linear(vocab_size, d_model)]
            for _ in range(num_layers):
                layers.append(_SelfAttention(rasptorch.nn.MultiheadAttention(d_model, num_heads)))
                layers.append(rasptorch.nn.Linear(d_model, d_model * 4))
                layers.append(rasptorch.nn.ReLU())
                layers.append(rasptorch.nn.Linear(d_model * 4, d_model))
            model = rasptorch.nn.Sequential(*layers)
            model_id = str(uuid.uuid4())[:8]
            
            model_data = {
                "model_id": model_id,
                "model": model,
                "type": "Transformer",
                "config": {
                    "vocab_size": vocab_size,
                    "d_model": d_model,
                    "num_heads": num_heads,
                    "num_layers": num_layers,
                },
            }
            self._save_session_model(model_id, model_data)
            
            return {
                "status": "success",
                "model_id": model_id,
                "type": "Transformer",
                "architecture": {
                    "vocab_size": vocab_size,
                    "d_model": d_model,
                    "num_heads": num_heads,
                    "num_layers": num_layers,
                }
            }
        except Exception as e:
            return {"error": str(e)}

    def create_lora_adapter(self, model_id: str, rank: int = 8, alpha: float = 16.0) -> Dict[str, Any]:
        """Create LoRA adapter."""
        if model_id not in self.models:
            return {"error": f"Model {model_id} not found"}
        try:
            adapter_id = str(id({}))
            self.lora_adapters[adapter_id] = {
                "base_model_id": model_id,
                "rank": rank,
                "alpha": alpha,
                "scaling": alpha / rank,
            }
            return {
                "status": "success",
                "adapter_id": adapter_id,
                "rank": rank,
                "alpha": alpha,
                "scaling": alpha / rank,
            }
        except Exception as e:
            return {"error": str(e)}

    def create_optimizer(self, model_id: str, optimizer_type: str = "Adam", lr: float = 0.001) -> Dict[str, Any]:
        """Create optimizer."""
        if model_id not in self.models:
            return {"error": f"Model {model_id} not found"}
        try:
            model = self.models[model_id]["model"]
            opt = str(optimizer_type).strip().lower().replace("-", "").replace("_", "")
            if opt == "adam":
                optimizer = rasptorch.Adam(model.parameters(), lr=lr)
            elif opt == "adamw":
                optimizer = rasptorch.AdamW(model.parameters(), lr=lr)
            elif opt == "sgd":
                optimizer = rasptorch.SGD(model.parameters(), lr=lr)
            elif opt == "rmsprop":
                optimizer = rasptorch.RMSProp(model.parameters(), lr=lr)
            else:
                return {"error": f"Unknown optimizer: {optimizer_type}"}
            optimizer_id = str(id(optimizer))
            self.optimizers[optimizer_id] = {
                "optimizer": optimizer,
                "model_id": model_id,
                "type": optimizer_type,
                "lr": lr,
            }
            return {
                "status": "success",
                "optimizer_id": optimizer_id,
                "type": optimizer_type,
                "learning_rate": lr,
            }
        except Exception as e:
            return {"error": str(e)}

    def list_models(self) -> Dict[str, Any]:
        """List all models."""
        models_info = []
        for model_id, model_data in self.models.items():
            models_info.append({
                "model_id": model_id[:8],
                "type": model_data.get("type", "Unknown"),
            })
        return {
            "status": "success",
            "models": models_info,
            "total": len(models_info),
        }

    def delete_model(self, model_id: str) -> Dict[str, Any]:
        """Delete a model from session storage."""
        if model_id not in self.models:
            return {"error": f"Model {model_id} not found"}
        try:
            # Remove from dictionary
            del self.models[model_id]
            
            # Remove session file
            session_file = self._get_session_file(model_id)
            if os.path.exists(session_file):
                os.remove(session_file)
            
            return {
                "status": "success",
                "message": f"Model {model_id} deleted successfully",
                "model_id": model_id,
            }
        except Exception as e:
            return {"error": str(e)}

    def save_model(self, model_id: str, path: str) -> Dict[str, Any]:
        """Save model to file."""
        if model_id not in self.models:
            return {"error": f"Model {model_id} not found"}
        try:
            model_data = self.models[model_id]
            model = self._ensure_model(model_id)
            if model is None:
                model_type = model_data.get("type", "Unknown")
                if model_type == "Combined":
                    cfg2 = model_data.get("config", {}) or {}
                    a_id = str(cfg2.get("model_a_id", ""))
                    b_id = str(cfg2.get("model_b_id", ""))
                    missing: List[str] = []
                    if not a_id:
                        missing.append("model_a_id")
                    if not b_id:
                        missing.append("model_b_id")
                    if a_id and a_id not in self.models:
                        missing.append(f"A not found: {a_id}")
                    if b_id and b_id not in self.models:
                        missing.append(f"B not found: {b_id}")

                    # If IDs exist, try to ensure each side to identify which reconstruction failed.
                    if a_id and a_id in self.models:
                        try:
                            if self._ensure_model(a_id) is None:
                                a_type = (self.models.get(a_id) or {}).get("type", "Unknown")
                                missing.append(f"A cannot reconstruct: {a_type}[{a_id[:8]}]")
                        except Exception as ex:
                            missing.append(f"A reconstruct error: {type(ex).__name__}")
                    if b_id and b_id in self.models:
                        try:
                            if self._ensure_model(b_id) is None:
                                b_type = (self.models.get(b_id) or {}).get("type", "Unknown")
                                missing.append(f"B cannot reconstruct: {b_type}[{b_id[:8]}]")
                        except Exception as ex:
                            missing.append(f"B reconstruct error: {type(ex).__name__}")
                    extra = f" (missing: {', '.join(missing)})" if missing else ""
                    return {"error": f"Cannot reconstruct Combined model{extra}"}
                return {"error": f"Cannot reconstruct {model_type} model"}

            state_dict = model.state_dict()

            save_data = {
                "model_type": model_data.get("type", "Unknown"),
                "config": model_data.get("config", {}),
                "state_dict": state_dict,
            }

            ext = os.path.splitext(str(path))[1].lower()
            if ext in {".pth", ".pt"}:
                try:
                    import torch  # type: ignore
                except Exception as e:
                    return {"error": f"Saving {ext} requires torch (import failed: {e})"}

                def _torch_safe(obj: Any) -> Any:
                    # Make payload compatible with torch.load(weights_only=True).
                    if isinstance(obj, np.ndarray):
                        return torch.from_numpy(np.asarray(obj, dtype=np.float32))
                    if isinstance(obj, (np.floating, np.integer)):
                        return obj.item()
                    if isinstance(obj, dict):
                        return {str(k): _torch_safe(v) for k, v in obj.items()}
                    if isinstance(obj, (list, tuple)):
                        return type(obj)(_torch_safe(v) for v in obj)
                    return obj

                torch.save(_torch_safe(save_data), path)
                fmt = "torch"
            else:
                import pickle
                with open(path, "wb") as f:
                    pickle.dump(save_data, f)
                fmt = "pickle"
            
            return {
                "status": "success",
                "model_id": model_id,
                "path": path,
                "format": fmt,
                "message": f"Model saved successfully",
            }
        except Exception as e:
            return {"error": str(e)}

    def _safe_model_path(self, path: str) -> str:
        """Resolve user-provided path to a safe location under a fixed root directory."""
        # Fixed base directory for all model files.
        base_dir = os.path.abspath(os.path.join(tempfile.gettempdir(), "rasptorch_models"))
        os.makedirs(base_dir, exist_ok=True)

        # Normalize the user-provided path string.
        path = (path or "").strip()
        if not path:
            raise ValueError("Invalid model path")

        # Disallow arbitrary absolute paths by treating them as simple filenames.
        if os.path.isabs(path):
            path = os.path.basename(path)

        # Build a normalized absolute path under the base directory.
        candidate = os.path.abspath(os.path.normpath(os.path.join(base_dir, path)))

        # Ensure the resulting path is within the base directory.
        if os.path.commonpath([base_dir, candidate]) != base_dir:
            raise ValueError("Invalid model path")

        return candidate

    def load_model(self, path: str) -> Dict[str, Any]:
        """Load model from file."""
        try:
            safe_path = self._safe_model_path(str(path))
            ext = os.path.splitext(str(safe_path))[1].lower()
            unsafe_load = False
            if ext in {".pth", ".pt"}:
                try:
                    import torch  # type: ignore
                except Exception as e:
                    return {"error": f"Loading {ext} requires torch (import failed: {e})"}
                try:
                    save_data = torch.load(safe_path, map_location="cpu")
                except Exception as e:
                    msg = str(e)
                    # Back-compat: handle older files that contained NumPy arrays.
                    if "Weights only load failed" in msg or "weights_only" in msg:
                        try:
                            save_data = torch.load(safe_path, map_location="cpu", weights_only=False)
                            unsafe_load = True
                        except TypeError:
                            raise
                    else:
                        raise
                fmt = "torch"
            else:
                import pickle
                with open(safe_path, "rb") as f:
                    save_data = pickle.load(f)
                fmt = "pickle"
            
            model_type = save_data.get("model_type", "Unknown")
            config = save_data.get("config", {})
            state_dict = self._state_dict_to_numpy(save_data.get("state_dict", {}))

            model_id = str(uuid.uuid4())[:8]
            self.models[model_id] = {
                "model": None,
                "type": model_type,
                "config": config,
                "state_dict": state_dict,
            }
            
            return {
                "status": "success",
                "model_id": model_id,
                "model_type": model_type,
                "format": fmt,
                **({"unsafe_load": True} if unsafe_load else {}),
                "message": f"Model loaded successfully",
            }
        except Exception as e:
            return {"error": str(e)}

    def train_model(
        self,
        model_id: str,
        epochs: int = 10,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        device: str = "cpu",
        optimizer_type: str = "Adam",
    ) -> Dict[str, Any]:
        """Train a model."""
        if model_id not in self.models:
            return {"error": f"Model {model_id} not found"}
        
        try:
            model_data = self.models[model_id]
            model = self._ensure_model(model_id)
            if model is None:
                model_type = model_data.get("type", "Unknown")
                return {"error": f"Cannot reconstruct {model_type} model"}
            
            # Move model to device if needed
            if device == "gpu":
                try:
                    model.to("gpu")
                except Exception as e:
                    return {"error": f"Failed to move model to GPU: {str(e)}"}
            
            # Create optimizer
            opt = str(optimizer_type).strip().lower().replace("-", "").replace("_", "")
            if opt == "adam":
                optimizer = rasptorch.Adam(model.parameters(), lr=learning_rate)
            elif opt == "adamw":
                optimizer = rasptorch.AdamW(model.parameters(), lr=learning_rate)
            elif opt == "sgd":
                optimizer = rasptorch.SGD(model.parameters(), lr=learning_rate)
            elif opt == "rmsprop":
                optimizer = rasptorch.RMSProp(model.parameters(), lr=learning_rate)
            else:
                return {"error": f"Unknown optimizer: {optimizer_type}"}
            
            # Infer input size from model config
            config = model_data.get("config", {})

            def _find_input_gru(m: Any) -> Optional[Any]:
                """Return a GRU that is on the model input path (first layer), if any."""
                if hasattr(rasptorch.nn, "GRU") and isinstance(m, rasptorch.nn.GRU):
                    return m
                if hasattr(m, "gru") and hasattr(rasptorch.nn, "GRU") and isinstance(getattr(m, "gru"), rasptorch.nn.GRU):
                    return getattr(m, "gru")
                layers = getattr(m, "layers", None)
                if isinstance(layers, list) and layers:
                    return _find_input_gru(layers[0])
                return None

            def _forward_for_loss(m: Any, x: Any) -> Any:
                out = m(x)
                if hasattr(rasptorch.nn, "GRU") and isinstance(m, rasptorch.nn.GRU) and isinstance(out, tuple) and len(out) == 2:
                    _o, h = out
                    return h[0]
                if isinstance(out, tuple):
                    for item in out:
                        if hasattr(item, "shape"):
                            return item
                    return out[0]
                return out

            first_gru = _find_input_gru(model)
            if first_gru is not None:
                seq_len = int(config.get("seq_len") or config.get("sequence_length") or 8)
                inferred_in = getattr(first_gru, "input_size", None)
                input_size = int(inferred_in if inferred_in is not None else (config.get("input_size") or 128))
                input_shape = (batch_size, seq_len, input_size)
            else:
                input_size = int(config.get("input_size") or config.get("vocab_size") or config.get("layer_sizes", [10])[0])
                input_shape = (batch_size, input_size)

            X_probe = rasptorch.Tensor(np.random.randn(*input_shape).astype(np.float32), device=device)
            y_probe = _forward_for_loss(model, X_probe)
            if not hasattr(y_probe, "shape"):
                return {"error": f"Model forward did not return a Tensor (got {type(y_probe)})"}
            label_shape = tuple(int(s) for s in y_probe.shape)
            
            training_history = []
            
            for epoch in range(epochs):
                # Generate random batch
                X_batch = rasptorch.Tensor(
                    np.random.randn(*input_shape).astype(np.float32),
                    device=device
                )
                y_batch = rasptorch.Tensor(
                    np.random.randn(*label_shape).astype(np.float32),
                    device=device
                )
                
                # Forward pass
                output = _forward_for_loss(model, X_batch)
                if not hasattr(output, "shape"):
                    return {"error": f"Model forward did not return a Tensor (got {type(output)})"}
                
                # Simple MSE loss (no power operator - use multiplication)
                diff = output - y_batch
                loss = (diff * diff).mean()
                
                # Backward pass
                loss.backward()
                
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()
                
                # Convert loss to scalar (handle GPU tensors)
                loss_np = loss.numpy()
                if isinstance(loss_np, np.ndarray):
                    loss_value = float(loss_np.item() if loss_np.size == 1 else loss_np.flatten()[0])
                else:
                    loss_value = float(loss_np)
                training_history.append(loss_value)
            
            # Save updated model
            self._save_session_model(model_id, model_data)
            
            return {
                "status": "success",
                "model_id": model_id,
                "epochs": epochs,
                "device": device,
                "learning_rate": learning_rate,
                "optimizer": optimizer_type,
                "final_loss": float(training_history[-1]) if training_history else 0.0,
                "training_history": training_history,
            }
        except Exception as e:
            import traceback

            msg = str(e) if e is not None else ""
            if not msg.strip():
                msg = f"{type(e).__name__}: {repr(e)}"
            tb = traceback.format_exc(limit=8)
            return {
                "error": msg,
                "error_type": type(e).__name__,
                "traceback": tb,
            }


_model_commands = ModelCommands()

def get_model_commands() -> ModelCommands:
    """Get global model commands instance."""
    return _model_commands
