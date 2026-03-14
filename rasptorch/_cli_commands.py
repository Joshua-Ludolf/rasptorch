"""Command implementations for rasptorch CLI."""

from typing import Any, Dict, Tuple, List, Optional
import numpy as np
import rasptorch
import os
import tempfile
import uuid


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

    def combine_models(self, model_a_id: str, model_b_id: str) -> Dict[str, Any]:
        """Combine two existing models sequentially (A then B) into a new model."""
        if model_a_id not in self.models:
            return {"error": f"Model {model_a_id} not found"}
        if model_b_id not in self.models:
            return {"error": f"Model {model_b_id} not found"}

        try:
            model_a_data = self.models[model_a_id]
            model_b_data = self.models[model_b_id]

            model_a = model_a_data.get("model")
            model_b = model_b_data.get("model")

            if model_a is None:
                model_a = self._reconstruct_model(model_a_data.get("type", "Unknown"), model_a_data.get("config", {}))
                if model_a is None:
                    return {"error": f"Cannot reconstruct {model_a_data.get('type', 'Unknown')} model"}
                model_a_data["model"] = model_a

            if model_b is None:
                model_b = self._reconstruct_model(model_b_data.get("type", "Unknown"), model_b_data.get("config", {}))
                if model_b is None:
                    return {"error": f"Cannot reconstruct {model_b_data.get('type', 'Unknown')} model"}
                model_b_data["model"] = model_b

            a_in, a_out = self._infer_linear_in_out(model_a)
            b_in, b_out = self._infer_linear_in_out(model_b)

            # If we can infer linear IO dims, enforce compatibility.
            if a_out is not None and b_in is not None and a_out != b_in:
                return {
                    "error": f"Incompatible models: first outputs {a_out} but second expects {b_in}"
                }

            combined_layers = self._flatten_layers(model_a) + self._flatten_layers(model_b)
            if not combined_layers:
                return {"error": "Failed to combine models (no layers found)"}

            combined_model = rasptorch.nn.Sequential(*combined_layers)
            combined_id = str(uuid.uuid4())[:8]

            combined_config: Dict[str, Any] = {
                "combine": "sequential",
                "model_a_id": model_a_id,
                "model_b_id": model_b_id,
            }
            if a_in is not None:
                combined_config["input_size"] = a_in
            if b_out is not None:
                combined_config["output_size"] = b_out

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
                },
                "architecture": {
                    "combine": "sequential",
                    "input_size": a_in,
                    "output_size": b_out,
                    "num_layers": len(combined_layers),
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
                model = rasptorch.nn.GRU(input_size, hidden_size, num_layers)
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
                layers.append(rasptorch.nn.MultiheadAttention(d_model, num_heads))
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
            if optimizer_type.lower() == "adam":
                optimizer = rasptorch.Adam(model.parameters(), lr=lr)
            elif optimizer_type.lower() == "sgd":
                optimizer = rasptorch.SGD(model.parameters(), lr=lr)
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
            import pickle
            model_data = self.models[model_id]
            model = model_data.get("model")
            
            # Reconstruct if needed
            if model is None:
                model_type = model_data.get("type", "Unknown")
                config = model_data.get("config", {})
                model = self._reconstruct_model(model_type, config)
                if model is None:
                    return {"error": f"Cannot reconstruct {model_type} model"}
                model_data["model"] = model
            
            state_dict = model.state_dict()
            
            save_data = {
                "model_type": model_data.get("type", "Unknown"),
                "config": model_data.get("config", {}),
                "state_dict": state_dict,
            }
            
            with open(path, "wb") as f:
                pickle.dump(save_data, f)
            
            return {
                "status": "success",
                "model_id": model_id,
                "path": path,
                "message": f"Model saved successfully",
            }
        except Exception as e:
            return {"error": str(e)}

    def load_model(self, path: str) -> Dict[str, Any]:
        """Load model from file."""
        try:
            import pickle
            with open(path, "rb") as f:
                save_data = pickle.load(f)
            
            model_type = save_data.get("model_type", "Unknown")
            config = save_data.get("config", {})
            state_dict = save_data.get("state_dict", {})
            
            model_id = str(id({}))
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
            model = model_data.get("model")
            
            # Reconstruct model if needed
            if model is None:
                model_type = model_data.get("type", "Unknown")
                config = model_data.get("config", {})
                model = self._reconstruct_model(model_type, config)
                if model is None:
                    return {"error": f"Cannot reconstruct {model_type} model"}
                model_data["model"] = model
            
            # Move model to device if needed
            if device == "gpu":
                try:
                    model.to("gpu")
                except Exception as e:
                    return {"error": f"Failed to move model to GPU: {str(e)}"}
            
            # Create optimizer
            if optimizer_type.lower() == "adam":
                optimizer = rasptorch.Adam(model.parameters(), lr=learning_rate)
            elif optimizer_type.lower() == "sgd":
                optimizer = rasptorch.SGD(model.parameters(), lr=learning_rate)
            else:
                return {"error": f"Unknown optimizer: {optimizer_type}"}
            
            # Infer input size from model config
            config = model_data.get("config", {})
            input_size = config.get("input_size") or config.get("layer_sizes", [10])[0]
            output_size = config.get("output_size") or config.get("layer_sizes", [10, 2])[-1]
            
            input_shape = (batch_size, input_size)
            label_shape = (batch_size, output_size)
            
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
                output = model(X_batch)
                
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
            return {"error": str(e)}


_model_commands = ModelCommands()

def get_model_commands() -> ModelCommands:
    """Get global model commands instance."""
    return _model_commands
