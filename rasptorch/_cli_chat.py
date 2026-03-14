"""Chat-like REPL interface for rasptorch CLI.

Provides an interactive, conversational interface similar to OpenCode.
"""

import os
import sys
from typing import Optional, Dict, Any, List
from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
import json
import numpy as np

try:
    import rasptorch
    from ._cli_commands import TensorCommands, get_model_commands
    _HAS_RASPTORCH = True
except ImportError:
    _HAS_RASPTORCH = False


class ChatREPL:
    """Interactive chat-like REPL for rasptorch."""

    def __init__(self):
        self.history_file = os.path.expanduser("~/.rasptorch_history")
        self.session: Optional[PromptSession] = None
        self.context: Dict[str, Any] = {
            "train_epochs": 5,
            "batch_size": 32,
            "device": "cpu",
            "optimizer": "Adam",
            "learning_rate": 0.001,
        }
        self.style = Style.from_dict({
            "prompt": "#00d4ff bold",
            "model-id": "#00ff88",
            "status": "#ffaa00",
        })

    def initialize(self):
        """Initialize the REPL session."""
        self.session = PromptSession(
            history=FileHistory(self.history_file),
            style=self.style,
        )
        self._print_banner()

    def _ensure_session(self) -> None:
        if self.session is None:
            self.initialize()

    def _prompt_text(self, prompt: str, *, default: Optional[str] = None) -> Optional[str]:
        self._ensure_session()
        suffix = f" [{default}]" if default is not None else ""
        try:
            val = self.session.prompt(f"{prompt}{suffix}: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("Cancelled.")
            return None
        if val == "" and default is not None:
            return str(default)
        return val

    def _prompt_int(self, prompt: str, *, default: Optional[int] = None, min_value: Optional[int] = None) -> Optional[int]:
        while True:
            raw = self._prompt_text(prompt, default=str(default) if default is not None else None)
            if raw is None:
                return None
            try:
                val = int(raw)
            except ValueError:
                print("✗ Please enter an integer")
                continue
            if min_value is not None and val < min_value:
                print(f"✗ Must be >= {min_value}")
                continue
            return val

    def _prompt_float(self, prompt: str, *, default: Optional[float] = None, min_value: Optional[float] = None) -> Optional[float]:
        while True:
            raw = self._prompt_text(prompt, default=str(default) if default is not None else None)
            if raw is None:
                return None
            try:
                val = float(raw)
            except ValueError:
                print("✗ Please enter a number")
                continue
            if min_value is not None and val < min_value:
                print(f"✗ Must be >= {min_value}")
                continue
            return val

    def _prompt_activation(self, prompt: str, *, default: str = "relu") -> Optional[str]:
        allowed = {"relu", "gelu", "tanh", "sigmoid", "silu", "leaky_relu", "elu", "none"}
        while True:
            raw = self._prompt_text(prompt, default=default)
            if raw is None:
                return None
            val = str(raw).strip().lower().replace("-", "_")
            if val in {"leakyrelu", "leaky"}:
                val = "leaky_relu"
            if val in allowed:
                return val
            print(f"✗ Unknown activation '{raw}'. Choose one of: {', '.join(sorted(allowed))}")

    def _parse_builder_args(self, tokens: List[str]) -> Dict[str, Any]:
        """Parse act/activation/activations tokens used by builders."""
        activation = "relu"
        activations = None
        for tok in tokens:
            t = str(tok).strip()
            if t.startswith("activations="):
                activations = [x.strip() for x in t.split("=", 1)[1].split(",") if x.strip()]
            elif t.startswith("act=") or t.startswith("activation="):
                activation = t.split("=", 1)[1].strip()
            else:
                # Allow shorthand: `... gelu` or `... relu,none`
                if "," in t:
                    activations = [x.strip() for x in t.split(",") if x.strip()]
                else:
                    activation = t
        return {"activation": activation, "activations": activations}

    def _wizard_mlp(self) -> Optional[Dict[str, Any]]:
        n_layers = self._prompt_int("Number of layers (including input+output)", default=4, min_value=2)
        if n_layers is None:
            return None
        sizes: List[int] = []
        for i in range(n_layers):
            label = "input" if i == 0 else ("output" if i == n_layers - 1 else f"hidden{i}")
            val = self._prompt_int(f"Units for {label} layer", min_value=1)
            if val is None:
                return None
            sizes.append(val)

        if n_layers <= 2:
            return {"layer_sizes": sizes, "activation": "none", "activations": None}

        mode = self._prompt_text("Activation mode (single/per-layer)", default="single")
        if mode is None:
            return None
        mode = mode.strip().lower()

        if mode in ("per", "per-layer", "per_layer", "layer", "layers"):
            acts: List[str] = []
            for i in range(n_layers - 2):
                act = self._prompt_activation(f"Activation after hidden transition {i+1}", default="relu")
                if act is None:
                    return None
                acts.append(act)
            return {"layer_sizes": sizes, "activation": "relu", "activations": acts}

        act = self._prompt_activation("Activation between layers", default="relu")
        if act is None:
            return None
        return {"layer_sizes": sizes, "activation": act, "activations": None}

    def _wizard_linear(self) -> Optional[Dict[str, Any]]:
        input_size = self._prompt_int("Input size", default=10, min_value=1)
        if input_size is None:
            return None
        n_hidden = self._prompt_int("Number of hidden layers", default=2, min_value=0)
        if n_hidden is None:
            return None
        hidden_sizes: List[int] = []
        for i in range(n_hidden):
            val = self._prompt_int(f"Units for hidden{i+1}", default=32, min_value=1)
            if val is None:
                return None
            hidden_sizes.append(val)
        output_size = self._prompt_int("Output size", default=2, min_value=1)
        if output_size is None:
            return None

        if n_hidden == 0:
            return {
                "input_size": input_size,
                "hidden_sizes": hidden_sizes,
                "output_size": output_size,
                "activation": "none",
                "activations": None,
            }

        mode = self._prompt_text("Activation mode (single/per-layer)", default="single")
        if mode is None:
            return None
        mode = mode.strip().lower()
        if mode in ("per", "per-layer", "per_layer", "layer", "layers"):
            acts: List[str] = []
            for i in range(n_hidden):
                act = self._prompt_activation(f"Activation after hidden{i+1}", default="relu")
                if act is None:
                    return None
                acts.append(act)
            return {
                "input_size": input_size,
                "hidden_sizes": hidden_sizes,
                "output_size": output_size,
                "activation": "relu",
                "activations": acts,
            }
        act = self._prompt_activation("Activation after each hidden layer", default="relu")
        if act is None:
            return None
        return {
            "input_size": input_size,
            "hidden_sizes": hidden_sizes,
            "output_size": output_size,
            "activation": act,
            "activations": None,
        }

    def _wizard_cnn(self) -> Optional[Dict[str, Any]]:
        in_ch = self._prompt_int("Input channels", default=3, min_value=1)
        if in_ch is None:
            return None
        n_layers = self._prompt_int("Number of layers", default=3, min_value=1)
        if n_layers is None:
            return None
        out_chs: List[int] = []
        for i in range(n_layers):
            val = self._prompt_int(f"Output channels for layer{i+1}", default=32 if i == 0 else 64, min_value=1)
            if val is None:
                return None
            out_chs.append(val)

        kernels_mode = self._prompt_text("Kernel sizes? (default/enter)", default="default")
        if kernels_mode is None:
            return None
        kernels = None
        if kernels_mode.strip().lower() not in ("default", "d", ""):
            # Ask each kernel size explicitly
            kernels = []
            for i in range(n_layers):
                k = self._prompt_int(f"Kernel size for layer{i+1}", default=3, min_value=1)
                if k is None:
                    return None
                kernels.append(k)

        if n_layers <= 1:
            return {"in_channels": in_ch, "out_channels": out_chs, "kernels": kernels, "activation": "none", "activations": None}

        mode = self._prompt_text("Activation mode (single/per-layer)", default="single")
        if mode is None:
            return None
        mode = mode.strip().lower()
        if mode in ("per", "per-layer", "per_layer", "layer", "layers"):
            acts: List[str] = []
            for i in range(n_layers - 1):
                act = self._prompt_activation(f"Activation after layer{i+1}", default="relu")
                if act is None:
                    return None
                acts.append(act)
            return {"in_channels": in_ch, "out_channels": out_chs, "kernels": kernels, "activation": "relu", "activations": acts}
        act = self._prompt_activation("Activation between layers", default="relu")
        if act is None:
            return None
        return {"in_channels": in_ch, "out_channels": out_chs, "kernels": kernels, "activation": act, "activations": None}

    def _wizard_gru(self) -> Optional[Dict[str, Any]]:
        input_sz = self._prompt_int("Input size", default=128, min_value=1)
        if input_sz is None:
            return None
        hidden_sz = self._prompt_int("Hidden size", default=256, min_value=1)
        if hidden_sz is None:
            return None
        num_layers = self._prompt_int("Number of layers", default=1, min_value=1)
        if num_layers is None:
            return None
        return {"input_size": input_sz, "hidden_size": hidden_sz, "num_layers": num_layers}

    def _wizard_transformer(self) -> Optional[Dict[str, Any]]:
        vocab_size = self._prompt_int("Vocab size", default=1000, min_value=2)
        if vocab_size is None:
            return None
        d_model = self._prompt_int("d_model", default=128, min_value=1)
        if d_model is None:
            return None
        num_heads = self._prompt_int("num_heads", default=4, min_value=1)
        if num_heads is None:
            return None
        num_layers = self._prompt_int("num_layers", default=2, min_value=1)
        if num_layers is None:
            return None
        return {"vocab_size": vocab_size, "d_model": d_model, "num_heads": num_heads, "num_layers": num_layers}

    def _wizard_lora(self, cmds: Any) -> Optional[Dict[str, Any]]:
        if not getattr(cmds, "models", None):
            print("✗ No models available. Create a model first.")
            return None
        default_base = self.context.get("current_model")
        base = self._prompt_text("Base model id", default=str(default_base) if default_base else None)
        if base is None:
            return None
        rank = self._prompt_int("Rank", default=8, min_value=1)
        if rank is None:
            return None
        alpha = self._prompt_float("Alpha", default=16.0, min_value=0.0)
        if alpha is None:
            return None
        return {"base_model": base, "rank": rank, "alpha": alpha}

    def _print_banner(self):
        """Print welcome banner."""
        banner = """
╔════════════════════════════════════════════════════════════╗
║           🤖 rasptorch CLI - Chat Mode                    ║
║                                                            ║
║  Deep learning on Raspberry Pi 5                          ║
║  Type 'help' for commands, 'exit' to quit                 ║
╚════════════════════════════════════════════════════════════╝
"""
        print(banner)

    def get_prompt_text(self) -> str:
        """Get current prompt based on context."""
        if self.context.get("current_model"):
            model_id = self.context["current_model"][:8]
            return f"rasptorch [model: {model_id}]> "
        elif self.context.get("last_action"):
            return f"rasptorch> "
        return "rasptorch> "

    def prompt(self) -> str:
        """Get user input."""
        if self.session is None:
            self.initialize()
        try:
            return self.session.prompt(self.get_prompt_text())
        except (KeyboardInterrupt, EOFError):
            return "exit"

    def print_result(self, result: Dict[str, Any], as_json: bool = False):
        """Print command result."""
        if as_json:
            print(json.dumps(result, indent=2))
        else:
            status = result.get("status", "done")
            if status == "success":
                message = result.get("message", "Done!")
                print(f"✓ {message}")
                if "data" in result:
                    for key, value in result["data"].items():
                        print(f"  {key}: {value}")
            elif status == "error":
                print(f"✗ Error: {result.get('message', 'Unknown error')}")
            else:
                print(result)

    def set_context(self, key: str, value: Any):
        """Set REPL context."""
        self.context[key] = value

    def get_help(self) -> str:
        """Get help text."""
        return """
rasptorch CLI - Interactive Commands

TENSOR OPERATIONS:
  tensor create <shape>        Create random tensor (e.g., 2,3,4)
  tensor zeros <shape>         Create zeros tensor
  tensor ones <shape>          Create ones tensor

MODEL BUILDERS:
    model mlp [args]             Create MLP (args or interactive)
    model linear [args]          Create Linear/MLP-style model (args or interactive)
    model cnn [args]             Create CNN (args or interactive)
    model gru [args]             Create GRU (args or interactive)
    model transformer [args]     Create Transformer (args or interactive)
    model lora [args]            Create LoRA adapter (args or interactive)

MODEL MANAGEMENT:
  model list                   List all models
  model info <id>              Show model details
  model use|select <id>        Select model for training
    model combine <id1> <id2>     Combine models (A then B)
  model deselect               Deselect current model
  model remove <id>            Remove a model
  model remove-all             Remove all models

LAYER/ACTIVATION OPTIONS (for builders):
    act=<name>                   Set activation (relu, gelu, tanh, sigmoid, silu, leaky_relu, elu, none)
    activations=a,b,c            Per-layer activations (for MLP: count=layers-2)

OPTIMIZER:
  optimizer create <type>      Create optimizer (adam, sgd)
  optimizer set-lr <value>     Set learning rate

TRAINING:
  train epochs <n>             Set training epochs
  train batch-size <n>         Set batch size
  train start                  Start training

DEVICE:
  device cpu                   Use CPU for operations
  device gpu                   Use Vulkan GPU for operations
  device status                Show current device

UTILITY:
  info                         Show version & environment
  clear                        Clear session
  exit                         Exit CLI

Type 'help <command>' for more details.
"""

    def run(self):
        """Start the interactive REPL."""
        self.initialize()
        try:
            while True:
                try:
                    user_input = self.prompt()
                    if not user_input or user_input.lower() == "exit":
                        print("Goodbye! 👋")
                        break

                    self._handle_command(user_input)

                except (KeyboardInterrupt, EOFError):
                    print("\nGoodbye! 👋")
                    break
                except Exception as e:
                    print(f"✗ Error: {e}")

        except Exception as e:
            print(f"REPL error: {e}", file=sys.stderr)
            sys.exit(1)

    def _handle_command(self, command_str: str):
        """Parse and execute command."""
        parts = command_str.strip().split()
        if not parts:
            return

        cmd = parts[0].lower()

        if cmd == "help":
            if len(parts) > 1:
                subcommand = parts[1].lower()
                self._print_command_help(subcommand)
            else:
                print(self.get_help())
        
        elif cmd == "info":
            self._cmd_info()
        
        elif cmd == "clear":
            os.system("clear" if os.name != "nt" else "cls")
        
        elif cmd == "tensor":
            self._handle_tensor_command(parts[1:])
        
        elif cmd == "model":
            self._handle_model_command(parts[1:])
        
        elif cmd == "train":
            self._handle_train_command(parts[1:])
        
        elif cmd == "optimizer":
            self._handle_optimizer_command(parts[1:])
        
        elif cmd == "device":
            self._handle_device_command(parts[1:])
        
        else:
            print(f"✗ Command not recognized: {cmd}")
            print(f"  Type 'help' for available commands")

    def _print_command_help(self, subcommand: str):
        """Print help for specific command."""
        helps = {
            "tensor": "tensor create|zeros|ones <shape> - Create tensors",
            "model": "model mlp|linear|cnn|gru|transformer|lora <args> - Create/manage models (interactive if args omitted)",
            "train": "train epochs|batch-size|start - Configure and start training",
            "optimizer": "optimizer create|set-lr - Configure optimizer",
            "device": "device cpu|gpu|status - Set compute device (GPU requires Vulkan)",
            "info": "info - Show system information",
        }
        print(helps.get(subcommand, f"No help available for {subcommand}"))

    def _cmd_info(self):
        """Show system info."""
        try:
            import rasptorch
            from .vulkan_backend import _HAS_VULKAN, _VULKAN_DISABLED_REASON

            device = self.context.get("device", "cpu")

            print(f"rasptorch version: {rasptorch.__version__}")
            print(f"numpy version: {np.__version__}")
            if _HAS_VULKAN:
                print(f"vulkan: ✓ Available")
                if device == "gpu":
                    print(f"device: gpu")
                else:
                    print(f"device: cpu (gpu available via Vulkan)")
            else:
                print(f"vulkan: ✗ Not available")
                if _VULKAN_DISABLED_REASON:
                    print(f"  Reason: {_VULKAN_DISABLED_REASON}")

                # If user requested GPU in this session but Vulkan isn't available,
                # surface that mismatch.
                if device == "gpu":
                    print(f"device: gpu (requested, but Vulkan unavailable)")
                else:
                    print(f"device: cpu")
        except Exception as e:
            print(f"✗ Error: {e}")

    def _handle_tensor_command(self, args: List[str]):
        """Handle tensor operations."""
        if not args:
            print("✗ Usage: tensor create|zeros|ones <shape>")
            return
        
        subcmd = args[0].lower()
        if len(args) < 2:
            print("✗ Missing shape argument")
            return
        
        shape_str = args[1]
        try:
            shape = tuple(int(x.strip()) for x in shape_str.split(","))
        except ValueError:
            print(f"✗ Invalid shape: {shape_str}")
            return
        
        try:
            if subcmd == "create":
                result = TensorCommands.create_random(shape)
            elif subcmd == "zeros":
                result = TensorCommands.create_zeros(shape)
            elif subcmd == "ones":
                result = TensorCommands.create_ones(shape)
            else:
                print(f"✗ Unknown tensor command: {subcmd}")
                return
            
            tensor_id = result.get("tensor_id", "?")
            print(f"✓ Created {subcmd} tensor: {result['shape']} (id: {tensor_id[:8]})")
        except Exception as e:
            print(f"✗ Error: {e}")

    def _handle_model_command(self, args: List[str]):
        """Handle model operations."""
        if not args:
            print("✗ Usage: model mlp|linear|cnn|gru|transformer|lora|combine|list|info|use|select|deselect|remove|remove-all <args>")
            return
        
        subcmd = args[0].lower()
        cmds = get_model_commands()
        
        try:
            if subcmd == "list":
                result = cmds.list_models()
                if result["total"] == 0:
                    print("(no models)")
                else:
                    print(f"\nModels ({result['total']}):")
                    for m in result["models"]:
                        print(f"  {m['model_id']}: {m['type']}")
            
            elif subcmd == "mlp":
                if len(args) < 2:
                    wiz = self._wizard_mlp()
                    if wiz is None:
                        return
                    result = cmds.create_mlp(
                        wiz["layer_sizes"],
                        activation=wiz.get("activation", "relu"),
                        activations=wiz.get("activations"),
                    )
                else:
                    layers = [int(x.strip()) for x in args[1].split(",")]
                    parsed = self._parse_builder_args(args[2:])
                    result = cmds.create_mlp(layers, activation=parsed["activation"], activations=parsed["activations"])
                if "error" not in result:
                    mid = result["model_id"]
                    self.context["current_model"] = mid
                    print(f"✓ Created MLP: {mid[:8]}")
                else:
                    print(f"✗ Error: {result['error']}")

            elif subcmd == "linear":
                if len(args) < 2:
                    wiz = self._wizard_linear()
                    if wiz is None:
                        return
                    result = cmds.create_linear_model(
                        wiz["input_size"],
                        wiz["hidden_sizes"],
                        wiz["output_size"],
                        activation=wiz.get("activation", "relu"),
                        activations=wiz.get("activations"),
                    )
                else:
                    # Format: model linear <input> <hidden1,hidden2,...> <output> [act/activations]
                    if len(args) < 4:
                        print("✗ Usage: model linear <input_size> <hidden_sizes_csv> <output_size> [act=relu|activations=...]")
                        return
                    input_size = int(args[1])
                    hidden_sizes = [int(x.strip()) for x in args[2].split(",") if x.strip()]
                    output_size = int(args[3])
                    parsed = self._parse_builder_args(args[4:])
                    result = cmds.create_linear_model(
                        input_size,
                        hidden_sizes,
                        output_size,
                        activation=parsed["activation"],
                        activations=parsed["activations"],
                    )
                if "error" not in result:
                    mid = result["model_id"]
                    self.context["current_model"] = mid
                    print(f"✓ Created Linear: {mid[:8]}")
                else:
                    print(f"✗ Error: {result['error']}")
            
            elif subcmd == "cnn":
                try:
                    if len(args) < 2:
                        wiz = self._wizard_cnn()
                        if wiz is None:
                            return
                        result = cmds.create_cnn(
                            wiz["in_channels"],
                            wiz["out_channels"],
                            wiz.get("kernels"),
                            activation=wiz.get("activation", "relu"),
                            activations=wiz.get("activations"),
                        )
                    else:
                        # Format: model cnn <in_channels> <out_channels_csv> [kernels_csv] [act/activations]
                        if len(args) < 3:
                            print("✗ Usage: model cnn <in_channels> <out_channels_csv> [kernels_csv] [act=relu|activations=...]")
                            return
                        in_ch = int(args[1])
                        out_chs = [int(x.strip()) for x in args[2].split(",") if x.strip()]
                        kernels = None
                        extra_start = 3
                        if len(args) >= 4 and args[3].replace(",", "").strip().isdigit():
                            kernels = [int(x.strip()) for x in args[3].split(",") if x.strip()]
                            extra_start = 4
                        parsed = self._parse_builder_args(args[extra_start:])
                        result = cmds.create_cnn(
                            in_ch,
                            out_chs,
                            kernels,
                            activation=parsed["activation"],
                            activations=parsed["activations"],
                        )

                    if "error" not in result:
                        mid = result["model_id"]
                        self.context["current_model"] = mid
                        print(f"✓ Created CNN: {mid[:8]}")
                    else:
                        print(f"✗ Error: {result['error']}")
                except ValueError:
                    print("✗ Invalid numeric value")
            
            elif subcmd == "gru":
                try:
                    if len(args) < 2:
                        wiz = self._wizard_gru()
                        if wiz is None:
                            return
                        result = cmds.create_gru(wiz["input_size"], wiz["hidden_size"], wiz["num_layers"])
                    else:
                        if len(args) < 3:
                            print("✗ Usage: model gru <input_size> <hidden_size> [num_layers]")
                            return
                        input_sz = int(args[1])
                        hidden_sz = int(args[2])
                        num_layers = int(args[3]) if len(args) > 3 else 1
                        result = cmds.create_gru(input_sz, hidden_sz, num_layers)

                    if "error" not in result:
                        mid = result["model_id"]
                        self.context["current_model"] = mid
                        print(f"✓ Created GRU: {mid[:8]}")
                    else:
                        print(f"✗ Error: {result['error']}")
                except ValueError:
                    print("✗ Invalid numeric value")

            elif subcmd == "transformer":
                try:
                    if len(args) < 2:
                        wiz = self._wizard_transformer()
                        if wiz is None:
                            return
                        result = cmds.create_transformer(
                            wiz["vocab_size"], wiz["d_model"], wiz["num_heads"], wiz["num_layers"]
                        )
                    else:
                        if len(args) < 5:
                            print("✗ Usage: model transformer <vocab_size> <d_model> <num_heads> <num_layers>")
                            return
                        vocab_size = int(args[1])
                        d_model = int(args[2])
                        num_heads = int(args[3])
                        num_layers = int(args[4])
                        result = cmds.create_transformer(vocab_size, d_model, num_heads, num_layers)

                    if "error" not in result:
                        mid = result["model_id"]
                        self.context["current_model"] = mid
                        print(f"✓ Created Transformer: {mid[:8]}")
                    else:
                        print(f"✗ Error: {result['error']}")
                except ValueError:
                    print("✗ Invalid numeric value")

            elif subcmd == "lora":
                if len(args) < 2:
                    wiz = self._wizard_lora(cmds)
                    if wiz is None:
                        return
                    result = cmds.create_lora_adapter(wiz["base_model"], rank=wiz["rank"], alpha=wiz["alpha"])
                else:
                    # Format: model lora <base_model_id> [rank] [alpha]
                    base = args[1]
                    rank = int(args[2]) if len(args) > 2 else 8
                    alpha = float(args[3]) if len(args) > 3 else 16.0
                    result = cmds.create_lora_adapter(base, rank=rank, alpha=alpha)

                if "error" not in result:
                    aid = result.get("adapter_id", "?")
                    print(f"✓ Created LoRA adapter: {str(aid)[:8]}")
                else:
                    print(f"✗ Error: {result['error']}")
            
            elif subcmd in ("use", "select"):
                if len(args) < 2:
                    print("✗ Usage: model use|select <model_id>")
                    return
                model_id = args[1]
                if model_id in cmds.models:
                    self.context["current_model"] = model_id
                    print(f"✓ Selected model: {model_id[:8]}")
                else:
                    print(f"✗ Model not found: {model_id}")

            elif subcmd == "combine":
                if len(args) < 3:
                    print("✗ Usage: model combine <model_id_a> <model_id_b>")
                    return
                model_a_id = args[1]
                model_b_id = args[2]
                result = cmds.combine_models(model_a_id, model_b_id)
                if "error" in result:
                    print(f"✗ Error: {result['error']}")
                    return
                mid = result["model_id"]
                self.context["current_model"] = mid
                print(f"✓ Combined model: {mid[:8]}")
            
            elif subcmd == "deselect":
                if "current_model" not in self.context:
                    print("✗ No model currently selected")
                    return
                model_id = self.context["current_model"]
                del self.context["current_model"]
                print(f"✓ Deselected model: {model_id[:8]}")
            
            elif subcmd == "info":
                if "current_model" not in self.context:
                    print("✗ No model selected. Use 'model use <id>' first")
                    return
                model_id = self.context["current_model"]
                if model_id in cmds.models:
                    model_data = cmds.models[model_id]
                    print(f"\nModel: {model_id[:8]}")
                    print(f"  Type: {model_data.get('type', 'Unknown')}")
                    config = model_data.get('config', {})
                    for k, v in config.items():
                        print(f"  {k}: {v}")
                else:
                    print(f"✗ Model not found: {model_id}")
            
            elif subcmd == "remove":
                if len(args) < 2:
                    print("✗ Usage: model remove <model_id>")
                    return
                model_id = args[1]
                result = cmds.delete_model(model_id)
                if "error" in result:
                    print(f"✗ Error: {result['error']}")
                else:
                    # Clear context if the removed model was selected
                    if self.context.get("current_model") == model_id:
                        del self.context["current_model"]
                    print(f"✓ Removed model: {model_id[:8]}")
            
            elif subcmd == "remove-all":
                if cmds.models:
                    count = len(cmds.models)
                    # Confirm removal using session.prompt instead of input()
                    if self.session is None:
                        self.initialize()
                    try:
                        confirm = self.session.prompt(f"⚠️  Remove all {count} model(s)? (Y/n): ").strip().lower()
                        if confirm not in ("y", "yes", ""):  # empty string defaults to yes
                            print("Cancelled.")
                            return
                    except (KeyboardInterrupt, EOFError):
                        print("Cancelled.")
                        return
                    
                    # Remove all models
                    model_ids = list(cmds.models.keys())
                    for mid in model_ids:
                        cmds.delete_model(mid)
                    
                    # Clear context
                    if "current_model" in self.context:
                        del self.context["current_model"]
                    
                    print(f"✓ Removed all {count} model(s)")
                else:
                    print("(no models to remove)")
            
            else:
                print(f"✗ Unknown model command: {subcmd}")
        
        except Exception as e:
            print(f"✗ Error: {e}")

    def _handle_train_command(self, args: List[str]):
        """Handle training commands."""
        if not args:
            print("✗ Usage: train epochs|batch-size|start <value>")
            return
        
        subcmd = args[0].lower()
        
        if subcmd == "epochs":
            if len(args) < 2:
                print("✗ Usage: train epochs <n>")
                return
            try:
                self.context["train_epochs"] = int(args[1])
                print(f"✓ Set epochs: {self.context['train_epochs']}")
            except ValueError:
                print("✗ Invalid epoch value")
        
        elif subcmd == "batch-size":
            if len(args) < 2:
                print("✗ Usage: train batch-size <n>")
                return
            try:
                self.context["batch_size"] = int(args[1])
                print(f"✓ Set batch size: {self.context['batch_size']}")
            except ValueError:
                print("✗ Invalid batch size")
        
        elif subcmd == "start":
            self._train_model()
        
        else:
            print(f"✗ Unknown train command: {subcmd}")

    def _train_model(self):
        """Execute model training."""
        if "current_model" not in self.context:
            print("✗ No model selected. Use 'model use <id>' first")
            return
        
        model_id = self.context["current_model"]
        epochs = self.context.get("train_epochs", 5)
        batch_size = self.context.get("batch_size", 32)
        device = self.context.get("device", "cpu")
        
        cmds = get_model_commands()
        
        if model_id not in cmds.models:
            print(f"✗ Model not found: {model_id}")
            return
        
        try:
            print(f"\n🚀 Training {model_id[:8]} ({epochs} epochs, batch_size={batch_size}, device={device})...")
            result = cmds.train_model(
                model_id,
                epochs=epochs,
                learning_rate=0.001,
                batch_size=batch_size,
                device=device,
                optimizer_type="Adam"
            )
            
            if "error" in result:
                print(f"✗ Training failed: {result['error']}")
                return
            
            print(f"✓ Training complete!")
            print(f"  Final loss: {result.get('final_loss', 0.0):.6f}")
            history = result.get("training_history", [])
            if history:
                print(f"  Loss history: {' → '.join(f'{l:.3f}' for l in history[:5])}")
                if len(history) > 5:
                    print(f"             ... {' → '.join(f'{l:.3f}' for l in history[-3:])}")
        
        except Exception as e:
            print(f"✗ Error: {e}")

    def _handle_optimizer_command(self, args: List[str]):
        """Handle optimizer commands."""
        if not args:
            print("✗ Usage: optimizer create|set-lr <value>")
            return
        
        subcmd = args[0].lower()
        
        if subcmd == "create":
            if len(args) < 2:
                opt_type = "Adam"
            else:
                opt_type = args[1].lower()
            
            if opt_type not in ["adam", "sgd"]:
                print(f"✗ Unknown optimizer: {opt_type}")
                return
            
            self.context["optimizer"] = opt_type
            print(f"✓ Selected optimizer: {opt_type}")
        
        elif subcmd == "set-lr":
            if len(args) < 2:
                print("✗ Usage: optimizer set-lr <value>")
                return
            try:
                lr = float(args[1])
                self.context["learning_rate"] = lr
                print(f"✓ Set learning rate: {lr}")
            except ValueError:
                print("✗ Invalid learning rate")
        
        else:
            print(f"✗ Unknown optimizer command: {subcmd}")

    def _handle_device_command(self, args: List[str]):
        """Handle device configuration."""
        if not args:
            print("✗ Usage: device cpu|gpu|status")
            return
        
        subcmd = args[0].lower()
        
        if subcmd == "cpu":
            self.context["device"] = "cpu"
            print(f"✓ Device set to: CPU")
        
        elif subcmd == "gpu":
            from .vulkan_backend import _HAS_VULKAN
            if not _HAS_VULKAN:
                print(f"✗ Vulkan GPU not available. Run 'info' for details.")
                return
            self.context["device"] = "gpu"
            print(f"✓ Device set to: GPU (Vulkan)")
        
        elif subcmd == "status":
            from .vulkan_backend import _HAS_VULKAN, _VULKAN_DISABLED_REASON
            device = self.context.get("device", "cpu")
            print(f"Current device: {device.upper()}")
            if _HAS_VULKAN:
                print(f"Vulkan: ✓ Available (can use GPU)")
            else:
                print(f"Vulkan: ✗ Not available (CPU only)")
                if _VULKAN_DISABLED_REASON:
                    print(f"  Reason: {_VULKAN_DISABLED_REASON}")
        
        else:
            print(f"✗ Unknown device command: {subcmd}")

    def display_table(self, headers: List[str], rows: List[List[str]]):
        """Display formatted table."""
        if not rows:
            print("(no data)")
            return

        col_widths = [len(str(h)) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))

        header_line = " | ".join(
            str(h).ljust(col_widths[i]) for i, h in enumerate(headers)
        )
        print(header_line)
        print("-" * len(header_line))

        for row in rows:
            print(" | ".join(
                str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)
            ))
