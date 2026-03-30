"""rasptorch CLI - Command-line interface for deep learning on Raspberry Pi."""

import click
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple
from .. import __version__
from ._cli_utils import parse_shape, format_error, format_json_output
from ._cli_commands import TensorCommands, get_model_commands
from ._cli_chat import ChatREPL
from ..utils import resolve_device


@click.group(invoke_without_command=True)
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.option(
    "--device",
    default="auto",
    type=click.Choice(["cpu", "gpu", "auto"], case_sensitive=False),
    show_default=True,
    help="Device selection: cpu, gpu, or auto (gpu when Vulkan is working)",
)
@click.version_option(__version__)
@click.pass_context
def cli(ctx: click.Context, json_output: bool, device: str):
    """rasptorch - Deep learning on Raspberry Pi 5."""
    ctx.ensure_object(dict)
    ctx.obj["json_output"] = json_output
    ctx.obj["device"] = resolve_device(device)


@cli.group()
def tensor():
    """Tensor operations."""
    pass


@tensor.command()
@click.option("--shape", required=True)
@click.option("--device", default="auto", type=click.Choice(["cpu", "gpu", "auto"], case_sensitive=False), show_default=True)
@click.option("--dtype", default="float32")
@click.pass_context
def random(ctx, shape, device, dtype):
    """Create random tensor."""
    try:
        shape_tuple = parse_shape(shape)
        result = TensorCommands.create_random(shape_tuple, dtype, resolve_device(device))
        if ctx.obj.get("json_output"):
            click.echo(format_json_output(result))
        else:
            click.echo(f"✓ Random tensor: {result['shape']}")
    except Exception as e:
        if ctx.obj.get("json_output"):
            click.echo(format_json_output(format_error(str(e))))
        else:
            click.echo(f"✗ Error: {e}")
        sys.exit(1)


@tensor.command()
@click.option("--shape", required=True)
@click.option("--device", default="auto", type=click.Choice(["cpu", "gpu", "auto"], case_sensitive=False), show_default=True)
@click.pass_context
def zeros(ctx, shape, device):
    """Create zeros tensor."""
    try:
        shape_tuple = parse_shape(shape)
        result = TensorCommands.create_zeros(shape_tuple, "float32", resolve_device(device))
        if ctx.obj.get("json_output"):
            click.echo(format_json_output(result))
        else:
            click.echo(f"✓ Zeros tensor: {result['shape']}")
    except Exception as e:
        if ctx.obj.get("json_output"):
            click.echo(format_json_output(format_error(str(e))))
        else:
            click.echo(f"✗ Error: {e}")
        sys.exit(1)


@tensor.command()
@click.option("--shape", required=True)
@click.option("--device", default="auto", type=click.Choice(["cpu", "gpu", "auto"], case_sensitive=False), show_default=True)
@click.pass_context
def ones(ctx, shape, device):
    """Create ones tensor."""
    try:
        shape_tuple = parse_shape(shape)
        result = TensorCommands.create_ones(shape_tuple, "float32", resolve_device(device))
        if ctx.obj.get("json_output"):
            click.echo(format_json_output(result))
        else:
            click.echo(f"✓ Ones tensor: {result['shape']}")
    except Exception as e:
        if ctx.obj.get("json_output"):
            click.echo(format_json_output(format_error(str(e))))
        else:
            click.echo(f"✗ Error: {e}")
        sys.exit(1)


@cli.group()
def model():
    """Model operations."""
    pass


@model.command("linear")
@click.option("--input-size", required=True, type=int)
@click.option("--hidden-sizes", required=True)
@click.option("--output-size", required=True, type=int)
@click.option(
    "--activation",
    default="relu",
    type=click.Choice(["relu", "gelu", "tanh", "sigmoid", "silu", "leaky_relu", "elu", "none"], case_sensitive=False),
    show_default=True,
    help="Activation between layers (or 'none')",
)
@click.option(
    "--activations",
    default=None,
    help="Comma-separated per-hidden-layer activations (overrides --activation)",
)
@click.pass_context
def create_linear(ctx, input_size, hidden_sizes, output_size, activation, activations):
    """Create linear model."""
    try:
        hidden_list = [int(x.strip()) for x in hidden_sizes.split(",")]
        activations_list = None
        if activations:
            activations_list = [x.strip() for x in str(activations).split(",") if x.strip()]
            if len(activations_list) != len(hidden_list):
                raise ValueError(
                    f"--activations must have {len(hidden_list)} value(s) (one per hidden layer)"
                )
        cmds = get_model_commands()
        result = cmds.create_linear_model(
            input_size,
            hidden_list,
            output_size,
            activation=str(activation),
            activations=activations_list,
        )
        if ctx.obj.get("json_output"):
            click.echo(format_json_output(result))
        else:
            click.echo(f"✓ Linear model: {result['model_id'][:8]}")
    except Exception as e:
        if ctx.obj.get("json_output"):
            click.echo(format_json_output(format_error(str(e))))
        else:
            click.echo(f"✗ Error: {e}")
        sys.exit(1)


@model.command("mlp")
@click.option("--layers", required=True)
@click.option(
    "--activation",
    default="relu",
    type=click.Choice(["relu", "gelu", "tanh", "sigmoid", "silu", "leaky_relu", "elu", "none"], case_sensitive=False),
    show_default=True,
    help="Activation between layers (or 'none')",
)
@click.option(
    "--activations",
    default=None,
    help="Comma-separated per-layer activations (count must be layers-2; overrides --activation)",
)
@click.pass_context
def create_mlp(ctx, layers, activation, activations):
    """Create MLP."""
    try:
        layer_sizes = [int(x.strip()) for x in layers.split(",")]
        activations_list = None
        if activations:
            activations_list = [x.strip() for x in str(activations).split(",") if x.strip()]
            expected = max(len(layer_sizes) - 2, 0)
            if len(activations_list) != expected:
                raise ValueError(
                    f"--activations must have {expected} value(s) (one per hidden transition)"
                )
        cmds = get_model_commands()
        result = cmds.create_mlp(
            layer_sizes,
            activation=str(activation),
            activations=activations_list,
        )
        if "error" in result:
            raise ValueError(result["error"])
        if ctx.obj.get("json_output"):
            click.echo(format_json_output(result))
        else:
            click.echo(f"✓ MLP model: {result['model_id'][:8]}")
    except Exception as e:
        if ctx.obj.get("json_output"):
            click.echo(format_json_output(format_error(str(e))))
        else:
            click.echo(f"✗ Error: {e}")
        sys.exit(1)


@model.command("cnn")
@click.option("--in-channels", required=True, type=int)
@click.option("--out-channels", required=True)
@click.option("--kernels", default=None)
@click.option(
    "--activation",
    default="relu",
    type=click.Choice(["relu", "gelu", "tanh", "sigmoid", "silu", "leaky_relu", "elu", "none"], case_sensitive=False),
    show_default=True,
    help="Activation between layers (or 'none')",
)
@click.option(
    "--activations",
    default=None,
    help="Comma-separated per-layer activations (count must be out_channels-1; overrides --activation)",
)
@click.pass_context
def create_cnn(ctx, in_channels, out_channels, kernels, activation, activations):
    """Create CNN."""
    try:
        out_ch_list = [int(x.strip()) for x in out_channels.split(",")]
        kernel_list = [int(x.strip()) for x in kernels.split(",")] if kernels else None
        activations_list = None
        if activations:
            activations_list = [x.strip() for x in str(activations).split(",") if x.strip()]
            expected = max(len(out_ch_list) - 1, 0)
            if len(activations_list) != expected:
                raise ValueError(
                    f"--activations must have {expected} value(s) (one per transition)"
                )
        cmds = get_model_commands()
        result = cmds.create_cnn(
            in_channels,
            out_ch_list,
            kernel_list,
            activation=str(activation),
            activations=activations_list,
        )
        if "error" in result:
            raise ValueError(result["error"])
        if ctx.obj.get("json_output"):
            click.echo(format_json_output(result))
        else:
            click.echo(f"✓ CNN model: {result['model_id'][:8]}")
    except Exception as e:
        if ctx.obj.get("json_output"):
            click.echo(format_json_output(format_error(str(e))))
        else:
            click.echo(f"✗ Error: {e}")
        sys.exit(1)


@model.command("gru")
@click.option("--input-size", required=True, type=int)
@click.option("--hidden-size", required=True, type=int)
@click.option("--num-layers", default=1, type=int)
@click.pass_context
def create_gru(ctx, input_size, hidden_size, num_layers):
    """Create GRU."""
    try:
        cmds = get_model_commands()
        result = cmds.create_gru(input_size, hidden_size, num_layers)
        if "error" in result:
            raise ValueError(result["error"])
        if ctx.obj.get("json_output"):
            click.echo(format_json_output(result))
        else:
            click.echo(f"✓ GRU model: {result['model_id'][:8]}")
    except Exception as e:
        if ctx.obj.get("json_output"):
            click.echo(format_json_output(format_error(str(e))))
        else:
            click.echo(f"✗ Error: {e}")
        sys.exit(1)


@model.command("transformer")
@click.option("--vocab-size", required=True, type=int)
@click.option("--d-model", required=True, type=int)
@click.option("--num-heads", required=True, type=int)
@click.option("--num-layers", required=True, type=int)
@click.pass_context
def create_transformer(ctx, vocab_size, d_model, num_heads, num_layers):
    """Create Transformer."""
    try:
        cmds = get_model_commands()
        result = cmds.create_transformer(vocab_size, d_model, num_heads, num_layers)
        if "error" in result:
            raise ValueError(result["error"])
        if ctx.obj.get("json_output"):
            click.echo(format_json_output(result))
        else:
            click.echo(f"✓ Transformer: {result['model_id'][:8]}")
    except Exception as e:
        if ctx.obj.get("json_output"):
            click.echo(format_json_output(format_error(str(e))))
        else:
            click.echo(f"✗ Error: {e}")
        sys.exit(1)


@model.command("lora")
@click.option("--base-model", required=True)
@click.option("--rank", default=8, type=int)
@click.option("--alpha", default=16.0, type=float)
@click.pass_context
def create_lora(ctx, base_model, rank, alpha):
    """Create LoRA adapter."""
    try:
        cmds = get_model_commands()
        result = cmds.create_lora_adapter(base_model, rank, alpha)
        if "error" in result:
            raise ValueError(result["error"])
        if ctx.obj.get("json_output"):
            click.echo(format_json_output(result))
        else:
            click.echo(f"✓ LoRA adapter: {result['adapter_id'][:8]}")
    except Exception as e:
        if ctx.obj.get("json_output"):
            click.echo(format_json_output(format_error(str(e))))
        else:
            click.echo(f"✗ Error: {e}")
        sys.exit(1)


@model.command("list")
@click.pass_context
def list_models(ctx):
    """List models."""
    try:
        cmds = get_model_commands()
        result = cmds.list_models()
        if ctx.obj.get("json_output"):
            click.echo(format_json_output(result))
        else:
            if result["total"] == 0:
                click.echo("(no models)")
            else:
                for m in result["models"]:
                    click.echo(f"  {m['model_id']}: {m['type']}")
    except Exception as e:
        if ctx.obj.get("json_output"):
            click.echo(format_json_output(format_error(str(e))))
        else:
            click.echo(f"✗ Error: {e}")
        sys.exit(1)


@model.command("remove")
@click.option("--model-id", required=True, help="Model ID to remove")
@click.pass_context
def remove_model(ctx, model_id):
    """Remove a model from session storage."""
    try:
        cmds = get_model_commands()
        result = cmds.delete_model(model_id)
        if "error" in result:
            raise ValueError(result["error"])
        if ctx.obj.get("json_output"):
            click.echo(format_json_output(result))
        else:
            click.echo(f"✓ Removed model {model_id}")
    except Exception as e:
        if ctx.obj.get("json_output"):
            click.echo(format_json_output(format_error(str(e))))
        else:
            click.echo(f"✗ Error: {e}")
        sys.exit(1)


@model.command("save")
@click.option("--model-id", required=True, help="Model ID to save")
@click.option("--path", "-p", required=True, help="Save path")
@click.pass_context
def save_model(ctx, model_id, path):
    """Save model to file."""
    try:
        cmds = get_model_commands()
        result = cmds.save_model(model_id, path)
        if "error" in result:
            raise ValueError(result["error"])
        if ctx.obj.get("json_output"):
            click.echo(format_json_output(result))
        else:
            click.echo(f"✓ Saved model to {path}")
    except Exception as e:
        if ctx.obj.get("json_output"):
            click.echo(format_json_output(format_error(str(e))))
        else:
            click.echo(f"✗ Error: {e}")
        sys.exit(1)


@model.command("load")
@click.option("--path", "-p", required=True, help="Model file path")
@click.pass_context
def load_model(ctx, path):
    """Load model from file."""
    try:
        cmds = get_model_commands()
        result = cmds.load_model(path)
        if "error" in result:
            raise ValueError(result["error"])
        if ctx.obj.get("json_output"):
            click.echo(format_json_output(result))
        else:
            click.echo(f"✓ Loaded {result['model_type']} model: {result['model_id'][:8]}")
    except Exception as e:
        if ctx.obj.get("json_output"):
            click.echo(format_json_output(format_error(str(e))))
        else:
            click.echo(f"✗ Error: {e}")
        sys.exit(1)


@model.command("train")
@click.option("--model-id", required=True, help="Model ID to train")
@click.option("--epochs", "-e", type=int, default=10, help="Number of epochs")
@click.option("--lr", type=float, default=0.001, help="Learning rate")
@click.option("--batch-size", "-b", type=int, default=32, help="Batch size")
@click.option(
    "--device",
    type=click.Choice(["cpu", "gpu", "auto"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Device selection: cpu, gpu, or auto (gpu when Vulkan is working)",
)
@click.option("--optimizer", default="Adam", type=click.Choice(["Adam", "SGD"]), help="Optimizer type")
@click.pass_context
def train_model(ctx, model_id, epochs, lr, batch_size, device, optimizer):
    """Train a model."""
    try:
        cmds = get_model_commands()

        device = resolve_device(device)
        
        if ctx.obj.get("json_output"):
            click.echo('{"status": "training_started"}')
        
        result = cmds.train_model(
            model_id,
            epochs=epochs,
            learning_rate=lr,
            batch_size=batch_size,
            device=device,
            optimizer_type=optimizer,
        )
        
        if "error" in result:
            raise ValueError(result["error"])
        
        if ctx.obj.get("json_output"):
            click.echo(format_json_output(result))
        else:
            final_loss = result.get("final_loss", 0.0)
            click.echo(f"✓ Training complete")
            click.echo(f"  Epochs: {epochs}")
            click.echo(f"  Device: {device}")
            click.echo(f"  Final loss: {final_loss:.6f}")
    except Exception as e:
        if ctx.obj.get("json_output"):
            click.echo(format_json_output(format_error(str(e))))
        else:
            click.echo(f"✗ Error: {e}")
        sys.exit(1)


@model.command("combine")
@click.argument("model_a")
@click.argument("model_b")
@click.pass_context
def combine_models(ctx, model_a, model_b):
    """Combine two models sequentially (A then B)."""
    try:
        cmds = get_model_commands()
        result = cmds.combine_models(model_a, model_b)
        if "error" in result:
            raise ValueError(result["error"])

        if ctx.obj.get("json_output"):
            click.echo(format_json_output(result))
        else:
            click.echo(f"✓ Combined model: {result['model_id'][:8]}")
    except Exception as e:
        if ctx.obj.get("json_output"):
            click.echo(format_json_output(format_error(str(e))))
        else:
            click.echo(f"✗ Error: {e}")
        sys.exit(1)


@cli.command()
@click.pass_context
def chat(ctx):
    """Interactive chat mode."""
    repl = ChatREPL(device=ctx.obj.get("device", "cpu"))
    repl.run()


@cli.command()
@click.pass_context
def info(ctx):
    """Show info."""
    try:
        import numpy
        from .. import vulkan_backend as vk
        
        info_data = {
            "rasptorch_version": __version__,
            "numpy_version": numpy.__version__,
            "device": ctx.obj.get("device", "cpu"),
            "vulkan_available": vk.is_available(),
            "vulkan_using_real_gpu": vk.using_vulkan(),
        }

        info_data["vulkan_status"] = vk.disabled_reason() or ("Available" if vk.using_vulkan() else "Unavailable")
        
        if ctx.obj.get("json_output"):
            click.echo(format_json_output(info_data))
        else:
            click.echo(f"rasptorch: {info_data['rasptorch_version']}")
            click.echo(f"numpy: {info_data['numpy_version']}")
            click.echo(f"device: {info_data['device']}")
            if info_data["vulkan_using_real_gpu"]:
                click.echo(f"vulkan: ✓ Using GPU")
            else:
                click.echo(f"vulkan: ✗ Not using GPU")
                reason = vk.disabled_reason()
                if reason:
                    click.echo(f"  Reason: {reason}")
    except Exception as e:
        if ctx.obj.get("json_output"):
            click.echo(format_json_output(format_error(str(e))))
        else:
            click.echo(f"✗ Error: {e}")
        sys.exit(1)


@cli.command()
@click.option("--port", type=int, default=None, help="Port to run the Streamlit UI on")
@click.option("--server-headless", is_flag=True, default=True, show_default=True, help="Run Streamlit in headless mode")
@click.argument("streamlit_args", nargs=-1)
@click.pass_context
def ui(ctx: click.Context, port: int | None, server_headless: bool, streamlit_args: Tuple[str, ...]):
    """Launch the Streamlit UI."""
    # Resolve app path to the isometric SVG UI version in rasptorch/ui/app.py
    app_path = Path(__file__).resolve().parents[1] / "ui" / "app.py"
    if not app_path.exists():
        raise click.ClickException(f"UI entry not found at: {app_path}")

    cmd: List[str] = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    if server_headless:
        cmd.extend(["--server.headless", "true"])
    if port is not None:
        cmd.extend(["--server.port", str(port)])
    cmd.extend(list(streamlit_args))

    # Use subprocess so this works even when Streamlit isn't installed as a console script.
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    cli(obj={})
