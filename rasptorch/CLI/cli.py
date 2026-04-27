"""rasptorch CLI - Command-line interface for deep learning on Raspberry Pi."""

import click
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from .. import __version__
from ._cli_utils import parse_shape, format_error, format_json_output
from ._cli_commands import TensorCommands, get_model_commands
from ._cli_chat import ChatREPL
from ..utils import resolve_backend, resolve_device, backend_device_label


@click.group(invoke_without_command=True)
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.option(
    "--device",
    default="auto",
    type=click.Choice(["cpu", "gpu", "auto"], case_sensitive=False),
    show_default=True,
    help="Device selection: cpu, gpu, or auto (gpu when Vulkan is working)",
)
@click.option(
    "--backend",
    "backend_name",
    default="auto",
    type=click.Choice(["auto", "numpy", "vulkan", "opencl", "cuda"], case_sensitive=False),
    show_default=True,
    help="Backend selection: auto prefers Vulkan, otherwise NumPy",
)
@click.version_option(__version__)
@click.pass_context
def cli(ctx: click.Context, json_output: bool, device: str, backend_name: str):
    """rasptorch - Deep learning on Raspberry Pi 5."""
    ctx.ensure_object(dict)
    ctx.obj["json_output"] = json_output
    ctx.obj["backend"] = resolve_backend(backend_name)
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


@cli.group()
def backend():
    """Backend registry and connection commands."""
    pass


@backend.command("list")
@click.pass_context
def list_backends(ctx):
    """List registered backends and availability."""
    try:
        from ..backend import available_backends, get_backend, backend_manager

        registered = backend_manager.list_registered()
        availability = available_backends()
        active = get_backend().name
        active_label = "numpy" if active == "cpu" else active
        result = {
            "status": "success",
            "active": active_label,
            "backends": [
                {"name": ("numpy" if name == "cpu" else name), "available": bool(availability.get(name, False))}
                for name in registered
            ],
        }
        if ctx.obj.get("json_output"):
            click.echo(format_json_output(result))
        else:
            click.echo(f"Active backend: {active_label}")
            for item in result["backends"]:
                mark = "✓" if item["available"] else "✗"
                click.echo(f"  {mark} {item['name']}")
    except Exception as e:
        if ctx.obj.get("json_output"):
            click.echo(format_json_output(format_error(str(e))))
        else:
            click.echo(f"✗ Error: {e}")
        sys.exit(1)
@backend.command("connect")
@click.argument("name", type=click.Choice(["numpy", "vulkan", "opencl", "cuda"], case_sensitive=False))
@click.option("--strict", is_flag=True, help="Fail instead of falling back to CPU when backend is unavailable")
@click.pass_context
def connect_backend_cmd(ctx, name, strict):
    """Connect active compute backend."""
    try:
        from ..backend import connect_backend

        target = "cpu" if str(name).lower() == "numpy" else str(name).lower()
        backend_obj = connect_backend(target, strict=bool(strict))
        active_label = "numpy" if backend_obj.name == "cpu" else backend_obj.name
        result = {"status": "success", "active": active_label}
        if ctx.obj.get("json_output"):
            click.echo(format_json_output(result))
        else:
            click.echo(f"✓ Active backend: {active_label}")
    except Exception as e:
        if ctx.obj.get("json_output"):
            click.echo(format_json_output(format_error(str(e))))
        else:
            click.echo(f"✗ Error: {e}")
        sys.exit(1)


@backend.command("benchmark")
@click.option(
    "--backends",
    default="numpy,vulkan,opencl,cuda",
    show_default=True,
    help="Comma-separated backends to benchmark",
)
@click.option("--size", type=int, default=256, show_default=True, help="Square matrix size (N for NxN)")
@click.option("--iterations", type=int, default=20, show_default=True, help="Timed iterations per backend")
@click.option("--warmup", type=int, default=3, show_default=True, help="Warmup iterations per backend")
@click.option("--seed", type=int, default=42, show_default=True, help="Random seed for reproducible inputs")
@click.option(
    "--vulkan-kernel",
    type=click.Choice(
        ["auto", "matmul", "matmul_tiled", "matmul_vec4", "matmul_vec4_tiled", "matmul_vec4_wide_tiled", "matmul_a_bt", "matmul_a_bt_tiled"],
        case_sensitive=False,
    ),
    default="auto",
    show_default=True,
    help="Resident Vulkan kernel strategy, including new kernel options",
)
@click.option(
    "--vulkan-submit-every",
    type=int,
    default=8,
    show_default=True,
    help="Resident Vulkan mode: submit command buffer every N dispatches",
)
@click.option(
    "--vulkan-autotune-submit",
    is_flag=True,
    default=False,
    help="Probe multiple Vulkan submit chunk sizes and select the fastest stable one",
)
@click.pass_context
def benchmark_backends(
    ctx, backends, size, iterations, warmup, seed, vulkan_kernel, vulkan_submit_every, vulkan_autotune_submit
):
    """Benchmark backend matmul throughput with a reproducible workload."""
    if int(size) <= 0:
        raise click.ClickException("--size must be > 0")
    if int(iterations) <= 0:
        raise click.ClickException("--iterations must be > 0")
    if int(warmup) < 0:
        raise click.ClickException("--warmup must be >= 0")
    if int(vulkan_submit_every) <= 0:
        raise click.ClickException("--vulkan-submit-every must be > 0")

    requested = [b.strip().lower() for b in str(backends).split(",") if b.strip()]
    if not requested:
        raise click.ClickException("No backends specified")
    valid = {"numpy", "vulkan", "opencl", "cuda"}
    for name in requested:
        if name not in valid:
            raise click.ClickException(f"Unknown backend in --backends: {name}")

    try:
        from ..backend import connect_backend
        from .. import vulkan_backend as vk

        rng = np.random.default_rng(int(seed))
        a = rng.standard_normal((int(size), int(size)), dtype=np.float32)
        b = rng.standard_normal((int(size), int(size)), dtype=np.float32)

        results = []
        for label in requested:
            target = "cpu" if label == "numpy" else label
            try:
                backend_obj = connect_backend(target, strict=True)

                if label == "vulkan":
                    # Resident-buffer path: upload once, repeated on-device matmul, download once.
                    a_buf = vk.to_gpu(a)
                    b_buf = vk.to_gpu(b)
                    out_buf = vk.empty((int(size), int(size)))
                    b_t_buf = None
                    try:
                        def _run_vulkan_dispatches(count: int, dispatch_fn) -> None:
                            _run_vulkan_dispatches_chunked(count, dispatch_fn, int(vulkan_submit_every))

                        def _run_vulkan_dispatches_chunked(count: int, dispatch_fn, submit_every: int) -> None:
                            remaining = int(count)
                            chunk = int(submit_every)
                            while remaining > 0:
                                step = min(chunk, remaining)
                                vk.begin_batch()
                                try:
                                    for _ in range(step):
                                        dispatch_fn()
                                finally:
                                    vk.end_batch()
                                remaining -= step

                        requested_kernel = str(vulkan_kernel).lower()
                        if requested_kernel == "auto" and int(size) >= 4096:
                            requested_kernel = "matmul_vec4_wide_tiled"
                        dispatch_variants = {
                            "matmul_tiled": (lambda: vk.matmul_tiled_into(a_buf, b_buf, out_buf)),
                            "matmul_vec4_wide_tiled": (lambda: vk.matmul_vec4_wide_tiled_into(a_buf, b_buf, out_buf)),
                            "matmul_vec4_tiled": (lambda: vk.matmul_vec4_tiled_into(a_buf, b_buf, out_buf)),
                            "matmul": (lambda: vk.matmul_into(a_buf, b_buf, out_buf)),
                            "matmul_vec4": (lambda: vk.matmul_vec4_into(a_buf, b_buf, out_buf)),
                        }
                        allow_transpose_aware_auto = requested_kernel != "auto" or int(size) < 4096
                        if requested_kernel in {"auto", "matmul_a_bt", "matmul_a_bt_tiled"} and allow_transpose_aware_auto:
                            b_t_buf = vk.transpose2d(b_buf)
                            dispatch_variants["matmul_a_bt"] = (lambda: vk.matmul_a_bt_out(a_buf, b_t_buf, out_buf))
                            dispatch_variants["matmul_a_bt_tiled"] = (
                                lambda: vk.matmul_a_bt_tiled_out(a_buf, b_t_buf, out_buf)
                            )

                        selected_kernel = requested_kernel
                        selected_submit_every = int(vulkan_submit_every)
                        if requested_kernel == "auto":
                            best_kernel = None
                            best_chunk = selected_submit_every
                            best_time = float("inf")
                            probe_dispatches = max(4, min(24, int(iterations) // 2 or 4))
                            candidates = [1, 2, 4, 8, 16] if bool(vulkan_autotune_submit) else [1, 2, 4, 8]
                            for kernel_name, fn in dispatch_variants.items():
                                for candidate in candidates:
                                    try:
                                        t_start = time.perf_counter()
                                        _run_vulkan_dispatches_chunked(probe_dispatches, fn, candidate)
                                        t_elapsed = float(time.perf_counter() - t_start)
                                        if t_elapsed < best_time:
                                            best_time = t_elapsed
                                            best_kernel = kernel_name
                                            best_chunk = candidate
                                    except Exception:
                                        continue
                            if best_kernel is None:
                                raise RuntimeError("No Vulkan kernel variant succeeded during auto probe")
                            selected_kernel = str(best_kernel)
                            selected_submit_every = int(best_chunk)
                            dispatch_fn = dispatch_variants[selected_kernel]
                        else:
                            dispatch_fn = dispatch_variants[selected_kernel]

                        if bool(vulkan_autotune_submit) and requested_kernel != "auto":
                            candidates = [1, 2, 4, 8, 16]
                            best_chunk = selected_submit_every
                            best_time = float("inf")
                            probe_dispatches = max(2, min(8, int(iterations)))
                            for candidate in candidates:
                                try:
                                    t_start = time.perf_counter()
                                    _run_vulkan_dispatches_chunked(probe_dispatches, dispatch_fn, candidate)
                                    t_elapsed = float(time.perf_counter() - t_start)
                                    if t_elapsed < best_time:
                                        best_time = t_elapsed
                                        best_chunk = candidate
                                except Exception:
                                    continue
                            selected_submit_every = int(best_chunk)

                        if int(warmup) > 0:
                            _run_vulkan_dispatches_chunked(int(warmup), dispatch_fn, selected_submit_every)
                        start = time.perf_counter()
                        _run_vulkan_dispatches_chunked(int(iterations), dispatch_fn, selected_submit_every)
                        elapsed = float(time.perf_counter() - start)

                        out = vk.to_cpu(out_buf)
                        checksum = float(np.asarray(out, dtype=np.float32).reshape(-1)[0]) * float(int(iterations))
                    finally:
                        vk.free(a_buf)
                        vk.free(b_buf)
                        if b_t_buf is not None:
                            vk.free(b_t_buf)
                        vk.free(out_buf)
                else:
                    # Warmup includes setup/JIT/driver overhead and is excluded from timing.
                    for _ in range(int(warmup)):
                        backend_obj.matmul(a, b)

                    checksum = 0.0
                    start = time.perf_counter()
                    for _ in range(int(iterations)):
                        out = backend_obj.matmul(a, b)
                        checksum += float(np.asarray(out, dtype=np.float32).reshape(-1)[0])
                    elapsed = float(time.perf_counter() - start)

                iters_per_sec = float(int(iterations) / elapsed) if elapsed > 0 else 0.0
                gflops = float((2.0 * (int(size) ** 3) * int(iterations)) / elapsed / 1e9) if elapsed > 0 else 0.0
                result_item = {
                    "backend": label,
                    "status": "ok",
                    "elapsed_seconds": elapsed,
                    "iterations": int(iterations),
                    "iterations_per_second": iters_per_sec,
                    "estimated_gflops": gflops,
                    "checksum": checksum,
                }
                if label == "vulkan":
                    result_item["mode"] = "resident"
                    result_item["submit_every"] = int(selected_submit_every)
                    result_item["kernel"] = selected_kernel
                results.append(result_item)
            except Exception as e:
                error_msg = str(e).strip()
                if not error_msg:
                    error_msg = f"{type(e).__name__}: {repr(e)}".strip()
                if label == "vulkan":
                    try:
                        reason = vk.disabled_reason()
                    except Exception:
                        reason = None
                    if reason:
                        reason_text = str(reason).strip()
                        if reason_text and reason_text not in error_msg:
                            error_msg = f"{error_msg} | {reason_text}" if error_msg else reason_text
                results.append(
                    {
                        "backend": label,
                        "status": "unavailable",
                        "error": error_msg,
                    }
                )
                continue

        payload = {
            "status": "success",
            "workload": {
                "op": "matmul",
                "size": int(size),
                "iterations": int(iterations),
                "warmup": int(warmup),
                "seed": int(seed),
            },
            "results": results,
        }

        if ctx.obj.get("json_output"):
            click.echo(format_json_output(payload))
            return

        click.echo(
            f"Backend benchmark (matmul {size}x{size}, iterations={iterations}, warmup={warmup}, seed={seed})"
        )
        for item in results:
            if item["status"] != "ok":
                click.echo(f"  ✗ {item['backend']}: unavailable ({item['error']})")
                continue
            click.echo(
                f"  ✓ {item['backend']}: {item['elapsed_seconds']:.4f}s, "
                f"{item['iterations_per_second']:.2f} iter/s, {item['estimated_gflops']:.2f} GFLOP/s"
            )
    except Exception as e:
        if ctx.obj.get("json_output"):
            click.echo(format_json_output(format_error(str(e))))
        else:
            click.echo(f"✗ Error: {e}")
        sys.exit(1)


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


@model.command("convert-legacy")
@click.option("--src", "-s", required=True, help="Legacy torch checkpoint (.pt/.pth)")
@click.option("--dst", "-d", required=True, help="Output rasptorch checkpoint (.pt/.pth)")
@click.pass_context
def convert_legacy_model(ctx, src, dst):
    """Convert a legacy torch checkpoint to rasptorch format."""
    try:
        cmds = get_model_commands()
        result = cmds.convert_legacy_checkpoint(src, dst)
        if "error" in result:
            raise ValueError(result["error"])
        if ctx.obj.get("json_output"):
            click.echo(format_json_output(result))
        else:
            click.echo(f"✓ Converted {src} -> {dst}")
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
        from ..backend import get_backend

        active_backend = get_backend().name
        active_device = backend_device_label(active_backend)
        
        info_data = {
            "rasptorch_version": __version__,
            "numpy_version": numpy.__version__,
            "active_device": active_device,
            "active_backend": ("numpy" if active_backend == "cpu" else active_backend),
            "vulkan_available": vk.is_available(),
            "vulkan_using_real_gpu": vk.using_vulkan(),
        }

        info_data["vulkan_status"] = vk.disabled_reason() or ("Available" if vk.using_vulkan() else "Unavailable")
        
        if ctx.obj.get("json_output"):
            click.echo(format_json_output(info_data))
        else:
            click.echo(f"rasptorch: {info_data['rasptorch_version']}")
            click.echo(f"numpy: {info_data['numpy_version']}")
            click.echo(f"device: {active_device}")
            click.echo(f"backend: {info_data['active_backend']}")
            if info_data["vulkan_using_real_gpu"]:
                click.echo(f"vulkan: ✓ Using GPU")
            else:
                click.echo(f"vulkan: ✗ Not using GPU")
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
    app_path = Path(__file__).resolve().parent / "ui" / "app.py"
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
