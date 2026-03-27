from __future__ import annotations as annotate
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import json
import sys
import tempfile
import hashlib
import streamlit as st
import numpy as np

try:
    import plotly.graph_objects as go  # type: ignore
except Exception:  # pragma: no cover
    go = None  # type: ignore[assignment]

_UI_BUILD = "2026-03-27-activation-conditional-v1"


# When running `streamlit run` from inside `rasptorch/ui`, Python's import
# root is that folder, which can break `import rasptorch`. Ensure repo root is
# importable.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


rasptorch: Any = None
get_model_commands: Any = None
_vulkan_backend: Any = None
_HAS_RASPTORCH = False
_RASPTORCH_IMPORT_ERROR = ""

try:
    import rasptorch as rasptorch  # type: ignore[no-redef]
    from rasptorch._cli_commands import get_model_commands as get_model_commands  # type: ignore[no-redef]

    try:
        from rasptorch import vulkan_backend as _vulkan_backend  # type: ignore
    except Exception:
        _vulkan_backend = None

    _HAS_RASPTORCH = True
except Exception as e:
    _HAS_RASPTORCH = False
    _RASPTORCH_IMPORT_ERROR = str(e)
    _vulkan_backend = None


DEVICES = ["cpu", "gpu"]

# UI constants (some pages reference these by name).
ACTIVATIONS = [
    "relu",
    "leaky_relu",
    "elu",
    "gelu",
    "tanh",
    "sigmoid",
    "none",
]

OPTIMIZERS = ["Adam", "AdamW", "SGD", "RMSProp"]

HELP_TEXT = [
    "Commands:",
    "  help                          Show this help",
    "  info                          Show environment/device info",
    "  clear                         Clear console output",
    "  device cpu|gpu|status         Select device or show status",
    "  model list                    List available models",
    "  model select <id>             Select current model",
    "  model mlp <layers_csv> [act=relu|activations=a,b,c]",
    "  model linear <in> <hidden_csv> <out> [act=relu|activations=a,b,c]",
    "  model cnn <in_ch> <out_ch_csv> [kernels_csv] [act=relu|activations=a,b,c]",
    "  combine sequential <a_id> <b_id> [name]",
    "  train epochs|batch-size|lr <value>",
    "  train start                   Train the selected model",
    "  save <path>                   Save selected model (backend-dependent)",
]


TensorCommands: Any = None
if _HAS_RASPTORCH:
    try:
        from rasptorch._cli_commands import TensorCommands as TensorCommands  # type: ignore
    except Exception:
        TensorCommands = None


def _mermaid_escape(text: str) -> str:
    return str(text).replace("\n", " ").replace("\"", "'")


def _model_mermaid_diagram(models: Dict[str, Any], model_id: Optional[str]) -> Optional[str]:
    if not model_id or model_id not in models:
        return None

    md = models.get(model_id) or {}
    mtype = str(md.get("type", "Unknown"))
    cfg = md.get("config") or {}

    lines: List[str] = ["flowchart LR"]
    root = f"m_{_mermaid_escape(model_id)}"

    def node(nid: str, label: str) -> None:
        lines.append(f"  {nid}[\"{_mermaid_escape(label)}\"]")

    def edge(a: str, b: str, label: Optional[str] = None) -> None:
        if label:
            lines.append(f"  {a} -->|{_mermaid_escape(label)}| {b}")
        else:
            lines.append(f"  {a} --> {b}")

    if mtype == "Combined" and str(cfg.get("combine")) == "sequential":
        a_id = str(cfg.get("model_a_id", ""))
        b_id = str(cfg.get("model_b_id", ""))
        a_type = str(cfg.get("model_a_type", "A"))
        b_type = str(cfg.get("model_b_type", "B"))
        node(root, f"Combined {model_id[:8]}")
        a_node = f"{root}_a"
        b_node = f"{root}_b"
        node(a_node, f"{a_type} {a_id[:8]}")
        node(b_node, f"{b_type} {b_id[:8]}")
        edge(a_node, root, "A")
        edge(b_node, root, "B")
        io_in = cfg.get("input_size")
        io_out = cfg.get("output_size")
        if io_in is not None or io_out is not None:
            io_node = f"{root}_io"
            node(io_node, f"I/O: {io_in if io_in is not None else '?'} → {io_out if io_out is not None else '?'}")
            edge(root, io_node)
        return "\n".join(lines)

    if mtype == "MLP":
        sizes = cfg.get("layer_sizes") or []
        act = cfg.get("activation")
        acts = cfg.get("activations")
        act_label = None
        if isinstance(acts, list) and acts:
            act_label = "acts: " + ",".join(str(a) for a in acts)
        elif act is not None:
            act_label = f"act: {act}"

        header = f"MLP {model_id[:8]}"
        if act_label:
            header += f" ({act_label})"
        node(root, header)
        prev = root
        for i, sz in enumerate(sizes):
            n = f"{root}_l{i}"
            node(n, f"{sz}")
            edge(prev, n)
            prev = n
        return "\n".join(lines)

    if mtype == "Linear":
        node(root, f"Linear {model_id[:8]}")
        inp = cfg.get("input_size")
        hidden = cfg.get("hidden_sizes") or []
        out = cfg.get("output_size")
        prev = root
        n0 = f"{root}_in"
        node(n0, f"in: {inp}")
        edge(prev, n0)
        prev = n0
        for i, h in enumerate(hidden):
            n = f"{root}_h{i}"
            node(n, f"hidden: {h}")
            edge(prev, n)
            prev = n
        n1 = f"{root}_out"
        node(n1, f"out: {out}")
        edge(prev, n1)
        return "\n".join(lines)

    if mtype == "CNN":
        node(root, f"CNN {model_id[:8]}")
        in_ch = cfg.get("in_channels")
        out_ch = cfg.get("out_channels") or []
        kernels = cfg.get("kernels")
        prev = root
        n0 = f"{root}_in"
        node(n0, f"in_ch: {in_ch}")
        edge(prev, n0)
        prev = n0
        for i, ch in enumerate(out_ch):
            n = f"{root}_c{i}"
            k = None
            if isinstance(kernels, list) and i < len(kernels):
                k = kernels[i]
            node(n, f"ch: {ch}" + (f" (k={k})" if k is not None else ""))
            edge(prev, n)
            prev = n
        return "\n".join(lines)

    node(root, f"{mtype} {model_id[:8]}")
    return "\n".join(lines)


def _model_plotly_3d_figure(models: Dict[str, Any], model_id: Optional[str]):
    """Create a simple 3D schematic (stacked boxes) for the model.

    This is an approximate visualization intended for intuition, not exact tensor shapes.
    Returns None if Plotly isn't available or model is missing.
    """
    if go is None or not model_id or model_id not in models:
        return None

    # Keep the 3D diagram background consistent with the app panel.
    # Plotly uses separate background colors for the paper and the 3D scene.
    plot_bg = "rgba(35,38,45,1.0)"

    md = models.get(model_id) or {}
    mtype = str(md.get("type", "Unknown"))
    cfg = md.get("config") or {}

    # Special-case Combined(sequential): render as two stacks (left=A+Combined, right=B)
    # with a left-to-right flow indicator. This makes the execution direction explicit.
    if mtype == "Combined" and str(cfg.get("combine")) == "sequential":
        a_id = str(cfg.get("model_a_id", ""))
        b_id = str(cfg.get("model_b_id", ""))
        a_t = str(cfg.get("model_a_type", "A"))
        b_t = str(cfg.get("model_b_type", "B"))
        io_in = cfg.get("input_size")
        io_out = cfg.get("output_size")

        left_sizes: List[float] = []
        left_labels: List[str] = []
        left_hovers: List[str] = []
        right_sizes: List[float] = []
        right_labels: List[str] = []
        right_hovers: List[str] = []

        def _coerce_size(v: Any) -> float:
            try:
                fv = float(v)
            except Exception:
                fv = 1.0
            return max(1.0, fv)

        # Left stack: A then composition boundary.
        left_sizes.append(_coerce_size(io_in or 8))
        left_labels.append(f"A {a_t}")
        left_hovers.append(
            "type=Combined(sequential)"
            f"<br>direction=A → B"
            f"<br>A={a_t} ({a_id[:8] if a_id else '?'})"
            f"<br>input_size={io_in}"
        )
        left_sizes.append(_coerce_size(io_out or (io_in or 8)))
        left_labels.append("Combined")
        left_hovers.append(
            "type=Combined(sequential)"
            f"<br>direction=A → B"
            f"<br>output_size={io_out}"
        )

        # Right stack: B.
        right_sizes.append(_coerce_size(io_out or 8))
        right_labels.append(f"B {b_t}")
        right_hovers.append(
            "type=Combined(sequential)"
            f"<br>direction=A → B"
            f"<br>B={b_t} ({b_id[:8] if b_id else '?'})"
            f"<br>output_size={io_out}"
        )

        max_sz = max(left_sizes + right_sizes) if (left_sizes or right_sizes) else 1.0
        left_dims = [0.6 + 2.4 * (s / max_sz) for s in left_sizes]
        right_dims = [0.6 + 2.4 * (s / max_sz) for s in right_sizes]

        fig = go.Figure()
        label_x: List[float] = []
        label_y: List[float] = []
        label_z: List[float] = []
        label_text: List[str] = []

        def _add_box(center_x: float, z0: float, d: float, hov: str) -> float:
            w = d
            h = d
            t = 0.8
            x0, x1 = center_x - w / 2, center_x + w / 2
            y0, y1 = -h / 2, h / 2
            z1 = z0 + t

            xs = [x0, x1, x1, x0, x0, x1, x1, x0]
            ys = [y0, y0, y1, y1, y0, y0, y1, y1]
            zs = [z0, z0, z0, z0, z1, z1, z1, z1]

            I = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3]
            J = [1, 2, 5, 6, 4, 7, 5, 6, 3, 6, 0, 4]
            K = [2, 3, 6, 7, 5, 6, 6, 2, 6, 7, 4, 7]

            fig.add_trace(
                go.Mesh3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    i=I,
                    j=J,
                    k=K,
                    opacity=0.65,
                    hovertext=hov,
                    hoverinfo="text",
                    showscale=False,
                )
            )
            return z1

        # Spacing between the two stacks and between boxes.
        # Combined tends to have many layers; give it extra room.
        stack_dx = 3.6
        box_gap = 0.45
        z_left = 0.0
        left_centers: List[float] = []
        for d, lab, hov in zip(left_dims, left_labels, left_hovers):
            z1 = _add_box(center_x=-stack_dx, z0=z_left, d=d, hov=hov)
            label_x.append(-stack_dx)
            label_y.append(0.0)
            label_z.append((z_left + z1) / 2)
            label_text.append(lab)
            left_centers.append((z_left + z1) / 2)
            z_left = z1 + box_gap

        z_right = 0.0
        right_centers: List[float] = []
        for d, lab, hov in zip(right_dims, right_labels, right_hovers):
            z1 = _add_box(center_x=stack_dx, z0=z_right, d=d, hov=hov)
            label_x.append(stack_dx)
            label_y.append(0.0)
            label_z.append((z_right + z1) / 2)
            label_text.append(lab)
            right_centers.append((z_right + z1) / 2)
            z_right = z1 + box_gap

        # Flow arrow: from left-stack "Combined" level to right-stack "B" level.
        # Relationship connectors (lighter gray to match UI tone).
        conn_color = "rgba(200,200,200,0.75)"

        # 1) Vertical connector: A stack flow (left)
        if len(left_centers) >= 2:
            fig.add_trace(
                go.Scatter3d(
                    x=[-stack_dx + 0.55, -stack_dx + 0.55],
                    y=[0.0, 0.0],
                    z=[left_centers[0], left_centers[1]],
                    mode="lines+markers+text",
                    line=dict(width=6, color=conn_color),
                    marker=dict(size=4, color=conn_color),
                    text=["", "A → …"],
                    textposition="top center",
                    textfont=dict(size=13, color=conn_color),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        # 2) Horizontal connector: boundary → B stack
        # Anchor from the rendered "Combined" block (left stack, index 1) to the rendered B block (right stack, index 0).
        left_flow_z = left_centers[1] if len(left_centers) >= 2 else (left_centers[0] if left_centers else 0.4)
        right_flow_z = right_centers[0] if right_centers else 0.4
        fig.add_trace(
            go.Scatter3d(
                x=[-stack_dx + 0.35, stack_dx - 0.35],
                y=[0.0, 0.0],
                z=[left_flow_z, right_flow_z],
                mode="lines+markers+text",
                line=dict(width=6, color=conn_color),
                marker=dict(size=4, color=conn_color),
                text=["", "A → B"],
                textposition="top center",
                textfont=dict(size=13, color=conn_color),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=label_x,
                y=label_y,
                z=label_z,
                mode="text",
                text=label_text,
                textposition="middle center",
                textfont=dict(size=16, color="white"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        fig.update_layout(
            height=520,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor=plot_bg,
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor=plot_bg,
            ),
            showlegend=False,
        )
        return fig

    sizes: List[float] = []
    labels: List[str] = []
    hovers: List[str] = []

    def _add(sz: Any, label: str, hover: Optional[str] = None) -> None:
        try:
            v = float(sz)
        except Exception:
            v = 1.0
        sizes.append(max(1.0, v))
        labels.append(label)
        hovers.append(hover or label)

    def _act_summary() -> str:
        acts = cfg.get("activations")
        act = cfg.get("activation")
        if isinstance(acts, list) and acts:
            return "activations=" + ",".join(str(a) for a in acts)
        if act is not None:
            return f"activation={act}"
        return ""

    if mtype == "MLP":
        act_s = _act_summary()
        for i, s in enumerate(cfg.get("layer_sizes") or []):
            hover = f"type=MLP<br>layer={i}<br>size={s}"
            if act_s:
                hover += f"<br>{act_s}"
            _add(s, f"L{i}: {s}", hover=hover)
    elif mtype == "Linear":
        act_s = _act_summary()
        inp = cfg.get("input_size")
        hover = f"type=Linear<br>input_size={inp}"
        if act_s:
            hover += f"<br>{act_s}"
        _add(inp or 1, f"in: {inp}", hover=hover)
        for i, h in enumerate(cfg.get("hidden_sizes") or []):
            hover = f"type=Linear<br>hidden[{i}]={h}"
            if act_s:
                hover += f"<br>{act_s}"
            _add(h, f"h{i}: {h}", hover=hover)
        out = cfg.get("output_size")
        hover = f"type=Linear<br>output_size={out}"
        if act_s:
            hover += f"<br>{act_s}"
        _add(out or 1, f"out: {out}", hover=hover)
    elif mtype == "CNN":
        act_s = _act_summary()
        kernels = cfg.get("kernels")
        in_ch = cfg.get("in_channels")
        hover = f"type=CNN<br>in_channels={in_ch}"
        if act_s:
            hover += f"<br>{act_s}"
        _add(in_ch or 1, f"in_ch: {in_ch}", hover=hover)
        for i, ch in enumerate(cfg.get("out_channels") or []):
            k = None
            if isinstance(kernels, list) and i < len(kernels):
                k = kernels[i]
            lab = f"c{i}: {ch}" + (f" (k={k})" if k is not None else "")
            hover = f"type=CNN<br>out_channels[{i}]={ch}"
            if k is not None:
                hover += f"<br>kernel={k}"
            if act_s:
                hover += f"<br>{act_s}"
            _add(ch, lab, hover=hover)
    else:
        io_in = cfg.get("input_size")
        _add(io_in or 8, mtype, hover=f"type={mtype}<br>input_size={io_in}")

    if not sizes:
        return None

    max_sz = max(sizes) if sizes else 1.0
    dims = [0.6 + 2.4 * (s / max_sz) for s in sizes]

    fig = go.Figure()
    z0 = 0.0
    label_x: List[float] = []
    label_y: List[float] = []
    label_z: List[float] = []
    label_text: List[str] = []
    for d, lab, hov in zip(dims, labels, hovers):
        w = d
        h = d
        t = 0.8
        x0, x1 = -w / 2, w / 2
        y0, y1 = -h / 2, h / 2
        z1 = z0 + t

        xs = [x0, x1, x1, x0, x0, x1, x1, x0]
        ys = [y0, y0, y1, y1, y0, y0, y1, y1]
        zs = [z0, z0, z0, z0, z1, z1, z1, z1]

        # 12 triangles (2 per face)
        I = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3]
        J = [1, 2, 5, 6, 4, 7, 5, 6, 3, 6, 0, 4]
        K = [2, 3, 6, 7, 5, 6, 6, 2, 6, 7, 4, 7]

        fig.add_trace(
            go.Mesh3d(
                x=xs,
                y=ys,
                z=zs,
                i=I,
                j=J,
                k=K,
                opacity=0.65,
                hovertext=hov,
                hoverinfo="text",
                showscale=False,
            )
        )
        label_x.append(0.0)
        label_y.append(0.0)
        label_z.append((z0 + z1) / 2)
        label_text.append(lab)

        z0 = z1 + 0.3

    # Draw labels as one trace so we can control font globally.
    fig.add_trace(
        go.Scatter3d(
            x=label_x,
            y=label_y,
            z=label_z,
            mode="text",
            text=label_text,
            textposition="middle center",
            textfont=dict(size=16, color="white"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor=plot_bg,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor=plot_bg,
        ),
        showlegend=False,
    )
    return fig


def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("nav", "Models")
    ss.setdefault("repl_log", [])
    ss.setdefault(
        "repl_context",
        {
            "current_model": None,
            "train_epochs": 5,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "device": "cpu",
            "loaded_files": {},
            "upload_cache": {},
            "last_loaded_hash": None,
            # Models that have successfully completed at least one training run in this UI session.
            "trained_models": [],
        },
    )


def _append_log(lines: List[str] | str) -> None:
    if isinstance(lines, str):
        st.session_state.repl_log.append(lines)
    else:
        st.session_state.repl_log.extend(lines)


def _parse_csv_ints(raw: str) -> List[int]:
    items = [x.strip() for x in str(raw).split(",") if x.strip()]
    return [int(x) for x in items]


def _parse_builder_args(tokens: List[str]) -> Dict[str, Any]:
    activation = "relu"
    activations: Optional[List[str]] = None
    for tok in tokens:
        t = str(tok).strip()
        if t.startswith("activations="):
            activations = [x.strip() for x in t.split("=", 1)[1].split(",") if x.strip()]
        elif t.startswith("act=") or t.startswith("activation="):
            activation = t.split("=", 1)[1].strip()
        else:
            if "," in t:
                activations = [x.strip() for x in t.split(",") if x.strip()]
            else:
                activation = t
    return {"activation": activation, "activations": activations}


def _safe_vulkan_status() -> Tuple[bool, Optional[str]]:
    if not _HAS_RASPTORCH or _vulkan_backend is None:
        return False, "rasptorch not available"
    try:
        if _vulkan_backend.using_vulkan():
            return True, None
        return False, _vulkan_backend.disabled_reason()
    except (RuntimeError, OSError, ValueError, AttributeError) as e:
        return False, str(e)


def _try_set_device(device: str) -> Tuple[bool, str]:
    device = str(device).lower().strip()
    ctx = st.session_state.repl_context

    if device == "cpu":
        ctx["device"] = "cpu"
        return True, "Device set to CPU (Vulkan, if initialized, stays available until restart)"

    if device != "gpu":
        ctx["device"] = "cpu"
        return False, f"Unknown device '{device}'"

    try:
        _vulkan_backend.init(strict=True)
        ctx["device"] = "gpu"
        return True, "Device set to GPU (Vulkan)"
    except (RuntimeError, OSError, ValueError) as e:
        ctx["device"] = "cpu"
        reason = None
        try:
            reason = _vulkan_backend.disabled_reason()
        except (RuntimeError, OSError, ValueError):
            reason = None
        msg = f"Vulkan GPU init failed: {e}"
        if reason:
            msg = f"{msg}\nReason: {reason}"
        return False, msg


def _cmd_info() -> str:
    lines = []
    try:
        lines.append(f"rasptorch version: {getattr(rasptorch, '__version__', '?')}")
    except Exception:
        lines.append("rasptorch version: ?")
    lines.append(f"numpy version: {np.__version__}")
    ok, reason = _safe_vulkan_status()
    device = str(st.session_state.repl_context.get("device", "cpu"))
    lines.append(f"device: {device}")
    if ok:
        lines.append("vulkan backend: initialized" + (" (in use)" if device == "gpu" else " (available)"))
    else:
        lines.append("vulkan backend: unavailable")
        if reason:
            lines.append(f"  Reason: {reason}")
    return "\n".join(lines)


def _handle_repl_command(command_str: str) -> None:
    command_str = (command_str or "").strip()
    if not command_str:
        return

    ctx = st.session_state.repl_context
    cmds = get_model_commands()
    parts = command_str.split()
    root = parts[0].lower()

    def out(s: str) -> None:
        _append_log(s)

    def selected_model() -> Optional[str]:
        mid = ctx.get("current_model")
        if mid and mid in cmds.models:
            return mid
        return None

    try:
        if root == "help":
            out(HELP_TEXT)
            return
        if root == "info":
            out(_cmd_info())
            return
        if root == "clear":
            st.session_state.repl_log = []
            return

        if root == "device":
            if len(parts) < 2:
                out("✗ Usage: device cpu|gpu|status")
                return
            sub = parts[1].lower()
            if sub == "status":
                device = str(ctx.get("device", "cpu"))
                out(f"Current device: {device.upper()}")
                ok, reason = _safe_vulkan_status()
                if ok:
                    out("Vulkan: available")
                else:
                    out(f"Vulkan: unavailable ({reason})")
                return
            if sub == "cpu":
                ctx["device"] = "cpu"
                out("✓ Device set to: CPU")
                return
            if sub == "gpu":
                try:
                    _vulkan_backend.init(strict=True)
                    ctx["device"] = "gpu"
                    out("✓ Device set to: GPU (Vulkan)")
                except (RuntimeError, OSError, ValueError) as e:
                    ctx["device"] = "cpu"
                    out(f"✗ Vulkan GPU init failed: {e}")
                return
            out(f"✗ Unknown device command: {sub}")
            return

        if root == "optimizer":
            if len(parts) < 2:
                out("✗ Usage: optimizer create|set-lr <value>")
                return
            sub = parts[1].lower()
            if sub == "create":
                opt = parts[2] if len(parts) > 2 else "Adam"
                opt_norm = "Adam" if opt.lower() == "adam" else ("SGD" if opt.lower() == "sgd" else None)
                if opt_norm is None:
                    out("✗ Unknown optimizer. Use Adam or SGD")
                    return
                ctx["optimizer"] = opt_norm
                out(f"✓ Selected optimizer: {opt_norm}")
                return
            if sub == "set-lr":
                if len(parts) < 3:
                    out("✗ Usage: optimizer set-lr <value>")
                    return
                ctx["learning_rate"] = float(parts[2])
                out(f"✓ Set learning rate: {ctx['learning_rate']}")
                return
            out(f"✗ Unknown optimizer command: {sub}")
            return

        if root == "train":
            if len(parts) < 2:
                out("✗ Usage: train epochs|batch-size|lr|start <value>")
                return
            sub = parts[1].lower()
            if sub == "epochs":
                ctx["train_epochs"] = int(parts[2])
                out(f"✓ Set epochs: {ctx['train_epochs']}")
                return
            if sub == "batch-size":
                ctx["batch_size"] = int(parts[2])
                out(f"✓ Set batch size: {ctx['batch_size']}")
                return
            if sub in ("lr", "learning-rate", "learning_rate"):
                ctx["learning_rate"] = float(parts[2])
                out(f"✓ Set learning rate: {ctx['learning_rate']}")
                return
            if sub == "start":
                mid = selected_model()
                if not mid:
                    out("✗ No model selected")
                    return
                out(f"🚀 Training {mid[:8]}...")
                result = cmds.train_model(
                    mid,
                    epochs=int(ctx.get("train_epochs", 5)),
                    learning_rate=float(ctx.get("learning_rate", 0.001)),
                    batch_size=int(ctx.get("batch_size", 32)),
                    device=str(ctx.get("device", "cpu")),
                    optimizer_type=str(ctx.get("optimizer", "Adam")),
                )
                if "error" in result:
                    out(f"✗ Error: {result['error']}")
                    return
                out(f"✓ Training complete. Final loss: {result.get('final_loss', 0.0):.6f}")
                return
            out(f"✗ Unknown train command: {sub}")
            return

        if root == "tensor":
            if len(parts) < 3:
                out("✗ Usage: tensor create|zeros|ones <shape_csv>")
                return
            sub = parts[1].lower()
            shape = tuple(int(x.strip()) for x in parts[2].split(",") if x.strip())
            device = str(ctx.get("device", "cpu"))
            if sub == "create":
                result = TensorCommands.create_random(shape, device=device)
            elif sub == "zeros":
                result = TensorCommands.create_zeros(shape, device=device)
            elif sub == "ones":
                result = TensorCommands.create_ones(shape, device=device)
            else:
                out(f"✗ Unknown tensor command: {sub}")
                return
            out(f"✓ Created {sub} tensor: {result.get('shape')} (id: {str(result.get('tensor_id','?'))[:8]})")
            return

        if root == "model":
            if len(parts) < 2:
                out("✗ Usage: model <subcommand>")
                return
            sub = parts[1].lower()

            if sub == "list":
                res = cmds.list_models()
                if res.get("total", 0) == 0:
                    out("No models yet")
                    return
                lines = [f"Models ({res['total']}):"]
                for m in res.get("models", []):
                    lines.append(f"- {m.get('model_id','')[:8]}: {m.get('type')}")
                out("\n".join(lines))
                return

            if sub in ("use", "select"):
                if len(parts) < 3:
                    out("✗ Usage: model use <id>")
                    return
                mid = parts[2]
                if mid in cmds.models:
                    ctx["current_model"] = mid
                    out(f"✓ Selected: {mid[:8]}")
                else:
                    out(f"✗ Model not found: {mid}")
                return

            if sub == "deselect":
                old = ctx.get("current_model")
                ctx["current_model"] = None
                out(f"✓ Deselected model: {str(old)[:8]}")
                return

            if sub == "info":
                mid = selected_model()
                if not mid:
                    out("✗ No model selected")
                    return
                md = cmds.models.get(mid, {})
                payload = {
                    "model_id": mid,
                    "type": md.get("type", "Unknown"),
                    "config": md.get("config", {}),
                }
                out(json.dumps(payload, indent=2, default=str))
                return

            if sub == "combine":
                if len(parts) < 4:
                    out("✗ Usage: model combine <id_a> <id_b>")
                    return
                res = cmds.combine_models(parts[2], parts[3])
                if "error" in res:
                    out(f"✗ Error: {res['error']}")
                    return
                mid = res.get("model_id")
                ctx["current_model"] = mid
                out(f"✓ Combined model: {str(mid)[:8]}")
                return

            if sub == "save":
                if len(parts) < 3:
                    out("✗ Usage: model save <path> OR model save <id> <path>")
                    return
                if len(parts) == 3:
                    mid = selected_model()
                    if not mid:
                        out("✗ No model selected")
                        return
                    path = parts[2]
                else:
                    mid = parts[2]
                    path = parts[3]
                res = cmds.save_model(mid, path)
                if "error" in res:
                    out(f"✗ Error: {res['error']}")
                    return
                out(f"✓ Saved model {mid[:8]} to {path} ({res.get('format','?')})")
                return

            if sub == "load":
                if len(parts) < 3:
                    out("✗ Usage: model load <path>")
                    return
                res = cmds.load_model(parts[2])
                if "error" in res:
                    out(f"✗ Error: {res['error']}")
                    return
                mid = res.get("model_id")
                ctx["current_model"] = mid
                out(f"✓ Loaded model: {str(mid)[:8]}")
                return

            if sub == "remove":
                if len(parts) < 3:
                    out("✗ Usage: model remove <id>")
                    return
                mid = parts[2]
                res = cmds.delete_model(mid)
                if "error" in res:
                    out(f"✗ Error: {res['error']}")
                    return
                if ctx.get("current_model") == mid:
                    ctx["current_model"] = None
                out(f"✓ Removed model: {mid[:8]}")
                return

            if sub == "remove-all":
                ids = list(cmds.models.keys())
                if not ids:
                    out("No models to remove")
                    return
                for mid in ids:
                    cmds.delete_model(mid)
                ctx["current_model"] = None
                out(f"✓ Removed all {len(ids)} model(s)")
                return

            # Builders
            if sub == "mlp":
                if len(parts) < 3:
                    out("✗ Usage: model mlp <layers_csv> [act=relu|activations=a,b,c]")
                    return
                layers = _parse_csv_ints(parts[2])
                parsed = _parse_builder_args(parts[3:])
                res = cmds.create_mlp(layers, activation=parsed["activation"], activations=parsed["activations"])
                if "error" in res:
                    out(f"✗ Error: {res['error']}")
                    return
                mid = res.get("model_id")
                ctx["current_model"] = mid
                out(f"✓ Created MLP: {str(mid)[:8]}")
                return

            if sub == "linear":
                if len(parts) < 5:
                    out("✗ Usage: model linear <in> <hidden_csv> <out> [act=relu|activations=a,b,c]")
                    return
                input_size = int(parts[2])
                hidden = _parse_csv_ints(parts[3])
                output_size = int(parts[4])
                parsed = _parse_builder_args(parts[5:])
                res = cmds.create_linear_model(input_size, hidden, output_size, activation=parsed["activation"], activations=parsed["activations"])
                if "error" in res:
                    out(f"✗ Error: {res['error']}")
                    return
                mid = res.get("model_id")
                ctx["current_model"] = mid
                out(f"✓ Created Linear: {str(mid)[:8]}")
                return

            if sub == "cnn":
                if len(parts) < 4:
                    out("✗ Usage: model cnn <in_ch> <out_ch_csv> [kernels_csv] [act=relu|activations=a,b,c]")
                    return
                in_ch = int(parts[2])
                out_ch = _parse_csv_ints(parts[3])
                kernels: Optional[List[int]] = None
                rest = parts[4:]
                if rest and "," in rest[0] and rest[0].split(",")[0].strip().isdigit():
                    try:
                        kernels = _parse_csv_ints(rest[0])
                        rest = rest[1:]
                    except Exception:
                        kernels = None
                parsed = _parse_builder_args(rest)
                res = cmds.create_cnn(in_ch, out_ch, kernels, activation=parsed["activation"], activations=parsed["activations"])
                if "error" in res:
                    out(f"✗ Error: {res['error']}")
                    return
                mid = res.get("model_id")
                ctx["current_model"] = mid
                out(f"✓ Created CNN: {str(mid)[:8]}")
                return

            if sub == "gru":
                if len(parts) < 4:
                    out("✗ Usage: model gru <input_size> <hidden_size> [num_layers]")
                    return
                input_size = int(parts[2])
                hidden_size = int(parts[3])
                num_layers = int(parts[4]) if len(parts) > 4 else 1
                res = cmds.create_gru(input_size, hidden_size, num_layers)
                if "error" in res:
                    out(f"✗ Error: {res['error']}")
                    return
                mid = res.get("model_id")
                ctx["current_model"] = mid
                out(f"✓ Created GRU: {str(mid)[:8]}")
                return

            if sub == "transformer":
                if len(parts) < 6:
                    out("✗ Usage: model transformer <vocab> <d_model> <heads> <layers>")
                    return
                vocab = int(parts[2])
                d_model = int(parts[3])
                heads = int(parts[4])
                layers = int(parts[5])
                res = cmds.create_transformer(vocab, d_model, heads, layers)
                if "error" in res:
                    out(f"✗ Error: {res['error']}")
                    return
                mid = res.get("model_id")
                ctx["current_model"] = mid
                out(f"✓ Created Transformer: {str(mid)[:8]}")
                return

            out(f"✗ Unknown model command: {sub}")
            return

        out(f"✗ Command not recognized: {root}")
    except Exception as e:
        out(f"✗ Error: {e}")


def _render_models_page() -> None:
    st.header("Models")
    cmds = get_model_commands()
    ctx = st.session_state.repl_context

    # URL-based selection persistence.
    try:
        qp = st.query_params
        qmid = qp.get("model_id")
        if qmid and isinstance(qmid, str) and qmid in cmds.models:
            ctx["current_model"] = qmid
    except Exception:
        pass

    def _display_name(mid: str) -> str:
        md = cmds.models.get(mid) or {}
        cfg = md.get("config") or {}
        name = cfg.get("name")
        if name:
            return f"{name} ({mid[:8]})"
        return mid

    def _display_name_with_type(mid: str) -> str:
        md = cmds.models.get(mid) or {}
        t = md.get("type", "Unknown")
        return f"{_display_name(mid)} — {t}"

    models = cmds.list_models().get("models", [])
    if models:
        # Add display name column (stored in config['name'] when present).
        rows = []
        for m in models:
            mid8 = str(m.get("model_id"))
            # list_models returns truncated ids; find full id match.
            full_id = None
            for k in cmds.models.keys():
                if str(k).startswith(mid8):
                    full_id = k
                    break
            disp = _display_name(full_id) if full_id else mid8
            rows.append({"name": disp, "model_id": mid8, "type": m.get("type")})
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No models yet. Create one in Build & Train.")

    st.subheader("Select")
    # Use full IDs for selection (list_models returns truncated IDs).
    model_ids = list(cmds.models.keys())
    current = ctx.get("current_model")
    if current not in model_ids:
        current = model_ids[0] if model_ids else None
        ctx["current_model"] = current

    selected = st.selectbox(
        "Current model",
        options=[None] + model_ids,
        index=(model_ids.index(current) + 1) if (current in model_ids) else 0,
        format_func=lambda x: "(none)" if x is None else _display_name(x),
    )
    ctx["current_model"] = selected

    # Keep URL in sync.
    try:
        if selected is None:
            st.query_params.pop("model_id", None)
        else:
            st.query_params["model_id"] = str(selected)
    except Exception:
        pass

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("Refresh"):
            st.rerun()
    with col_b:
        if st.button("Deselect"):
            ctx["current_model"] = None
            st.rerun()
    with col_c:
        if st.button("Delete selected", disabled=(selected is None)):
            if selected is not None:
                res = cmds.delete_model(selected)
                if "error" in res:
                    st.error(res["error"])
                else:
                    ctx["current_model"] = None
                    st.success("Deleted")
                    st.rerun()

    if selected is not None and selected in cmds.models:
        st.subheader("Info")
        md = cmds.models[selected]
        cfg = dict(md.get("config", {}) or {})
        # Avoid showing both `activation` and per-layer `activations` in the UI.
        if "activations" in cfg and "activation" in cfg:
            cfg.pop("activation", None)
        st.code(
            json.dumps(
                {
                    "model_id": selected,
                    "type": md.get("type"),
                    "config": cfg,
                },
                indent=2,
                default=str,
            ),
            language="json",
        )

        diagram = _model_mermaid_diagram(cmds.models, selected)
        if diagram:
            st.subheader("Structure")
            st.markdown(f"```mermaid\n{diagram}\n```")

        fig3d = _model_plotly_3d_figure(cmds.models, selected)
        if fig3d is not None:
            st.subheader("Structure (3D)")
            st.plotly_chart(fig3d, use_container_width=True)
            st.markdown(
                """**Legend (3D)**

| Item | Meaning |
|---|---|
| Box | One layer/stage in the forward path |
| Stack direction |  Layer Construction order follows: Top -> Bottom (Bottom → Top: follows the forward order) |
| Box width/height | Relative magnitude (scaled to the largest layer in this diagram) |
| Depth (thickness) | Constant (visual spacing only) |
| Numbers used for sizing | MLP: `layer_sizes`; Linear: `input_size`, `hidden_sizes`, `output_size`; CNN: `in_channels`, `out_channels`; Combined: `input_size`/`output_size` (approx) |

**Combined (sequential)**: the stack shows **A → Combined → B** (A feeds into B). The size is based on the stored `input_size`/`output_size` metadata (approximate).

Note: this is an **approximate schematic**, not exact tensor shapes.
"""
            )
        else:
            st.caption("3D structure requires Plotly (install `plotly`).")

        st.subheader("Save")
        trained = ctx.get("trained_models")
        trained_set = set(trained) if isinstance(trained, list) else set()
        if selected not in trained_set:
            st.error("This model has not been trained in this session yet. Train it first, then save.")
        else:
            col_name, col_type = st.columns([4, 1])
            with col_type:
                save_type = st.selectbox("Type", options=[".pkl", ".pth", ".pt"], index=0, key="save_file_type")
            with col_name:
                save_name = st.text_input("File name", value=f"model_{selected}")

            save_name = (save_name or f"model_{selected}").strip()
            if Path(save_name).suffix.lower() != str(save_type).lower():
                save_name = str(Path(save_name).with_suffix(str(save_type)))

            dl_ext = Path(save_name).suffix.lower() if save_name else str(save_type)
            with tempfile.NamedTemporaryFile(delete=False, suffix=dl_ext) as f:
                tmp_out = f.name
            res = cmds.save_model(selected, tmp_out)
            if "error" in res:
                st.error(res["error"])
            else:
                data = Path(tmp_out).read_bytes()
                st.download_button(
                    "Download",
                    data=data,
                    file_name=save_name or f"model_{selected}{dl_ext}",
                    mime="application/octet-stream",
                )

    st.subheader("Load")
    up = st.file_uploader("Load a saved model (.pkl/.pth/.pt)", type=["pkl", "pth", "pt"], accept_multiple_files=False)
    if up is not None:
        upload_name = st.text_input("Name (optional)", value=Path(up.name).stem, key="upload_model_name")
        # Streamlit reruns often; avoid re-hashing large files unless the upload changed.
        upload_sig = f"{up.name}:{getattr(up, 'size', 'na')}"
        cache = ctx.setdefault("upload_cache", {})
        cached = cache.get(upload_sig)
        if isinstance(cached, dict) and cached.get("hash") and cached.get("raw"):
            raw = cached["raw"]
            file_hash = cached["hash"]
        else:
            raw = bytes(up.getbuffer())
            file_hash = hashlib.sha256(raw).hexdigest()
            cache[upload_sig] = {"hash": file_hash, "raw": raw}
        loaded_files = ctx.setdefault("loaded_files", {})
        already = loaded_files.get(file_hash)
        if already:
            st.info(f"Already loaded: {str(already)[:8]}")

        if not st.button("Load model", key="load_model_btn"):
            return
        suffix = Path(up.name).suffix or ".pkl"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(raw)
            tmp_path = f.name
        res = cmds.load_model(tmp_path)
        if "error" in res:
            st.error(res["error"])
            # If load failed, don't keep cached bytes around.
            cache.pop(upload_sig, None)
        else:
            ctx["current_model"] = res.get("model_id")
            mid = ctx["current_model"]
            # Persist a user-friendly name in config (UI-only metadata).
            if mid in cmds.models and upload_name:
                cfg0 = dict((cmds.models[mid].get("config") or {}))
                cfg0["name"] = str(upload_name)
                cmds.models[mid]["config"] = cfg0
            # Cache by content hash to prevent duplicates on re-upload.
            loaded_files[file_hash] = ctx["current_model"]
            ctx["last_loaded_hash"] = file_hash
            msg = f"Loaded as {str(ctx['current_model'])[:8]}"
            if res.get("unsafe_load"):
                msg += " (unsafe_load=True)"
            st.success(msg)
            st.rerun()



def _render_save_page() -> None:
    st.header("Save")
    cmds = get_model_commands()
    ctx = st.session_state.repl_context

    def _display_name(mid: str) -> str:
        md = cmds.models.get(mid) or {}
        cfg = md.get("config") or {}
        name = cfg.get("name")
        if name:
            return f"{name} ({mid[:8]})"
        return mid

    model_ids = list(cmds.models.keys())
    if not model_ids:
        st.info("No models available to save.")
        return

    # URL-based selection persistence.
    try:
        qmid = st.query_params.get("model_id")
        if qmid and isinstance(qmid, str) and qmid in model_ids:
            ctx["current_model"] = qmid
    except Exception:
        pass

    current = ctx.get("current_model")
    if current not in model_ids:
        current = model_ids[0]
        ctx["current_model"] = current

    selected = st.selectbox(
        "Model",
        options=model_ids,
        index=model_ids.index(current),
        format_func=_display_name,
    )
    ctx["current_model"] = selected
    try:
        st.query_params["model_id"] = str(selected)
    except Exception:
        pass

    col_name, col_type = st.columns([4, 1])
    with col_type:
        save_type = st.selectbox("Type", options=[".pkl", ".pth", ".pt"], index=0, key="save_file_type")
    with col_name:
        save_name = st.text_input("File name", value=f"model_{selected}")

    save_name = (save_name or f"model_{selected}").strip()
    if Path(save_name).suffix.lower() != str(save_type).lower():
        save_name = str(Path(save_name).with_suffix(str(save_type)))

    dl_ext = Path(save_name).suffix.lower() if save_name else str(save_type)
    with tempfile.NamedTemporaryFile(delete=False, suffix=dl_ext) as f:
        tmp_out = f.name
    res = cmds.save_model(selected, tmp_out)
    if "error" in res:
        st.error(res["error"])
        return
    data = Path(tmp_out).read_bytes()
    st.download_button(
        "Download",
        data=data,
        file_name=save_name or f"model_{selected}{dl_ext}",
        mime="application/octet-stream",
    )

def _render_build_train_page() -> None:
    st.header("Build & Train")
    cmds = get_model_commands()
    ctx = st.session_state.repl_context

    def _display_name(mid: str) -> str:
        md = cmds.models.get(mid) or {}
        cfg = md.get("config") or {}
        name = cfg.get("name")
        if name:
            return f"{name} ({mid[:8]})"
        return mid

    def _display_name_with_type(mid: str) -> str:
        md = cmds.models.get(mid) or {}
        t = md.get("type", "Unknown")
        return f"{_display_name(mid)} — {t}"

    st.subheader("Build")
    model_type = st.selectbox("Model type", options=["MLP", "Linear", "CNN", "GRU", "Transformer"], index=0)

    activation_mode = st.radio(
        "Activation mode",
        options=["Single (apply to all layers)", "Per-layer (CSV)"],
        index=0,
        horizontal=True,
        key="build_activation_mode",
    )

    activation = "relu"
    per_layer = ""
    if activation_mode.startswith("Per-layer"):
        per_layer = st.text_input(
            "Per-layer activations (CSV)",
            value=st.session_state.get("build_activation_per_layer", "relu, none"),
            key="build_activation_per_layer",
        )
    else:
        prior = str(st.session_state.get("build_activation_single", "relu"))
        activation = st.selectbox(
            "Activation",
            options=ACTIVATIONS,
            index=ACTIVATIONS.index(prior) if prior in ACTIVATIONS else 0,
            key="build_activation_single",
        )

    layers_csv: str = ""
    in_size: int = 0
    hidden_csv: str = ""
    out_size: int = 0
    in_ch: int = 0
    out_ch_csv: str = ""
    kernels_csv: str = ""
    gru_in: int = 0
    gru_hidden: int = 0
    gru_layers: int = 1
    vocab: int = 0
    d_model: int = 0
    heads: int = 0
    t_layers: int = 0

    with st.form("build_form"):
        if model_type == "MLP":
            layers_csv = st.text_input("Layer sizes (CSV)", value="10,32,32,2")
        elif model_type == "Linear":
            in_size = int(st.number_input("Input size", min_value=1, value=10, step=1))
            hidden_csv = st.text_input("Hidden sizes (CSV)", value="32,32")
            out_size = int(st.number_input("Output size", min_value=1, value=2, step=1))
        elif model_type == "CNN":
            in_ch = int(st.number_input("Input channels", min_value=1, value=3, step=1))
            out_ch_csv = st.text_input("Out channels (CSV)", value="32,64,128")
            kernels_csv = st.text_input("Kernel sizes (CSV, optional)", value="")
        elif model_type == "GRU":
            gru_in = int(st.number_input("Input size", min_value=1, value=128, step=1))
            gru_hidden = int(st.number_input("Hidden size", min_value=1, value=256, step=1))
            gru_layers = int(st.number_input("Num layers", min_value=1, value=1, step=1))
        else:
            vocab = int(st.number_input("Vocab size", min_value=2, value=1000, step=1))
            d_model = int(st.number_input("d_model", min_value=1, value=128, step=1))
            heads = int(st.number_input("Num heads", min_value=1, value=4, step=1))
            t_layers = int(st.number_input("Num layers", min_value=1, value=2, step=1))

        submitted = st.form_submit_button("Create")

    if submitted:
        activations = None
        if activation_mode.startswith("Per-layer"):
            activations = [x.strip() for x in per_layer.split(",") if x.strip()] or None
        if model_type == "MLP":
            try:
                layers = _parse_csv_ints(layers_csv)
                res = cmds.create_mlp(layers, activation=activation, activations=activations)
            except Exception as e:
                res = {"error": str(e)}
        elif model_type == "Linear":
            try:
                hidden = _parse_csv_ints(hidden_csv)
                res = cmds.create_linear_model(int(in_size), hidden, int(out_size), activation=activation, activations=activations)
            except Exception as e:
                res = {"error": str(e)}
        elif model_type == "CNN":
            try:
                out_ch = _parse_csv_ints(out_ch_csv)
                kernels = _parse_csv_ints(kernels_csv) if kernels_csv.strip() else None
                res = cmds.create_cnn(int(in_ch), out_ch, kernels, activation=activation, activations=activations)  # type: ignore[arg-type]
            except Exception as e:
                res = {"error": str(e)}
        elif model_type == "GRU":
            res = cmds.create_gru(int(gru_in), int(gru_hidden), int(gru_layers))
        else:
            res = cmds.create_transformer(int(vocab), int(d_model), int(heads), int(t_layers))

        if "error" in res:
            st.error(res["error"])
        else:
            ctx["current_model"] = res.get("model_id")
            st.success(f"Created {model_type}: {str(ctx['current_model'])[:8]}")
            st.rerun()

    st.divider()
    st.subheader("Combine")
    combine_ids = list(cmds.models.keys())
    if len(combine_ids) < 2:
        st.caption("Create at least two models to combine.")
    else:
        left = st.selectbox(
            "Model A",
            options=combine_ids,
            key="combine_a",
            format_func=_display_name_with_type,
        )
        right = st.selectbox(
            "Model B",
            options=combine_ids,
            key="combine_b",
            format_func=_display_name_with_type,
        )
        if st.button("Combine A → B"):
            res = cmds.combine_models(left, right)
            if "error" in res:
                st.error(res["error"])
            else:
                ctx["current_model"] = res.get("model_id")
                st.success(f"Combined model: {str(ctx['current_model'])[:8]}")
                st.rerun()

    st.divider()
    st.subheader("Train")
    model_ids = list(cmds.models.keys())
    if not model_ids:
        st.info("No models available to train.")
        return

    # URL-based selection persistence (Train section).
    try:
        qmid = st.query_params.get("model_id")
        if qmid and isinstance(qmid, str) and qmid in model_ids:
            ctx["current_model"] = qmid
    except Exception:
        pass

    current = ctx.get("current_model")
    if current not in model_ids:
        current = model_ids[0]
        ctx["current_model"] = current

    sel = st.selectbox(
        "Model",
        options=model_ids,
        index=model_ids.index(current),
        format_func=_display_name,
    )
    ctx["current_model"] = sel
    try:
        st.query_params["model_id"] = str(sel)
    except Exception:
        pass

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        epochs = st.number_input("Epochs", min_value=1, value=int(ctx.get("train_epochs", 5)), step=1)
    with col2:
        batch = st.number_input("Batch size", min_value=1, value=int(ctx.get("batch_size", 32)), step=1)
    with col3:
        lr = st.number_input("Learning rate", min_value=0.0, value=float(ctx.get("learning_rate", 0.001)), step=0.0005, format="%.6f")
    with col4:
        opt = st.selectbox("Optimizer", options=OPTIMIZERS, index=OPTIMIZERS.index(str(ctx.get("optimizer", "Adam"))))

    device = str(ctx.get("device", "cpu"))
    ctx["train_epochs"] = int(epochs)
    ctx["batch_size"] = int(batch)
    ctx["learning_rate"] = float(lr)
    ctx["optimizer"] = str(opt)

    # Keep a small history of training runs for plotting.
    ctx.setdefault("train_runs", [])

    if st.button("Train"):
        with st.spinner("Training..."):
            res = cmds.train_model(
                sel,
                epochs=int(epochs),
                learning_rate=float(lr),
                batch_size=int(batch),
                device=str(device),
                optimizer_type=str(opt),
            )
        if "error" in res:
            err = res.get("error")
            msg = str(err) if err is not None else ""
            if not msg.strip():
                msg = "Training failed (no error message returned)."
            st.error(msg)
            with st.expander("Details"):
                try:
                    import rasptorch as _rt
                    import rasptorch._cli_commands as _cc
                    st.caption(f"rasptorch: {getattr(_rt, '__file__', '?')}")
                    st.caption(f"_cli_commands: {getattr(_cc, '__file__', '?')}")
                except Exception:
                    pass

                # If this is a Combined model, show linkage diagnostics.
                try:
                    md_dbg = cmds.models.get(sel) or {}
                    if str(md_dbg.get("type")) == "Combined":
                        cfg_dbg = md_dbg.get("config") or {}
                        a_id = str(cfg_dbg.get("model_a_id", ""))
                        b_id = str(cfg_dbg.get("model_b_id", ""))
                        st.caption(f"Combined A id: {a_id} (exists={a_id in cmds.models})")
                        st.caption(f"Combined B id: {b_id} (exists={b_id in cmds.models})")
                        with st.expander("Combined config"):
                            st.json(cfg_dbg)
                except Exception:
                    pass

                if isinstance(res, dict) and (res.get("error_type") or res.get("traceback")):
                    if res.get("error_type"):
                        st.caption(f"error_type: {res.get('error_type')}")
                    if res.get("traceback"):
                        st.code(str(res.get("traceback")), language="text")
                st.json(res)
        else:
            hist = res.get("training_history")
            if isinstance(hist, list) and hist:
                ctx["train_runs"] = (ctx.get("train_runs", []) + [
                    {
                        "model_id": sel,
                        "device": str(device),
                        "optimizer": str(opt),
                        "learning_rate": float(lr),
                        "batch_size": int(batch),
                        "epochs": int(epochs),
                        "loss": [float(x) for x in hist],
                    }
                ])[-10:]
            st.success(f"Done. Final loss: {res.get('final_loss', 0.0):.6f}")
            try:
                trained = ctx.get("trained_models")
                if not isinstance(trained, list):
                    trained = []
                if sel not in trained:
                    trained.append(sel)
                ctx["trained_models"] = trained
            except Exception:
                pass

    # Plot most recent run, plus optionally overlay older runs.
    runs = ctx.get("train_runs", [])
    if runs:
        last = runs[-1]
        loss = last.get("loss") or []
        if isinstance(loss, list) and loss:
            st.caption("Train loss (most recent run)")
            st.line_chart(loss)


def _render_chat_page() -> None:
    st.header("Chat / REPL")
    st.caption("This is a web version of the rasptorch chat-style CLI. Type commands like `model list` or `train start`.")

    if st.button("Show help"):
        _append_log(HELP_TEXT)

    log = st.session_state.repl_log
    if log:
        st.code("\n".join(log[-400:]), language="text")
    else:
        st.info("No messages yet.")

    with st.form("repl_form", clear_on_submit=True):
        cmd = st.text_input("Command", value="", placeholder="e.g. model mlp 10,32,2 act=relu")
        submitted = st.form_submit_button("Run")
    if submitted:
        _append_log(f"> {cmd}")
        _handle_repl_command(cmd)
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="rasptorch UI", layout="wide")
    _init_state()

    st.title("rasptorch UI")
    st.caption(f"UI build: {_UI_BUILD}")

    if not _HAS_RASPTORCH:
        st.error("Failed to import rasptorch.")
        st.code(_RASPTORCH_IMPORT_ERROR, language="text")
        st.stop()

    ctx = st.session_state.repl_context

    with st.sidebar:
        st.subheader("Navigation")
        nav = st.radio(
            "Page",
            options=["Models", "Build & Train", "Chat/REPL"],
            index=["Models", "Build & Train", "Chat/REPL"].index(st.session_state.nav),
            label_visibility="collapsed",
        )
        st.session_state.nav = nav
        st.divider()
        st.subheader("Session")
        current_device = str(ctx.get("device", "cpu"))
        device_choice = st.selectbox(
            "Device",
            options=DEVICES,
            index=DEVICES.index(current_device) if current_device in DEVICES else 0,
            key="sidebar_device",
        )
        if str(device_choice) != current_device:
            ok, msg = _try_set_device(str(device_choice))
            if not ok:
                st.error(msg)
            else:
                st.success(msg)

        ok_vk, reason = _safe_vulkan_status()
        effective_device = str(ctx.get("device", "cpu"))
        if ok_vk:
            st.caption("Vulkan backend: initialized" + (" (in use)" if effective_device == "gpu" else " (available)"))
        else:
            st.caption("Vulkan backend: unavailable")
            if reason:
                st.caption(f"{reason}")

        mid = ctx.get("current_model")
        st.write(f"Selected model: {(mid[:8] if mid else '(none)')}")
        if st.button("Reset session state"):
            st.session_state.repl_log = []
            st.session_state.repl_context = {
                "current_model": None,
                "train_epochs": 5,
                "batch_size": 32,
                "learning_rate": 0.001,
                "optimizer": "Adam",
                "device": "cpu",
            }
            st.rerun()

    if nav == "Models":
        _render_models_page()
    elif nav == "Build & Train":
        _render_build_train_page()
    else:
        _render_chat_page()


if __name__ == "__main__":
    main()
