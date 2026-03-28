from __future__ import annotations as annotate
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import json
import sys
import tempfile
import hashlib
import platform
import os
import streamlit as st
import numpy as np

try:
    import streamlit.components.v1 as components  # type: ignore
except Exception:  # pragma: no cover
    components = None  # type: ignore[assignment]

try:
    import plotly.graph_objects as go  # type: ignore
except Exception:  # pragma: no cover
    go = None  # type: ignore[assignment]

try:
    import pyvista as pv  # type: ignore
except Exception:  # pragma: no cover
    pv = None  # type: ignore[assignment]

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
    # Pi/ARM WebGL drivers sometimes render Mesh3d very dark/black with defaults.
    # Use an explicit bright-ish color + high ambient lighting so the boxes remain visible.
    # Avoid alpha blending: on some Pi WebGL stacks, float framebuffers + blending
    # require EXT_float_blend (often missing). Fully-opaque traces are more portable.
    mesh_color = "rgb(145,150,170)"
    mesh_lighting = dict(ambient=0.95, diffuse=0.35, specular=0.05, roughness=1.0, fresnel=0.0)
    mesh_lightpos = dict(x=200, y=200, z=400)
    wire_color = "rgb(235,235,245)"

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

            # Wireframe overlay: helps on systems where Mesh3d faces don't render reliably.
            # Vert indices: 0..3 bottom face, 4..7 top face.
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
                (4, 5), (5, 6), (6, 7), (7, 4),  # top
                (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
            ]
            ex: List[float] = []
            ey: List[float] = []
            ez: List[float] = []
            for a, b in edges:
                ex.extend([xs[a], xs[b], None])
                ey.extend([ys[a], ys[b], None])
                ez.extend([zs[a], zs[b], None])
            fig.add_trace(
                go.Scatter3d(
                    x=ex,
                    y=ey,
                    z=ez,
                    mode="lines",
                    line=dict(width=6, color=wire_color),
                    hovertext=hov,
                    hoverinfo="text",
                    showlegend=False,
                )
            )

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
                    color=mesh_color,
                    opacity=1.0,
                    flatshading=True,
                    lighting=mesh_lighting,
                    lightposition=mesh_lightpos,
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
        conn_color = "rgb(200,200,200)"

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

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        ex: List[float] = []
        ey: List[float] = []
        ez: List[float] = []
        for a, b in edges:
            ex.extend([xs[a], xs[b], None])
            ey.extend([ys[a], ys[b], None])
            ez.extend([zs[a], zs[b], None])
        fig.add_trace(
            go.Scatter3d(
                x=ex,
                y=ey,
                z=ez,
                mode="lines",
                line=dict(width=6, color=wire_color),
                hovertext=hov,
                hoverinfo="text",
                showlegend=False,
            )
        )

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
                color=mesh_color,
                opacity=1.0,
                flatshading=True,
                lighting=mesh_lighting,
                lightposition=mesh_lightpos,
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


def _running_on_raspberry_pi() -> bool:
    """Best-effort detection for Raspberry Pi / ARM desktops.

    Plotly 3D traces use WebGL and can render poorly or not at all on some Pi
    browser + driver combinations. We keep 3D available elsewhere, and fall
    back to a simple 2D schematic on Pi/ARM.
    """
    # Escape hatch: allow users to force 3D (WebGL) even on Pi.
    # Usage: RASPTORCH_UI_FORCE_3D=1 streamlit run rasptorch/ui/app.py
    try:
        if str(os.environ.get("RASPTORCH_UI_FORCE_3D", "")).strip().lower() in {"1", "true", "yes", "on"}:
            return False
    except Exception:
        pass
    try:
        machine = platform.machine().lower()
        if machine not in {"aarch64", "arm64", "armv7l", "armv6l"}:
            return False
        # Prefer a concrete model check when present.
        model_path = Path("/proc/device-tree/model")
        if model_path.exists():
            txt = model_path.read_text(errors="ignore").lower()
            if "raspberry pi" in txt:
                return True
        # Otherwise: treat ARM as needing the safer fallback.
        return True
    except Exception:
        return False


def _is_raspberry_pi_arm() -> bool:
    """Detection for Pi/ARM that ignores UI escape hatches.

    We use this for choosing safe default renderers. Users can still override
    explicitly via `RASPTORCH_UI_3D_RENDER`.
    """
    try:
        machine = platform.machine().lower()
        if machine not in {"aarch64", "arm64", "armv7l", "armv6l"}:
            return False
        model_path = Path("/proc/device-tree/model")
        if model_path.exists():
            txt = model_path.read_text(errors="ignore").lower()
            if "raspberry pi" in txt:
                return True
        return True
    except Exception:
        return False


def _ui_3d_render_mode() -> str:
    """Select the renderer for the 3D structure panel.

    Modes:
      - plotly: interactive WebGL (default on non-Pi)
      - pyvista: interactive (requires a working `stpyvista` + browser WebGL)
      - pyvista_png: server-side offscreen render to PNG (WebGL-free; reliable)

    Set via: `RASPTORCH_UI_3D_RENDER=plotly|pyvista|pyvista_png`.
    """
    raw = str(os.environ.get("RASPTORCH_UI_3D_RENDER", "")).strip().lower()
    if raw in {"plotly", "pyvista", "pyvista_png", "png", "image", "static"}:
        if raw in {"png", "image", "static"}:
            return "pyvista_png" if pv is not None else "plotly"
        return raw

    # Default: on Raspberry Pi / ARM, prefer the WebGL-free PNG renderer when PyVista is present.
    if _is_raspberry_pi_arm() and pv is not None:
        return "pyvista_png"
    return "plotly"


def _model_pyvista_plotter(models: Dict[str, Any], model_id: Optional[str], *, off_screen: bool) -> Any:
    if pv is None or not model_id or model_id not in models:
        return None

    md = models.get(model_id) or {}
    mtype = str(md.get("type", "Unknown"))
    cfg = md.get("config") or {}

    def _act_summary() -> str:
        acts = cfg.get("activations")
        act = cfg.get("activation")
        if isinstance(acts, list) and acts:
            return "activations=" + ",".join(str(a) for a in acts)
        if act is not None:
            return f"activation={act}"
        return ""

    def _coerce_size(v: Any) -> float:
        try:
            fv = float(v)
        except Exception:
            fv = 1.0
        return max(1.0, fv)

    plotter = pv.Plotter(off_screen=off_screen, window_size=(980, 560))
    plotter.set_background((35 / 255.0, 38 / 255.0, 45 / 255.0))

    # Consistent “wire” tone.
    edge_color = (235 / 255.0, 235 / 255.0, 245 / 255.0)

    points: List[List[float]] = []
    texts: List[str] = []

    def _add_cube(center_x: float, z0: float, d: float, label: str, scalar: float) -> Tuple[float, float]:
        w = float(d)
        h = float(d)
        t = 0.8
        z1 = z0 + t
        zc = (z0 + z1) / 2
        cube = pv.Cube(center=(center_x, 0.0, zc), x_length=w, y_length=h, z_length=t)
        cube.cell_data["layer"] = np.full(cube.n_cells, scalar, dtype=np.float32)
        plotter.add_mesh(
            cube,
            scalars="layer",
            cmap="viridis",
            show_scalar_bar=False,
            smooth_shading=False,
            opacity=1.0,
            show_edges=True,
            edge_color=edge_color,
            line_width=2,
        )
        points.append([center_x, 0.0, zc])
        texts.append(label)
        return z1, zc

    def _add_line(a: List[float], b: List[float]) -> None:
        try:
            plotter.add_lines(np.array([a, b], dtype=np.float32), color=(200 / 255.0, 200 / 255.0, 200 / 255.0), width=4)
        except Exception:
            pass

    # Combined(sequential): two stacks.
    if mtype == "Combined" and str(cfg.get("combine")) == "sequential":
        a_t = str(cfg.get("model_a_type", "A"))
        b_t = str(cfg.get("model_b_type", "B"))
        io_in = cfg.get("input_size")
        io_out = cfg.get("output_size")

        left_sizes = [_coerce_size(io_in or 8), _coerce_size(io_out or (io_in or 8))]
        right_sizes = [_coerce_size(io_out or 8)]
        max_sz = max(left_sizes + right_sizes) if (left_sizes or right_sizes) else 1.0
        left_dims = [0.6 + 2.4 * (s / max_sz) for s in left_sizes]
        right_dims = [0.6 + 2.4 * (s / max_sz) for s in right_sizes]

        stack_dx = 3.6
        box_gap = 0.45

        z_left = 0.0
        left_centers: List[List[float]] = []
        for idx, (d, lab) in enumerate(zip(left_dims, [f"A {a_t}", "Combined"])):
            z1, zc = _add_cube(-stack_dx, z_left, d, lab, scalar=float(idx))
            left_centers.append([-stack_dx, 0.0, zc])
            z_left = z1 + box_gap

        z_right = 0.0
        right_centers: List[List[float]] = []
        for idx, (d, lab) in enumerate(zip(right_dims, [f"B {b_t}"])):
            z1, zc = _add_cube(stack_dx, z_right, d, lab, scalar=float(10 + idx))
            right_centers.append([stack_dx, 0.0, zc])
            z_right = z1 + box_gap

        if len(left_centers) >= 2:
            _add_line([left_centers[0][0] + 0.6, 0.0, left_centers[0][2]], [left_centers[1][0] + 0.6, 0.0, left_centers[1][2]])
        if left_centers and right_centers:
            _add_line([left_centers[-1][0] + 0.6, 0.0, left_centers[-1][2]], [right_centers[0][0] - 0.6, 0.0, right_centers[0][2]])

    else:
        sizes: List[float] = []
        labels: List[str] = []

        def _add(sz: Any, label: str) -> None:
            sizes.append(_coerce_size(sz))
            labels.append(label)

        act_s = _act_summary()

        if mtype == "MLP":
            for i, s in enumerate(cfg.get("layer_sizes") or []):
                _add(s, f"L{i}: {s}")
        elif mtype == "Linear":
            inp = cfg.get("input_size")
            _add(inp or 1, f"in: {inp}")
            for i, h in enumerate(cfg.get("hidden_sizes") or []):
                _add(h, f"h{i}: {h}")
            out = cfg.get("output_size")
            _add(out or 1, f"out: {out}")
        elif mtype == "CNN":
            kernels = cfg.get("kernels")
            in_ch = cfg.get("in_channels")
            _add(in_ch or 1, f"in_ch: {in_ch}")
            for i, ch in enumerate(cfg.get("out_channels") or []):
                k = None
                if isinstance(kernels, list) and i < len(kernels):
                    k = kernels[i]
                lab = f"c{i}: {ch}" + (f" (k={k})" if k is not None else "")
                _add(ch, lab)
        else:
            io_in = cfg.get("input_size")
            _add(io_in or 8, mtype)

        if sizes:
            max_sz = max(sizes)
            dims = [0.6 + 2.4 * (s / max_sz) for s in sizes]
            z0 = 0.0
            for idx, (d, lab) in enumerate(zip(dims, labels)):
                z1, _ = _add_cube(0.0, z0, d, lab, scalar=float(idx))
                z0 = z1 + 0.3

    if points and texts:
        try:
            plotter.add_point_labels(
                np.array(points, dtype=np.float32),
                texts,
                font_size=16,
                point_size=0,
                text_color="white",
                shape=None,
                fill_shape=False,
            )
        except Exception:
            pass

    if cfg.get("activation") is not None or cfg.get("activations") is not None:
        # If we can render text labels, include activation info in the title.
        title = f"{mtype} ({_act_summary()})" if _act_summary() else mtype
    else:
        title = mtype
    try:
        plotter.add_text(title, position="upper_left", color="white", font_size=12)
    except Exception:
        pass

    try:
        plotter.view_isometric()
        plotter.camera.zoom(1.35)
    except Exception:
        pass

    return plotter


def _model_pyvista_png(models: Dict[str, Any], model_id: Optional[str]) -> Optional[bytes]:
    """Render the model schematic via PyVista offscreen and return PNG bytes.

    This avoids browser WebGL entirely (useful on Raspberry Pi where WebGL Mesh3d
    and vtk.js can be unreliable).
    """
    if pv is None:
        return None
    plotter = _model_pyvista_plotter(models, model_id, off_screen=True)
    if plotter is None:
        return None
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name
        plotter.show(screenshot=tmp_path, auto_close=True)
        data = Path(tmp_path).read_bytes()
        return data
    except Exception:
        try:
            plotter.close()
        except Exception:
            pass
        return None
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass


def _model_plotly_2d_figure(models: Dict[str, Any], model_id: Optional[str]):
    """2D version of the model schematic.

    Uses Plotly 2D primitives (SVG/canvas) rather than WebGL.
    """
    if go is None or not model_id or model_id not in models:
        return None

    plot_bg = "rgba(35,38,45,1.0)"
    box_fill = "rgba(120,120,140,0.65)"
    box_line = "rgba(230,230,240,0.55)"
    box_shadow = "rgba(230,230,240,0.16)"
    conn_color = "rgba(200,200,200,0.75)"

    md = models.get(model_id) or {}
    mtype = str(md.get("type", "Unknown"))
    cfg = md.get("config") or {}

    def _coerce_size(v: Any) -> float:
        try:
            fv = float(v)
        except Exception:
            fv = 1.0
        return max(1.0, fv)

    def _act_summary() -> str:
        acts = cfg.get("activations")
        act = cfg.get("activation")
        if isinstance(acts, list) and acts:
            return "activations=" + ",".join(str(a) for a in acts)
        if act is not None:
            return f"activation={act}"
        return ""

    def _model_summary() -> str:
        if mtype == "Combined" and str(cfg.get("combine")) == "sequential":
            a_t = str(cfg.get("model_a_type", "A"))
            b_t = str(cfg.get("model_b_type", "B"))
            io_in = cfg.get("input_size")
            io_out = cfg.get("output_size")
            return f"type=Combined(sequential)\nA={a_t}\nB={b_t}\ninput_size={io_in}\noutput_size={io_out}"
        if mtype in {"Linear", "MLP", "CNN"}:
            bits: List[str] = [f"type={mtype}"]
            if cfg.get("input_size") is not None:
                bits.append(f"input_size={cfg.get('input_size')}")
            if cfg.get("output_size") is not None:
                bits.append(f"output_size={cfg.get('output_size')}")
            if mtype == "CNN" and cfg.get("in_channels") is not None:
                bits.append(f"in_channels={cfg.get('in_channels')}")
            act_s0 = _act_summary()
            if act_s0:
                bits.append(act_s0)
            return "\n".join(bits)
        return f"type={mtype}"

    def _stack_specs_for_model(model_type: str, model_cfg: Dict[str, Any]) -> Tuple[List[float], List[str], List[str]]:
        sizes: List[float] = []
        labels: List[str] = []
        hovers: List[str] = []

        def _add(sz: Any, label: str, hover: Optional[str] = None) -> None:
            sizes.append(_coerce_size(sz))
            labels.append(label)
            hovers.append(hover or label)

        act_s = ""
        if model_cfg is cfg:
            act_s = _act_summary()

        if model_type == "MLP":
            for i, s in enumerate(model_cfg.get("layer_sizes") or []):
                hover = f"type=MLP<br>layer={i}<br>size={s}"
                if act_s:
                    hover += f"<br>{act_s}"
                _add(s, f"L{i}: {s}", hover=hover)
        elif model_type == "Linear":
            inp = model_cfg.get("input_size")
            hover = f"type=Linear<br>input_size={inp}"
            if act_s:
                hover += f"<br>{act_s}"
            _add(inp or 1, f"in: {inp}", hover=hover)
            for i, h in enumerate(model_cfg.get("hidden_sizes") or []):
                hover = f"type=Linear<br>hidden[{i}]={h}"
                if act_s:
                    hover += f"<br>{act_s}"
                _add(h, f"h{i}: {h}", hover=hover)
            out = model_cfg.get("output_size")
            hover = f"type=Linear<br>output_size={out}"
            if act_s:
                hover += f"<br>{act_s}"
            _add(out or 1, f"out: {out}", hover=hover)
        elif model_type == "CNN":
            kernels = model_cfg.get("kernels")
            in_ch = model_cfg.get("in_channels")
            hover = f"type=CNN<br>in_channels={in_ch}"
            if act_s:
                hover += f"<br>{act_s}"
            _add(in_ch or 1, f"in_ch: {in_ch}", hover=hover)
            for i, ch in enumerate(model_cfg.get("out_channels") or []):
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
            io_in = model_cfg.get("input_size")
            _add(io_in or 8, model_type, hover=f"type={model_type}<br>input_size={io_in}")

        return sizes, labels, hovers

    fig = go.Figure()

    def _draw_stack(center_x: float, stack_sizes: List[float], stack_labels: List[str], stack_hovers: List[str]) -> Tuple[List[float], List[float]]:
        if not stack_sizes:
            return [], []

        max_sz = max(stack_sizes) if stack_sizes else 1.0
        dims = [0.6 + 2.4 * (s / max_sz) for s in stack_sizes]

        y_top = 0.0
        gap = 0.45
        centers_y: List[float] = []
        centers_x: List[float] = []
        text_x: List[float] = []
        text_y: List[float] = []
        text_t: List[str] = []

        for d, lab, hov in zip(dims, stack_labels, stack_hovers):
            w = d
            h = d
            x0, x1 = center_x - w / 2, center_x + w / 2
            y0, y1 = y_top, y_top + h

            # Use a filled polygon trace (not a layout shape) so hover works over the box area.
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1, x1, x0, x0],
                    y=[y0, y0, y1, y1, y0],
                    mode="lines",
                    fill="toself",
                    fillcolor=box_fill,
                    line=dict(color=box_line, width=1),
                    hovertext=hov,
                    hoverinfo="text",
                    showlegend=False,
                )
            )

            cx = center_x
            cy = (y0 + y1) / 2
            centers_x.append(cx)
            centers_y.append(cy)
            text_x.append(cx)
            text_y.append(cy)
            text_t.append(lab)

            y_top = y1 + gap

        # Labels.
        fig.add_trace(
            go.Scatter(
                x=text_x,
                y=text_y,
                mode="text",
                text=text_t,
                textposition="middle center",
                textfont=dict(size=16, color="white"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        return centers_x, centers_y

    # Special-case Combined(sequential): two stacks + flow indicator.
    if mtype == "Combined" and str(cfg.get("combine")) == "sequential":
        a_id = str(cfg.get("model_a_id", ""))
        b_id = str(cfg.get("model_b_id", ""))
        a_t = str(cfg.get("model_a_type", "A"))
        b_t = str(cfg.get("model_b_type", "B"))
        io_in = cfg.get("input_size")
        io_out = cfg.get("output_size")

        left_sizes = [_coerce_size(io_in or 8), _coerce_size(io_out or (io_in or 8))]
        left_labels = [f"A {a_t}", "Combined"]
        left_hovers = [
            "type=Combined(sequential)"
            f"<br>direction=A → B"
            f"<br>A={a_t} ({a_id[:8] if a_id else '?'})"
            f"<br>input_size={io_in}",
            "type=Combined(sequential)"
            f"<br>direction=A → B"
            f"<br>output_size={io_out}",
        ]

        right_sizes = [_coerce_size(io_out or 8)]
        right_labels = [f"B {b_t}"]
        right_hovers = [
            "type=Combined(sequential)"
            f"<br>direction=A → B"
            f"<br>B={b_t} ({b_id[:8] if b_id else '?'})"
            f"<br>output_size={io_out}",
        ]

        stack_dx = 3.6
        left_cx, left_cy = _draw_stack(-stack_dx, left_sizes, left_labels, left_hovers)
        right_cx, right_cy = _draw_stack(stack_dx, right_sizes, right_labels, right_hovers)

        # Connectors.
        if len(left_cy) >= 2:
            fig.add_trace(
                go.Scatter(
                    x=[-stack_dx + 0.65, -stack_dx + 0.65],
                    y=[left_cy[0], left_cy[1]],
                    mode="lines+text",
                    line=dict(width=4, color=conn_color),
                    text=["", "A → …"],
                    textposition="top center",
                    textfont=dict(size=13, color=conn_color),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        left_flow_y = left_cy[1] if len(left_cy) >= 2 else (left_cy[0] if left_cy else 0.6)
        right_flow_y = right_cy[0] if right_cy else 0.6
        fig.add_trace(
            go.Scatter(
                x=[-stack_dx + 0.45, stack_dx - 0.45],
                y=[left_flow_y, right_flow_y],
                mode="lines+text",
                line=dict(width=4, color=conn_color),
                text=["", "A → B"],
                textposition="top center",
                textfont=dict(size=13, color=conn_color),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        # Layout bounds.
        max_y = 0.0
        if left_cy:
            max_y = max(max_y, max(left_cy) + 2.2)
        if right_cy:
            max_y = max(max_y, max(right_cy) + 2.2)
        fig.update_layout(
            height=520,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor=plot_bg,
            plot_bgcolor=plot_bg,
            hovermode="closest",
            dragmode="pan",
            xaxis=dict(visible=False, range=[-7.0, 7.0]),
            yaxis=dict(visible=False, range=[max_y, -1.0]),
            showlegend=False,
        )
        return fig

    sizes, labels, hovers = _stack_specs_for_model(mtype, cfg)
    if not sizes:
        return None

    cx, cy = _draw_stack(0.0, sizes, labels, hovers)
    max_y = (max(cy) + 2.2) if cy else 6.0
    fig.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor=plot_bg,
        plot_bgcolor=plot_bg,
        hovermode="closest",
        dragmode="pan",
        xaxis=dict(visible=False, range=[-4.2, 4.2]),
        yaxis=dict(visible=False, range=[max_y, -1.0]),
        showlegend=False,
    )
    return fig


def _model_structure_svg_html(models: Dict[str, Any], model_id: Optional[str]) -> Optional[str]:
    """Interactive SVG 2D schematic (pan/zoom + hover tooltips).

    Uses no WebGL, so it works on Raspberry Pi browsers where 3D renderers
    (Plotly Mesh3d / vtk.js) can fail due to missing WebGL extensions.
    """
    if not model_id or model_id not in models:
        return None

    md = models.get(model_id) or {}
    mtype = str(md.get("type", "Unknown"))
    cfg = md.get("config") or {}

    plot_bg = "rgba(35,38,45,1.0)"
    box_fill = "rgba(120,120,140,0.65)"
    box_line = "rgba(230,230,240,0.55)"
    conn_color = "rgba(200,200,200,0.75)"

    def _esc(s: Any) -> str:
        t = str(s)
        return (
            t.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    def _coerce_size(v: Any) -> float:
        try:
            fv = float(v)
        except Exception:
            fv = 1.0
        return max(1.0, fv)

    def _act_summary() -> str:
        acts = cfg.get("activations")
        act = cfg.get("activation")
        if isinstance(acts, list) and acts:
            return "activations=" + ",".join(str(a) for a in acts)
        if act is not None:
            return f"activation={act}"
        return ""

    # Build stack specs.
    sizes: List[float] = []
    labels: List[str] = []
    hovers: List[str] = []

    def _add(sz: Any, label: str, hover: str) -> None:
        sizes.append(_coerce_size(sz))
        labels.append(label)
        hovers.append(hover)

    act_s = _act_summary()

    # Combined(sequential): render as two stacks.
    combined_seq = (mtype == "Combined" and str(cfg.get("combine")) == "sequential")
    if combined_seq:
        a_id = str(cfg.get("model_a_id", ""))
        b_id = str(cfg.get("model_b_id", ""))
        a_t = str(cfg.get("model_a_type", "A"))
        b_t = str(cfg.get("model_b_type", "B"))
        io_in = cfg.get("input_size")
        io_out = cfg.get("output_size")

        left_sizes = [_coerce_size(io_in or 8), _coerce_size(io_out or (io_in or 8))]
        left_labels = [f"A {a_t}", "Combined"]
        left_hovers = [
            "type=Combined(sequential)" + f"\nA={a_t} ({a_id[:8] if a_id else '?'})" + f"\ninput_size={io_in}",
            "type=Combined(sequential)" + f"\noutput_size={io_out}",
        ]
        right_sizes = [_coerce_size(io_out or 8)]
        right_labels = [f"B {b_t}"]
        right_hovers = [
            "type=Combined(sequential)" + f"\nB={b_t} ({b_id[:8] if b_id else '?'})" + f"\noutput_size={io_out}",
        ]
    else:
        if mtype == "MLP":
            for i, s in enumerate(cfg.get("layer_sizes") or []):
                hover = f"type=MLP\nlayer={i}\nsize={s}" + (f"\n{act_s}" if act_s else "")
                _add(s, f"L{i}: {s}", hover)
        elif mtype == "Linear":
            inp = cfg.get("input_size")
            hover = f"type=Linear\ninput_size={inp}" + (f"\n{act_s}" if act_s else "")
            _add(inp or 1, f"in: {inp}", hover)
            for i, h in enumerate(cfg.get("hidden_sizes") or []):
                hover = f"type=Linear\nhidden[{i}]={h}" + (f"\n{act_s}" if act_s else "")
                _add(h, f"h{i}: {h}", hover)
            out = cfg.get("output_size")
            hover = f"type=Linear\noutput_size={out}" + (f"\n{act_s}" if act_s else "")
            _add(out or 1, f"out: {out}", hover)
        elif mtype == "CNN":
            kernels = cfg.get("kernels")
            in_ch = cfg.get("in_channels")
            hover = f"type=CNN\nin_channels={in_ch}" + (f"\n{act_s}" if act_s else "")
            _add(in_ch or 1, f"in_ch: {in_ch}", hover)
            for i, ch in enumerate(cfg.get("out_channels") or []):
                k = None
                if isinstance(kernels, list) and i < len(kernels):
                    k = kernels[i]
                lab = f"c{i}: {ch}" + (f" (k={k})" if k is not None else "")
                hover = f"type=CNN\nout_channels[{i}]={ch}" + (f"\nkernel={k}" if k is not None else "")
                if act_s:
                    hover += f"\n{act_s}"
                _add(ch, lab, hover)
        else:
            io_in = cfg.get("input_size")
            _add(io_in or 8, mtype, f"type={mtype}\ninput_size={io_in}")

        left_sizes = sizes
        left_labels = labels
        left_hovers = hovers
        right_sizes = []
        right_labels = []
        right_hovers = []

    if not left_sizes:
        return None

    # Geometry in "units" (similar to the Plotly 2D schematic), then scale to pixels.
    unit = 120.0
    gap = 0.45
    box_gap_px = gap * unit

    max_sz = max(left_sizes + right_sizes) if right_sizes else max(left_sizes)
    left_dims = [0.6 + 2.4 * (s / max_sz) for s in left_sizes]
    right_dims = [0.6 + 2.4 * (s / max_sz) for s in right_sizes]

    # Layout: one or two stacks.
    stack_dx_units = 3.6
    stack_dx_px = stack_dx_units * unit

    shapes: List[str] = []
    lines: List[str] = []
    texts: List[str] = []

    depth_dx = 10.0
    depth_dy = 10.0

    def _draw_stack(center_x_px: float, dims: List[float], labs: List[str], hvs: List[str]) -> List[Tuple[float, float]]:
        y_top = 0.0
        centers: List[Tuple[float, float]] = []
        for d, lab, hv in zip(dims, labs, hvs):
            w = d * unit
            h = d * unit
            x0 = center_x_px - w / 2
            y0 = y_top
            x1 = center_x_px + w / 2
            y1 = y_top + h
            cx = center_x_px
            cy = (y0 + y1) / 2
            centers.append((cx, cy))
            shapes.append(
                "\n".join(
                    [
                        f'<g class="rt-box" tabindex="0">',
                        f'  <title>{_esc(hv)}</title>',
                        f'  <rect class="shadow" x="{x0 + depth_dx:.1f}" y="{y0 + depth_dy:.1f}" width="{w:.1f}" height="{h:.1f}" rx="8" ry="8"/>',
                        f'  <rect class="box" x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h:.1f}" rx="8" ry="8"/>',
                        f'  <text class="label" x="{cx:.1f}" y="{cy:.1f}">{_esc(lab)}</text>',
                        f'</g>',
                    ]
                )
            )
            y_top = y1 + box_gap_px
        return centers

    left_centers = _draw_stack((-stack_dx_px if right_dims else 0.0), left_dims, left_labels, left_hovers)
    right_centers = _draw_stack((stack_dx_px if right_dims else 0.0), right_dims, right_labels, right_hovers)

    # Connectors.
    if combined_seq:
        if len(left_centers) >= 2:
            a0 = (left_centers[0][0] + 0.65 * unit, left_centers[0][1])
            a1 = (left_centers[1][0] + 0.65 * unit, left_centers[1][1])
            lines.append(f'<line class="conn" x1="{a0[0]:.1f}" y1="{a0[1]:.1f}" x2="{a1[0]:.1f}" y2="{a1[1]:.1f}"/>')
        if left_centers and right_centers:
            b0 = (left_centers[-1][0] + 0.45 * unit, left_centers[-1][1])
            b1 = (right_centers[0][0] - 0.45 * unit, right_centers[0][1])
            lines.append(f'<line class="conn" x1="{b0[0]:.1f}" y1="{b0[1]:.1f}" x2="{b1[0]:.1f}" y2="{b1[1]:.1f}"/>')
            # Arrow label.
            texts.append(f'<text class="flow" x="{(b0[0]+b1[0])/2:.1f}" y="{(b0[1]+b1[1])/2 - 14:.1f}">A → B</text>')

    content_h = 0.0
    for g in [left_centers, right_centers]:
        if g:
            content_h = max(content_h, max(y for _, y in g) + 2.2 * unit)
    content_h = max(content_h, 520.0)

    content_w = 8.4 * unit if right_dims else 4.2 * unit
    pad = 48.0
    vb_x = -content_w / 2 - pad
    vb_y = -pad
    vb_w = content_w + 2 * pad
    vb_h = content_h + 2 * pad

    svg_id = f"rt-structure-svg-{_esc(model_id)}"

    initial_info_js = json.dumps(_model_summary())

    lines_html = "\n      ".join(lines)
    texts_html = "\n      ".join(texts)
    shapes_html = "\n      ".join(shapes)

    html = f"""
<div style=\"background:{plot_bg}; border-radius: 0.5rem; padding: 0.25rem;\">
  <svg id=\"{svg_id}\" viewBox=\"{vb_x:.1f} {vb_y:.1f} {vb_w:.1f} {vb_h:.1f}\" width=\"100%\" height=\"520\" style=\"touch-action:none;\">
    <style>
            .shadow {{ fill: {box_shadow}; stroke: none; }}
      .box {{ fill: {box_fill}; stroke: {box_line}; stroke-width: 1; }}
      .rt-box {{ cursor: pointer; }}
      .rt-box:hover .box, .rt-box:focus .box {{ stroke-width: 3; }}
            .rt-box.selected .box {{ stroke-width: 3; }}
      .label {{ fill: white; font-size: 18px; font-family: sans-serif; dominant-baseline: middle; text-anchor: middle; user-select: none; pointer-events: none; }}
      .conn {{ stroke: {conn_color}; stroke-width: 6; stroke-linecap: round; }}
      .flow {{ fill: {conn_color}; font-size: 14px; font-family: sans-serif; text-anchor: middle; user-select: none; pointer-events: none; }}
            .hud-bg {{ fill: {plot_bg}; opacity: 0.92; }}
            .hud-border {{ fill: none; stroke: {box_line}; stroke-width: 1; opacity: 0.6; }}
            .hud-title {{ fill: white; font-size: 14px; font-family: sans-serif; opacity: 0.95; }}
            .hud-text {{ fill: white; font-size: 12px; font-family: monospace; opacity: 0.9; }}
    </style>
    <rect x=\"{vb_x:.1f}\" y=\"{vb_y:.1f}\" width=\"{vb_w:.1f}\" height=\"{vb_h:.1f}\" fill=\"{plot_bg}\" />
    <g id=\"viewport\">
            {lines_html}
            {texts_html}
            {shapes_html}
    </g>
        <g id=\"hud\">
            <rect class=\"hud-bg\" x=\"{vb_x + 8:.1f}\" y=\"{vb_y + 8:.1f}\" width=\"{vb_w - 16:.1f}\" height=\"86\" rx=\"10\" ry=\"10\" />
            <rect class=\"hud-border\" x=\"{vb_x + 8:.1f}\" y=\"{vb_y + 8:.1f}\" width=\"{vb_w - 16:.1f}\" height=\"86\" rx=\"10\" ry=\"10\" />
            <text class=\"hud-title\" x=\"{vb_x + 20:.1f}\" y=\"{vb_y + 30:.1f}\">Model / Layer Info (hover or click a box)</text>
            <text id=\"infoText\" class=\"hud-text\" x=\"{vb_x + 20:.1f}\" y=\"{vb_y + 50:.1f}\"></text>
        </g>
    <script>
      (function() {{
        const svg = document.getElementById('{svg_id}');
        if (!svg) return;
        const viewport = svg.querySelector('#viewport');
        if (!viewport) return;
                const infoText = svg.querySelector('#infoText');

                function setInfo(raw) {{
                    if (!infoText) return;
                    while (infoText.firstChild) infoText.removeChild(infoText.firstChild);
                    const lines = String(raw || '').split('\n');
                    lines.forEach((ln, i) => {{
                        const tspan = document.createElementNS('http://www.w3.org/2000/svg', 'tspan');
                        tspan.setAttribute('x', infoText.getAttribute('x'));
                        tspan.setAttribute('dy', i === 0 ? '0' : '1.25em');
                        tspan.textContent = ln;
                        infoText.appendChild(tspan);
                    }});
                }}

                setInfo({initial_info_js});

                let selectedBox = null;
                function selectBox(g) {{
                    if (selectedBox && selectedBox !== g) selectedBox.classList.remove('selected');
                    selectedBox = g;
                    if (selectedBox) selectedBox.classList.add('selected');
                }}

                const boxes = svg.querySelectorAll('.rt-box');
                boxes.forEach((g) => {{
                    const titleEl = g.querySelector('title');
                    const getText = () => titleEl ? titleEl.textContent : '';
                    g.addEventListener('pointerenter', () => setInfo(getText()));
                    g.addEventListener('focus', () => setInfo(getText()));
                    g.addEventListener('click', (e) => {{
                        e.stopPropagation();
                        selectBox(g);
                        setInfo(getText());
                    }});
                }});

        let scale = 1.0;
        let tx = 0.0;
        let ty = 0.0;
        let panning = false;
        let lastX = 0.0;
        let lastY = 0.0;

        function apply() {{
          viewport.setAttribute('transform', `translate(${{tx}} ${{ty}}) scale(${{scale}})`);
        }}

                svg.addEventListener('pointerdown', (e) => {{
                    if (e.target && e.target.closest && (e.target.closest('.rt-box') || e.target.closest('#hud'))) return;
          panning = true;
          lastX = e.clientX;
          lastY = e.clientY;
          svg.setPointerCapture(e.pointerId);
        }});

        svg.addEventListener('pointerup', (e) => {{
          panning = false;
          try {{ svg.releasePointerCapture(e.pointerId); }} catch (_) {{}}
        }});

        svg.addEventListener('pointermove', (e) => {{
          if (!panning) return;
          const dx = e.clientX - lastX;
          const dy = e.clientY - lastY;
          lastX = e.clientX;
          lastY = e.clientY;
          tx += dx;
          ty += dy;
          apply();
        }});

        svg.addEventListener('wheel', (e) => {{
          e.preventDefault();
          const rect = svg.getBoundingClientRect();
          const mx = e.clientX - rect.left;
          const my = e.clientY - rect.top;
          const delta = Math.max(-1, Math.min(1, e.deltaY));
          const zoom = delta < 0 ? 1.12 : 1/1.12;
          const newScale = Math.min(4.0, Math.max(0.35, scale * zoom));
          const k = newScale / scale;
          // zoom about mouse position in screen coords
          tx = mx - k * (mx - tx);
          ty = my - k * (my - ty);
          scale = newScale;
          apply();
        }}, {{ passive: false }});

        apply();
      }})();
    </script>
  </svg>
</div>
"""
    return html


def _model_plotly_structure_figure(models: Dict[str, Any], model_id: Optional[str]) -> Tuple[Any, str] | Tuple[None, str]:
    """Return (figure, mode) where mode is '3D' or '2D'."""
    if go is None:
        return None, ""
    if _running_on_raspberry_pi():
        return _model_plotly_2d_figure(models, model_id), "2D"
    return _model_plotly_3d_figure(models, model_id), "3D"


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

        forced_3d = bool(str(os.environ.get("RASPTORCH_UI_3D_RENDER", "")).strip())
        if _running_on_raspberry_pi() and components is not None and not forced_3d:
            st.subheader("Structure (SVG)")
            svg = _model_structure_svg_html(cmds.models, selected)
            if svg:
                components.html(svg, height=560)
            else:
                st.caption("SVG structure diagram unavailable.")
        else:
            st.subheader("Structure (3D)")

            mode = _ui_3d_render_mode()
            rendered = False

            if mode == "pyvista_png":
                png = _model_pyvista_png(cmds.models, selected)
                if png is not None:
                    st.image(png, use_container_width=True)
                    rendered = True
                else:
                    st.caption("PyVista PNG render failed; falling back to Plotly.")

            if not rendered and mode == "pyvista":
                if pv is None:
                    st.caption("PyVista renderer requires `pyvista`.")
                else:
                    try:
                        from stpyvista import stpyvista as _stpyvista  # type: ignore
                    except Exception as e:
                        st.caption(
                            "PyVista interactive renderer requires a Streamlit-compatible `stpyvista`. "
                            "On this environment it failed to import; try upgrading `stpyvista` (and ensure Streamlit version compatibility)."
                        )
                        st.caption(f"stpyvista import error: {e}")
                    else:
                        plotter = _model_pyvista_plotter(cmds.models, selected, off_screen=False)
                        if plotter is None:
                            st.caption("PyVista renderer unavailable for this model.")
                        else:
                            _stpyvista(plotter, key=f"pyvista_{selected}")
                            rendered = True

            if not rendered:
                fig3d = _model_plotly_3d_figure(cmds.models, selected)
                if fig3d is not None:
                    st.plotly_chart(fig3d, use_container_width=True)
                    rendered = True
                else:
                    st.caption("3D structure requires Plotly (install `plotly`).")

            if rendered:
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

    dl_ext = str(save_type).lower()
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
