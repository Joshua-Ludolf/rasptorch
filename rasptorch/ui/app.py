from __future__ import annotations as annotate
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import csv
import io
import itertools
import random
from urllib.parse import quote

import json
import sys
import tempfile
import hashlib
import os
import streamlit as st
import numpy as np

try:
    from PIL import Image
except Exception:
    Image = None  # type: ignore[assignment]

_UI_BUILD = "2026-04-10-layer-info-v2"


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
validate_dataset_structure: Any = None

try:
    import rasptorch as rasptorch  # type: ignore[no-redef]
    from rasptorch._cli_commands import get_model_commands as get_model_commands  # type: ignore[no-redef]
    from rasptorch.data_validation import validate_dataset_structure as validate_dataset_structure  # type: ignore[no-redef]

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

HELP_TEXT_FALLBACK = [
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
    "  model transfer <backbone> <num_classes> [freeze=true|false] [fine_tune=true|false]",
    "  combine sequential <a_id> <b_id> [name]",
    "  train epochs|batch-size|lr <value>",
    "  train start                   Train the selected model",
    "  save <path>                   Save selected model (backend-dependent)",
]


TENSORCOMMANDS: Any = None
if _HAS_RASPTORCH:
    try:
        from rasptorch._cli_commands import TensorCommands as TENSORCOMMANDS  # type: ignore
    except Exception:
        TENSORCOMMANDS = None


def _mermaid_escape(text: str) -> str:
    return str(text).replace("\n", " ").replace("\"", "'")


def _chat_repl_help_lines() -> List[str]:
    try:
        from rasptorch.CLI._cli_chat import ChatREPL

        help_text = ChatREPL.get_help(ChatREPL())
        lines = [line.rstrip() for line in str(help_text).splitlines() if line.strip()]
        return lines if lines else list(HELP_TEXT_FALLBACK)
    except Exception:
        return [f'Error loading help: {str(Exception)}'] + list(HELP_TEXT_FALLBACK)


HELP_TEXT = _chat_repl_help_lines()


def _combined_leaf_models(models: Dict[str, Any], model_id: str, seen: Optional[set[str]] = None) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Return non-combined source models for a potentially nested Combined model."""
    if seen is None:
        seen = set()
    if model_id in seen:
        return []
    seen.add(model_id)

    md = models.get(model_id) or {}
    mtype = str(md.get("type", "Unknown"))
    cfg = md.get("config") or {}
    if mtype == "Combined" and str(cfg.get("combine")) == "sequential":
        out: List[Tuple[str, str, Dict[str, Any]]] = []
        a_id = str(cfg.get("model_a_id", ""))
        b_id = str(cfg.get("model_b_id", ""))
        if a_id:
            out.extend(_combined_leaf_models(models, a_id, seen))
        if b_id:
            out.extend(_combined_leaf_models(models, b_id, seen))
        if out:
            return out
    return [(model_id, mtype, cfg)]


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
        node(root, f"Combined {model_id[:8]}")
        leaf_models = _combined_leaf_models(models, model_id)
        for i, (leaf_id, leaf_type, _leaf_cfg) in enumerate(leaf_models):
            src_node = f"{root}_src{i}"
            node(src_node, f"{leaf_type} {leaf_id[:8]}")
            edge(src_node, root, f"S{i + 1}")
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


def _model_structure_svg_html(models: Dict[str, Any], model_id: Optional[str]) -> Optional[str]:
    """Interactive isometric 3D schematic — pure SVG+JS, no WebGL."""
    if not model_id or model_id not in models:
        return None

    md = models.get(model_id) or {}
    mtype = str(md.get("type", "Unknown"))
    cfg = md.get("config") or {}

    def _esc(s: Any) -> str:
        t = str(s)
        return (
            t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;").replace("'", "&#39;")
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

    def _infer_io_sizes(model_type: str, model_cfg: Dict[str, Any]) -> Tuple[Optional[Any], Optional[Any]]:
        in_size = model_cfg.get("input_size")
        if in_size is None:
            in_size = model_cfg.get("in_channels")
        if in_size is None:
            layers = model_cfg.get("layer_sizes")
            if isinstance(layers, list) and layers:
                in_size = layers[0]
        if in_size is None and str(model_type) == "Transformer":
            in_size = model_cfg.get("d_model")

        out_size = model_cfg.get("output_size")
        if out_size is None:
            if str(model_type) == "GRU":
                out_size = model_cfg.get("hidden_size")
            elif str(model_type) == "CNN":
                out_ch = model_cfg.get("out_channels")
                if isinstance(out_ch, list) and out_ch:
                    out_size = out_ch[-1]
            elif str(model_type) == "MLP":
                layers = model_cfg.get("layer_sizes")
                if isinstance(layers, list) and layers:
                    out_size = layers[-1]
            elif str(model_type) == "Linear":
                hidden = model_cfg.get("hidden_sizes")
                if isinstance(hidden, list) and hidden:
                    out_size = hidden[-1]
            elif str(model_type) == "Transformer":
                out_size = model_cfg.get("d_model")
        return in_size, out_size

    def _model_summary() -> str:
        if mtype == "Combined" and str(cfg.get("combine")) == "sequential":
            leaf_models = _combined_leaf_models(models, str(model_id))
            leaf_types = ",".join(f"{t}:{mid[:8]}" for mid, t, _ in leaf_models)
            io_in = cfg.get("input_size")
            io_out = cfg.get("output_size")
            return (
                f"model_id={str(model_id)[:8]}\n"
                f"type=Combined(sequential)\n"
                f"source_count={len(leaf_models)}\n"
                f"source_types={leaf_types or '?'}\n"
                f"input_size={io_in}\n"
                f"output_size={io_out}"
            )
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
        return f"model_id={str(model_id)[:8]}\ntype={mtype}"

    # ── collect layer specs ───────────────────────────────────────────────────
    sizes: List[float] = []
    labels: List[str] = []
    hovers: List[str] = []

    def _add(sz: Any, label: str, hover: str) -> None:
        sizes.append(_coerce_size(sz))
        labels.append(label)
        hovers.append(hover)

    act_s = _act_summary()
    combined_seq = mtype == "Combined" and str(cfg.get("combine")) == "sequential"

    if combined_seq:
        leaf_models = _combined_leaf_models(models, str(model_id))
        io_in = cfg.get("input_size")
        io_out = cfg.get("output_size")
        leaf_types = ",".join(f"{t}:{mid[:8]}" for mid, t, _ in leaf_models)
        left_sizes  = [_coerce_size(io_out or io_in or 8)]
        left_labels = ["Combined"]
        left_hovers = [
            f"model_id={str(model_id)[:8]}\n"
            f"type=Combined(sequential)\n"
            f"source_count={len(leaf_models)}\n"
            f"source_types={leaf_types or '?'}\n"
            f"input_size={io_in}\n"
            f"output_size={io_out}"
        ]
        right_sizes = []
        right_labels = []
        right_hovers = []
        for i, (leaf_id, leaf_type, leaf_cfg) in enumerate(leaf_models):
            leaf_in_size, leaf_out_size = _infer_io_sizes(str(leaf_type), leaf_cfg)
            leaf_size = leaf_cfg.get("output_size")
            if leaf_size is None:
                leaf_size = leaf_cfg.get("input_size")
            if leaf_size is None:
                leaf_size = leaf_cfg.get("in_channels")
            if leaf_size is None:
                leaf_size = leaf_out_size
            if leaf_size is None:
                leaf_size = leaf_in_size
            right_sizes.append(_coerce_size(leaf_size or 8))
            right_labels.append(f"S{i + 1} {leaf_type}")
            leaf_name = str(leaf_cfg.get("name", "")).strip()
            leaf_label = leaf_name if leaf_name else leaf_id[:8]
            right_hovers.append(
                f"model_id={leaf_id[:8]}\n"
                f"type={leaf_type}\n"
                f"name={leaf_label}\n"
                f"source={i + 1}/{len(leaf_models)}\n"
                f"input_size={leaf_in_size}\n"
                f"output_size={leaf_out_size}"
            )
    else:
        if mtype == "MLP":
            for i, s in enumerate(cfg.get("layer_sizes") or []):
                _add(s, f"L{i}: {s}", f"type=MLP\nlayer={i}\nsize={s}" + (f"\n{act_s}" if act_s else ""))
        elif mtype == "Linear":
            inp = cfg.get("input_size")
            _add(inp or 1, f"in: {inp}", f"type=Linear\ninput_size={inp}" + (f"\n{act_s}" if act_s else ""))
            for i, h in enumerate(cfg.get("hidden_sizes") or []):
                _add(h, f"h{i}: {h}", f"type=Linear\nhidden[{i}]={h}" + (f"\n{act_s}" if act_s else ""))
            out = cfg.get("output_size")
            _add(out or 1, f"out: {out}", f"type=Linear\noutput_size={out}" + (f"\n{act_s}" if act_s else ""))
        elif mtype == "CNN":
            kernels = cfg.get("kernels")
            in_ch = cfg.get("in_channels")
            _add(in_ch or 1, f"in_ch: {in_ch}", f"type=CNN\nin_channels={in_ch}" + (f"\n{act_s}" if act_s else ""))
            for i, ch in enumerate(cfg.get("out_channels") or []):
                k = kernels[i] if isinstance(kernels, list) and i < len(kernels) else None
                lab = f"c{i}: {ch}" + (f" (k={k})" if k is not None else "")
                hov = f"type=CNN\nout_channels[{i}]={ch}" + (f"\nkernel={k}" if k is not None else "") + (f"\n{act_s}" if act_s else "")
                _add(ch, lab, hov)
        else:
            io_in = cfg.get("input_size")
            _add(io_in or 8, mtype, f"type={mtype}\ninput_size={io_in}")
        left_sizes, left_labels, left_hovers = sizes, labels, hovers
        right_sizes, right_labels, right_hovers = [], [], []

    if not left_sizes:
        return None

    # ── isometric geometry constants ──────────────────────────────────────────
    ISO_DX   = 24.0    # depth x-offset  (right)
    ISO_DY   = -14.0   # depth y-offset  (up)
    SLAB_H   = 72.0    # constant slab face height
    GAP      = 48.0    # gap between slabs
    MIN_W    = 80.0
    MAX_W    = 240.0

    max_sz = max(left_sizes + right_sizes) if right_sizes else max(left_sizes)

    def _w(sz: float) -> float:
        return MIN_W + (MAX_W - MIN_W) * (sz / max_sz)

    def _poly(*pts: Tuple[float, float]) -> str:
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)

    # Accent palettes: (front_top, front_bot, top_face, right_face)
    PALETTES = [
        ("#5b8ce8", "#3a6bcf", "#7aabf5", "#1e3f8a"),   # blue
        ("#9b6de8", "#7248cf", "#b98af5", "#401e8a"),   # purple
        ("#3fb8b0", "#288a83", "#5ed4cc", "#145854"),   # teal
        ("#e87a3f", "#cf5a28", "#f5a06e", "#8a3214"),   # amber
        ("#5bbf5b", "#3a9a3a", "#7ad47a", "#1e6018"),   # green
    ]

    grad_defs: List[str] = []
    grad_seen: set = set()
    shapes_svg: List[str] = []
    conns_svg: List[str] = []

    def _ensure_palette(idx: int) -> int:
        pid = idx % len(PALETTES)
        if pid not in grad_seen:
            grad_seen.add(pid)
            fa, fb, tc, rc = PALETTES[pid]
            grad_defs.append(f"""
  <linearGradient id="gf{pid}" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%"   stop-color="{fa}" stop-opacity="0.95"/>
    <stop offset="100%" stop-color="{fb}" stop-opacity="0.88"/>
  </linearGradient>
  <linearGradient id="gt{pid}" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0%"   stop-color="{tc}" stop-opacity="0.80"/>
    <stop offset="100%" stop-color="{tc}" stop-opacity="0.55"/>
  </linearGradient>
  <linearGradient id="gr{pid}" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%"   stop-color="{rc}" stop-opacity="0.85"/>
    <stop offset="100%" stop-color="{rc}" stop-opacity="0.60"/>
  </linearGradient>""")
        return pid

    def _draw_slab(cx: float, y0: float, w: float, label: str, hover: str, idx: int) -> Tuple[float, float]:
        """Draw one isometric slab. Returns (cx, y_bottom_of_front_face)."""
        pid = _ensure_palette(idx)
        x0, x1 = cx - w / 2, cx + w / 2
        y1 = y0 + SLAB_H
        cy = (y0 + y1) / 2

        top_pts   = _poly((x0, y0), (x1, y0), (x1+ISO_DX, y0+ISO_DY), (x0+ISO_DX, y0+ISO_DY))
        right_pts = _poly((x1, y0), (x1+ISO_DX, y0+ISO_DY), (x1+ISO_DX, y1+ISO_DY), (x1, y1))

        shapes_svg.append(
            f'<g class="rt-box" tabindex="0">\n'
            f'  <title>{_esc(hover)}</title>\n'
            f'  <polygon class="face-r" points="{right_pts}" fill="url(#gr{pid})"/>\n'
            f'  <polygon class="face-t" points="{top_pts}"   fill="url(#gt{pid})"/>\n'
            f'  <rect    class="face-f" x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{SLAB_H:.1f}" rx="3" fill="url(#gf{pid})"/>\n'
            f'  <rect    class="face-g" x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{SLAB_H:.1f}" rx="3"/>\n'
            f'  <text    class="slab-lbl" x="{cx:.1f}" y="{cy:.1f}">{_esc(label)}</text>\n'
            f'</g>'
        )
        return cx, y1

    def _draw_stack(cx: float, szs: List[float], labs: List[str], hvs: List[str], idx_off: int = 0) -> List[Tuple[float, float]]:
        centers: List[Tuple[float, float]] = []
        y = 0.0
        prev_bot: Optional[float] = None
        for i, (sz, lab, hov) in enumerate(zip(szs, labs, hvs)):
            w = _w(sz)
            box_cx, bot = _draw_slab(cx, y, w, lab, hov, i + idx_off)
            mid_y = (y + bot) / 2
            centers.append((box_cx, mid_y))
            if prev_bot is not None:
                conns_svg.append(
                    f'<line class="fc" x1="{cx:.1f}" y1="{prev_bot:.1f}" '
                    f'x2="{cx:.1f}" y2="{y:.1f}" marker-end="url(#ah)"/>'
                )
            prev_bot = bot
            y = bot + GAP
        return centers

    STACK_SEP = MAX_W + ISO_DX + 80.0

    if combined_seq:
        lcx = -STACK_SEP / 2
        rcx =  STACK_SEP / 2
        left_centers  = _draw_stack(lcx, left_sizes,  left_labels,  left_hovers,  idx_off=0)
        right_centers = _draw_stack(rcx, right_sizes, right_labels, right_hovers, idx_off=len(left_sizes))
        # Draw arrows from all source models to Combined.
        if right_centers and left_centers:
            cx, cy = left_centers[-1]
            for bx, by in right_centers:
                conns_svg.append(
                    f'<line class="fc cross" x1="{bx:.1f}" y1="{by:.1f}" '
                    f'x2="{cx:.1f}" y2="{cy:.1f}" marker-end="url(#ah)"/>'
                )
    else:
        left_centers  = _draw_stack(0.0, left_sizes, left_labels, left_hovers)
        right_centers = []

    # ── viewBox ───────────────────────────────────────────────────────────────
    all_cy = [cy for _, cy in left_centers + right_centers]
    content_h = (max(all_cy) + SLAB_H + GAP) if all_cy else 480.0
    content_h = max(content_h, 440.0)
    content_w = (STACK_SEP + MAX_W + ISO_DX + 40) if combined_seq else (MAX_W + ISO_DX + 40)
    PAD = 56.0
    vb_x = -content_w / 2 - PAD
    vb_y =  ISO_DY - PAD
    vb_w =  content_w + 2 * PAD
    vb_h =  content_h + 2 * PAD

    svg_id       = f"rt-iso-{_esc(model_id[:12])}"
    info_js      = json.dumps(_model_summary())
    defs_html    = "\n".join(grad_defs)
    shapes_html  = "\n".join(shapes_svg)
    conns_html   = "\n".join(conns_svg)

    # HUD / reset positions
    hud_x  = vb_x + 10
    hud_y  = vb_y + 10
    hud_w  = vb_w - 20
    rst_x  = vb_x + vb_w - 82
    rst_y  = vb_y + 10

    return f"""<div style="background:#13151c;border-radius:10px;padding:4px;box-shadow:0 6px 40px #0009;">
<svg id="{svg_id}" viewBox="{vb_x:.1f} {vb_y:.1f} {vb_w:.1f} {vb_h:.1f}"
     width="100%" height="560" style="touch-action:none;display:block;font-family:monospace;">
<defs>
{defs_html}
  <filter id="glow"  x="-30%" y="-30%" width="160%" height="160%">
    <feGaussianBlur stdDeviation="5" result="b"/>
    <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
  <filter id="glow2" x="-40%" y="-40%" width="180%" height="180%">
    <feGaussianBlur stdDeviation="10" result="b"/>
    <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
  <marker id="ah" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto">
    <polygon points="0 0,7 2.5,0 5" fill="rgba(255,255,255,0.9)"/>
  </marker>
  <pattern id="dots" width="32" height="32" patternUnits="userSpaceOnUse">
    <circle cx="1" cy="1" r="1" fill="rgba(255,255,255,0.045)"/>
  </pattern>
</defs>
<style>
  .bg   {{ fill:#13151c; }}
  .dots {{ fill:url(#dots); }}
  /* isometric faces */
  .face-r,.face-t {{ transition:opacity .15s; }}
  .face-r {{ opacity:.78; }}
  .face-t {{ opacity:.82; }}
  /* front glow overlay */
  .face-g {{
    fill:transparent;
    stroke:rgba(255,255,255,.13);
    stroke-width:1.5;
    pointer-events:none;
    transition:stroke .18s, filter .18s;
  }}
  /* box states */
  .rt-box {{ cursor:pointer; }}
  .rt-box:hover .face-r,
  .rt-box:hover .face-t,
  .rt-box:focus .face-r,
  .rt-box:focus .face-t {{ opacity:1; }}
  .rt-box:hover .face-g,
  .rt-box:focus .face-g {{
    stroke:rgba(255,255,255,.55);
    stroke-width:2.5;
    filter:url(#glow);
  }}
  .rt-box.sel .face-g {{
    stroke:rgba(140,210,255,.95);
    stroke-width:3;
    filter:url(#glow2);
  }}
  .rt-box.sel .face-r,
  .rt-box.sel .face-t {{ opacity:1; }}
  /* label */
  .slab-lbl {{
    fill:#fff;
    font-size:14px;
    font-weight:700;
    dominant-baseline:middle;
    text-anchor:middle;
    user-select:none;
    pointer-events:none;
    paint-order:stroke;
    stroke:rgba(0,0,0,.65);
    stroke-width:3px;
    stroke-linejoin:round;
  }}
  /* connectors */
  .fc {{
    stroke:rgba(150,175,220,.65);
    stroke-width:2;
    stroke-dasharray:6 4;
    fill:none;
    animation:dash 1.1s linear infinite;
  }}
  .cross {{
    stroke: rgba(255,255,255,0.9);  /* white */
    stroke-dasharray: 8 5;
  }}
  @keyframes dash {{ to {{ stroke-dashoffset:-20; }} }}
  .flow-lbl {{
    fill:rgba(200,165,255,.9);
    font-size:12px;
    text-anchor:middle;
    user-select:none;
    pointer-events:none;
  }}
  /* HUD */
    .hud-bg  {{ fill:rgba(14,18,30,.95); }}
    .hud-bd  {{ fill:none; stroke:rgba(125,152,210,.45); stroke-width:1.2; }}
    .hud-ttl {{ fill:rgba(174,200,255,.95); font-size:10px; font-weight:700; letter-spacing:.14em; }}
    .hud-sub {{ fill:rgba(130,153,201,.9); font-size:9px; letter-spacing:.09em; }}
    .hud-val {{ fill:rgba(236,242,255,.96); font-size:12px; }}
        .hud-grip-hit {{ fill:rgba(0,0,0,0); cursor:grab; }}
        .hud-grip-dot {{ fill:rgba(174,200,255,.95); opacity:.9; pointer-events:none; }}
        .hud-grip-line {{ stroke:rgba(130,153,201,.9); stroke-width:1.1; pointer-events:none; }}
  /* reset btn */
  .rbg {{ fill:rgba(50,60,95,.75); stroke:rgba(110,135,200,.4); stroke-width:1; cursor:pointer; }}
  .rbg:hover {{ fill:rgba(75,90,145,.9); }}
  .rlbl {{ fill:rgba(170,195,240,.9); font-size:10px; text-anchor:middle; dominant-baseline:middle; pointer-events:none; user-select:none; }}
</style>

<rect class="bg"   x="{vb_x:.1f}" y="{vb_y:.1f}" width="{vb_w:.1f}" height="{vb_h:.1f}"/>
<rect class="dots" x="{vb_x:.1f}" y="{vb_y:.1f}" width="{vb_w:.1f}" height="{vb_h:.1f}"/>

<g id="rt-vp">
  {conns_html}
  {shapes_html}
</g>

<!-- HUD -->
<g id="rt-hud">
    <rect class="hud-bg" x="{hud_x:.1f}" y="{hud_y:.1f}" width="{hud_w:.1f}" height="156" rx="10"/>
    <rect class="hud-bd" x="{hud_x:.1f}" y="{hud_y:.1f}" width="{hud_w:.1f}" height="156" rx="10"/>
        <g id="rt-hud-grip">
            <rect class="hud-grip-hit" x="{hud_x + hud_w/2 - 12:.1f}" y="{hud_y+12:.1f}" width="24" height="24" rx="4"/>
            <line class="hud-grip-line" x1="{hud_x + hud_w/2 - 6:.1f}" y1="{hud_y+18:.1f}" x2="{hud_x + hud_w/2 + 6:.1f}" y2="{hud_y+18:.1f}"/>
            <line class="hud-grip-line" x1="{hud_x + hud_w/2 - 6:.1f}" y1="{hud_y+24:.1f}" x2="{hud_x + hud_w/2 + 6:.1f}" y2="{hud_y+24:.1f}"/>
            <line class="hud-grip-line" x1="{hud_x + hud_w/2 - 6:.1f}" y1="{hud_y+30:.1f}" x2="{hud_x + hud_w/2 + 6:.1f}" y2="{hud_y+30:.1f}"/>
            <circle class="hud-grip-dot" cx="{hud_x + hud_w/2 - 8:.1f}" cy="{hud_y+18:.1f}" r="1.1"/>
            <circle class="hud-grip-dot" cx="{hud_x + hud_w/2 - 8:.1f}" cy="{hud_y+24:.1f}" r="1.1"/>
            <circle class="hud-grip-dot" cx="{hud_x + hud_w/2 - 8:.1f}" cy="{hud_y+30:.1f}" r="1.1"/>
            <circle class="hud-grip-dot" cx="{hud_x + hud_w/2 + 8:.1f}" cy="{hud_y+18:.1f}" r="1.1"/>
            <circle class="hud-grip-dot" cx="{hud_x + hud_w/2 + 8:.1f}" cy="{hud_y+24:.1f}" r="1.1"/>
            <circle class="hud-grip-dot" cx="{hud_x + hud_w/2 + 8:.1f}" cy="{hud_y+30:.1f}" r="1.1"/>
        </g>
    <text class="hud-ttl" x="{hud_x+12:.1f}" y="{hud_y+20:.1f}">LAYER INSPECTOR</text>
    <text class="hud-sub" x="{hud_x+12:.1f}" y="{hud_y+36:.1f}">HOVER OR CLICK A BLOCK FOR DETAILS</text>
    <text id="rt-inf" class="hud-val" x="{hud_x+12:.1f}" y="{hud_y+54:.1f}"></text>

        <!-- Reset button (inside inspector) -->
        <g id="rt-rst" style="cursor:pointer;">
            <rect class="rbg" x="{rst_x:.1f}" y="{rst_y:.1f}" width="70" height="20" rx="5"/>
            <text class="rlbl" x="{rst_x+35:.1f}" y="{rst_y+10:.1f}">RESET VIEW</text>
        </g>
</g>

<script>
(function(){{
  const svg = document.getElementById('{svg_id}');
  if (!svg) return;
  const vp  = svg.getElementById ? svg.getElementById('rt-vp') : svg.querySelector('#rt-vp');
  const inf = svg.querySelector('#rt-inf');
    const hud = svg.querySelector('#rt-hud');
    const modelInfo = String({info_js});

  function setInfo(raw) {{
    if (!inf) return;
        const rows = String(raw || '')
            .split('\\n')
            .map(s => s.trim())
            .filter(Boolean);
    while (inf.firstChild) inf.removeChild(inf.firstChild);
        rows.slice(0, 8).forEach((ln, i)=>{{
            const p = ln.split('=');
            const key = p.length > 1 ? p[0] : '';
            const val = p.length > 1 ? p.slice(1).join('=') : ln;
      const ts = document.createElementNS('http://www.w3.org/2000/svg','tspan');
      ts.setAttribute('x', inf.getAttribute('x'));
            ts.setAttribute('dy', i===0 ? '0' : '1.12em');
            ts.textContent = key ? `${{key}}: ${{val}}` : val;
      inf.appendChild(ts);
    }});
  }}
    setInfo(modelInfo);

  let sel = null;
  svg.querySelectorAll('.rt-box').forEach(g => {{
    const tt = g.querySelector('title');
    const txt = () => tt ? tt.textContent : '';
    g.addEventListener('pointerenter', ()=> setInfo(txt()));
    g.addEventListener('focus',        ()=> setInfo(txt()));
    g.addEventListener('click', e => {{
      e.stopPropagation();
      if (sel && sel !== g) sel.classList.remove('sel');
      sel = g; g.classList.add('sel');
      setInfo(txt());
    }});
  }});
  svg.addEventListener('click', ()=>{{ if(sel){{ sel.classList.remove('sel'); sel=null; }} }});

    // draggable HUD
    let hudDrag = false, hdx = 0, hdy = 0, hlx = 0, hly = 0;
    const applyHud = () => hud && hud.setAttribute('transform', `translate(${{hdx}} ${{hdy}})`);
    if (hud) {{
        hud.addEventListener('click', e => e.stopPropagation());
    }}

  // pan / zoom
    let sc=1, tx=0, ty=0, pan=false, lx=0, ly=0;
  const apply = () => vp && vp.setAttribute('transform',`translate(${{tx}} ${{ty}}) scale(${{sc}})`);

  svg.addEventListener('pointerdown', e => {{
    const t = e.target;
        if (t.closest && t.closest('#rt-hud-grip')) {{
            e.preventDefault();
            hudDrag = true;
            hlx = e.clientX;
            hly = e.clientY;
            try {{ svg.style.userSelect = 'none'; svg.style.webkitUserSelect = 'none'; }} catch(_) {{}}
            svg.setPointerCapture(e.pointerId);
            return;
        }}
        if (t.closest && t.closest('#rt-hud')) return;
        if (t.closest && (t.closest('.rt-box')||t.closest('#rt-rst'))) return;
    pan=true; lx=e.clientX; ly=e.clientY; svg.setPointerCapture(e.pointerId);
  }});
  svg.addEventListener('pointerup', e => {{
      pan=false; hudDrag=false;
      try {{ svg.style.userSelect = ''; svg.style.webkitUserSelect = ''; }} catch(_) {{}}
      try{{svg.releasePointerCapture(e.pointerId);}}catch(_){{}}
  }});
  svg.addEventListener('pointermove', e => {{
        if (hudDrag) {{
            hdx += e.clientX - hlx;
            hdy += e.clientY - hly;
            hlx = e.clientX;
            hly = e.clientY;
            applyHud();
            return;
        }}
    if(!pan) return;
    tx+=e.clientX-lx; ty+=e.clientY-ly; lx=e.clientX; ly=e.clientY; apply();
  }});
  svg.addEventListener('wheel', e => {{
    e.preventDefault();
    const r=svg.getBoundingClientRect();
    const mx=e.clientX-r.left, my=e.clientY-r.top;
    const z=e.deltaY<0?1.13:1/1.13;
    const ns=Math.min(5,Math.max(0.2,sc*z)), k=ns/sc;
    tx=mx-k*(mx-tx); ty=my-k*(my-ty); sc=ns; apply();
  }},{{passive:false}});

  const rst = svg.querySelector('#rt-rst');
  if(rst) rst.addEventListener('click',()=>{{sc=1;tx=0;ty=0;apply();}});

    applyHud();
  apply();
}})();
</script>
</svg></div>"""


def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("nav", "Models")
    ss.setdefault("repl_log", [])
    ss.setdefault("app_log", [])
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
            "preprocessing_pipeline": {
                "normalize": "none",
                "resize_hw": None,
            },
            # Models that have successfully completed at least one training run in this UI session.
            "trained_models": [],
            "search_runs": [],
        },
    )


def _append_log(lines: List[str] | str) -> None:
    if isinstance(lines, str):
        st.session_state.repl_log.append(lines)
    else:
        st.session_state.repl_log.extend(lines)


def _log_event(level: str, message: str, **extra: Any) -> None:
    logs = st.session_state.setdefault("app_log", [])
    logs.append(
        {
            "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "level": str(level).upper(),
            "message": str(message),
            "extra": extra or {},
        }
    )
    if len(logs) > 1000:
        st.session_state["app_log"] = logs[-1000:]


def _render_log_viewer(limit: int = 300) -> None:
    st.subheader("Logs")
    logs = st.session_state.get("app_log", [])
    if not isinstance(logs, list) or not logs:
        st.info("No log entries yet.")
        return

    rows = []
    for entry in logs[-int(limit):]:
        if not isinstance(entry, dict):
            continue
        rows.append(
            {
                "time": str(entry.get("ts", "")),
                "level": str(entry.get("level", "INFO")),
                "message": str(entry.get("message", "")),
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)


def _first_linear_weight(model: Any) -> Optional[np.ndarray]:
    try:
        layers = getattr(model, "layers", None)
        if isinstance(layers, list):
            for layer in layers:
                w = _first_linear_weight(layer)
                if w is not None:
                    return w
        if hasattr(rasptorch.nn, "Linear") and isinstance(model, rasptorch.nn.Linear):
            w = getattr(model, "weight", None)
            if w is None:
                return None
            arr = np.asarray(getattr(w, "data", w), dtype=np.float32)
            if arr.ndim == 2:
                return arr
    except Exception:
        return None
    return None


def _infer_model_input_size(model_data: Dict[str, Any]) -> Optional[int]:
    cfg = model_data.get("config") or {}
    try:
        if cfg.get("input_size") is not None:
            return int(cfg.get("input_size"))
        if cfg.get("feature_size") is not None:
            return int(cfg.get("feature_size"))
        if cfg.get("vocab_size") is not None:
            return int(cfg.get("vocab_size"))
        layers = cfg.get("layer_sizes")
        if isinstance(layers, list) and layers:
            return int(layers[0])
        if cfg.get("in_channels") is not None:
            return int(cfg.get("in_channels"))
    except Exception:
        return None
    return None


def _jet_colormap(v: np.ndarray) -> np.ndarray:
    x = np.clip(v, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def _gradcam_like_overlay(image_rgb: np.ndarray, model_data: Dict[str, Any], model_obj: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Create a Grad-CAM-style overlay from model weights and input image.

    This is a lightweight approximation suitable for rasptorch-native models.
    """
    img = np.asarray(image_rgb, dtype=np.float32)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Expected RGB image with shape [H,W,3]")
    img01 = np.clip(img / 255.0, 0.0, 1.0)

    h, w, c = img01.shape
    gray = img01.mean(axis=2)
    cam: Optional[np.ndarray] = None

    w0 = _first_linear_weight(model_obj)
    in_size = _infer_model_input_size(model_data)

    if w0 is not None and in_size is not None:
        imp = np.mean(np.abs(w0), axis=0)
        if int(in_size) == h * w * c and imp.size == h * w * c:
            cam = imp.reshape(h, w, c).mean(axis=2)
        elif int(in_size) == h * w and imp.size == h * w:
            cam = imp.reshape(h, w)

    if cam is None:
        gy, gx = np.gradient(gray)
        cam = np.abs(gx) + np.abs(gy)

    cam = np.asarray(cam, dtype=np.float32)
    cam = cam - float(cam.min())
    denom = float(cam.max()) if float(cam.max()) > 0.0 else 1.0
    cam = cam / denom

    heat = _jet_colormap(cam)
    overlay = np.clip(0.60 * img01 + 0.40 * heat, 0.0, 1.0)
    overlay_u8 = (overlay * 255.0).astype(np.uint8)

    info = {
        "xai_method": "gradcam_style_approx",
        "input_shape": [int(h), int(w), int(c)],
        "used_linear_weight": bool(w0 is not None),
    }
    return overlay_u8, info


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


def _strip_redundant_activation(obj: Any) -> Any:
    """Remove legacy/single `activation` keys when per-layer `activations` exist.

    This runs recursively so Combined-model snapshots don't show both.
    """
    try:
        if isinstance(obj, dict):
            has_activations = "activations" in obj
            out: Dict[Any, Any] = {}
            for k, v in obj.items():
                if k == "activation" and has_activations:
                    continue
                out[k] = _strip_redundant_activation(v)
            return out
        if isinstance(obj, list):
            return [_strip_redundant_activation(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_strip_redundant_activation(v) for v in obj)
    except Exception:
        return obj
    return obj


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

    _log_event("INFO", "REPL command", command=command_str)

    ctx = st.session_state.repl_context
    cmds = get_model_commands()
    parts = command_str.split()
    root = parts[0].lower()

    def out(s: List[str] | str) -> None:
        _append_log(s)
        txt = str(s)
        lvl = "ERROR" if txt.startswith("✗") else "INFO"
        _log_event(lvl, txt)

    def selected_model() -> Optional[str]:
        mid = ctx.get("current_model")
        if mid and mid in cmds.models:
            return mid
        return None

    try:
        if root == "help":
            out(_chat_repl_help_lines())
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
            if TENSORCOMMANDS is None:
                out("✗ Tensor commands are unavailable in this build")
                return
            sub = parts[1].lower()
            shape = tuple(int(x.strip()) for x in parts[2].split(",") if x.strip())
            device = str(ctx.get("device", "cpu"))
            if sub == "create":
                result = TENSORCOMMANDS.create_random(shape, device=device)
            elif sub == "zeros":
                result = TENSORCOMMANDS.create_zeros(shape, device=device)
            elif sub == "ones":
                result = TENSORCOMMANDS.create_ones(shape, device=device)
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

            if sub == "transfer":
                if len(parts) < 4:
                    out("✗ Usage: model transfer <backbone> <num_classes> [freeze=true|false] [fine_tune=true|false]")
                    return
                backbone = str(parts[2]).strip().lower()
                num_classes = int(parts[3])
                freeze_backbone = True
                fine_tune = False
                for tok in parts[4:]:
                    t = str(tok).strip().lower()
                    if t.startswith("freeze="):
                        freeze_backbone = t.split("=", 1)[1] in {"1", "true", "yes", "y", "on"}
                    elif t.startswith("fine_tune="):
                        fine_tune = t.split("=", 1)[1] in {"1", "true", "yes", "y", "on"}
                res = cmds.create_transfer_model(
                    backbone,
                    num_classes,
                    freeze_backbone=freeze_backbone,
                    fine_tune=fine_tune,
                )
                if "error" in res:
                    out(f"✗ Error: {res['error']}")
                    return
                mid = res.get("model_id")
                ctx["current_model"] = mid
                out(f"✓ Created Transfer: {str(mid)[:8]} ({backbone})")
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
        st.dataframe(rows, width="stretch", hide_index=True)
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

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        if st.button("Refresh"):
            st.rerun()
    with col_b:
        if st.button("Deselect"):
            ctx["current_model"] = None
            try:
                st.query_params.pop("model_id", None)
            except Exception:
                pass
            st.rerun()
    with col_c:
        if st.button("Delete selected", disabled=(selected is None)):
            if selected is not None:
                res = cmds.delete_model(selected)
                if "error" in res:
                    st.error(res["error"])
                else:
                    ctx["current_model"] = None
                    try:
                        st.query_params.pop("model_id", None)
                    except Exception:
                        pass
                    st.success("Deleted")
                    st.rerun()
    with col_d:
        if st.button("Delete all models", disabled=(not model_ids)):
            deleted_ids = list(cmds.models.keys())
            for mid in deleted_ids:
                cmds.delete_model(mid)
            ctx["current_model"] = None
            try:
                st.query_params.pop("model_id", None)
            except Exception:
                pass
            st.success(f"Deleted {len(deleted_ids)} model(s)")
            st.rerun()

    if selected is not None and selected in cmds.models:
        st.subheader("Info")
        md = cmds.models[selected]
        cfg = _strip_redundant_activation(dict(md.get("config", {}) or {}))
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

        st.subheader("Structure")
        svg = _model_structure_svg_html(cmds.models, selected)
        if svg:
            data_url = "data:text/html;charset=utf-8," + quote(svg)
            st.iframe(data_url, height=560, width="stretch")

        st.subheader("Explainability (XAI)")
        st.caption("Upload an image to run prediction and generate a Grad-CAM-style overlay.")
        xai_up = st.file_uploader("Input image", type=["png", "jpg", "jpeg"], key=f"xai_upload_{selected}")
        if xai_up is not None:
            if Image is None:
                st.error("Pillow is required for image-based XAI preview.")
            else:
                try:
                    img = Image.open(io.BytesIO(xai_up.getvalue())).convert("RGB")
                    img_np = np.asarray(img, dtype=np.uint8)
                    st.image(img_np, caption="Input", width="stretch")

                    if st.button("Run prediction + XAI", key=f"xai_btn_{selected}"):
                        md0 = cmds.models.get(selected) or {}
                        model_obj = None
                        try:
                            model_obj = cmds._ensure_model(selected)  # type: ignore[attr-defined]
                        except Exception:
                            model_obj = md0.get("model")
                        if model_obj is None:
                            st.error("Model reconstruction failed for XAI")
                            _log_event("ERROR", "XAI failed: model reconstruction", model_id=selected)
                        else:
                            overlay, xai_info = _gradcam_like_overlay(img_np, md0, model_obj)
                            st.image(overlay, caption="Grad-CAM-style overlay", width="stretch")

                            # Best-effort prediction.
                            pred_msg = "Prediction unavailable"
                            try:
                                in_size = _infer_model_input_size(md0)
                                x = np.asarray(img_np, dtype=np.float32).reshape(-1)
                                if in_size is not None and int(in_size) > 0:
                                    n = int(in_size)
                                    if x.size < n:
                                        x = np.pad(x, (0, n - x.size), mode="constant")
                                    elif x.size > n:
                                        x = x[:n]
                                x = x.reshape(1, -1)
                                t = rasptorch.Tensor(x.astype(np.float32), device=str(ctx.get("device", "cpu")))
                                out = model_obj(t)
                                out_np = out.numpy() if hasattr(out, "numpy") else np.asarray(out)
                                out_flat = np.asarray(out_np).reshape(1, -1)
                                pred_idx = int(np.argmax(out_flat, axis=1)[0])
                                pred_msg = f"Predicted class index: {pred_idx}"
                            except Exception as e:
                                pred_msg = f"Prediction unavailable ({e})"

                            st.info(pred_msg)
                            st.json(xai_info)
                            _log_event("INFO", "XAI generated", model_id=selected, prediction=pred_msg)
                except Exception as e:
                    st.error(f"XAI failed: {e}")
                    _log_event("ERROR", "XAI failed", model_id=selected, error=str(e))


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
                _log_event("ERROR", "Model save failed", model_id=selected, error=str(res.get("error")))
            else:
                try:
                    data = Path(tmp_out).read_bytes()
                    st.download_button(
                        "Download",
                        data=data,
                        file_name=save_name or f"model_{selected}{dl_ext}",
                        mime="application/octet-stream",
                    )
                    _log_event("INFO", "Model saved for download", model_id=selected, format=res.get("format", "?"))
                except Exception as e:
                    st.error(f"Failed preparing download: {e}")
                    _log_event("ERROR", "Failed preparing model download", model_id=selected, error=str(e))
                finally:
                    try:
                        os.remove(tmp_out)
                    except Exception:
                        pass

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
            _log_event("ERROR", "Model load failed", source=up.name, error=str(res.get("error")))
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
            _log_event("INFO", "Model loaded", model_id=ctx.get("current_model"), format=res.get("format", "?"))
            st.rerun()

        try:
            os.remove(tmp_path)
        except Exception:
            pass



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
        _log_event("ERROR", "Model save failed", model_id=selected, error=str(res.get("error")))
        return
    try:
        data = Path(tmp_out).read_bytes()
        st.download_button(
            "Download",
            data=data,
            file_name=save_name or f"model_{selected}{dl_ext}",
            mime="application/octet-stream",
        )
        _log_event("INFO", "Model saved for download", model_id=selected, format=res.get("format", "?"))
    except Exception as e:
        st.error(f"Failed preparing download: {e}")
        _log_event("ERROR", "Failed preparing model download", model_id=selected, error=str(e))
    finally:
        try:
            os.remove(tmp_out)
        except Exception:
            pass

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
    model_type = st.selectbox("Model type", options=["MLP", "Linear", "CNN", "GRU", "Transformer", "Transfer"], index=0)

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
    transfer_backbone: str = "resnet50"
    transfer_classes: int = 2
    transfer_freeze: bool = True
    transfer_finetune: bool = False

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
        elif model_type == "Transformer":
            vocab = int(st.number_input("Vocab size", min_value=2, value=1000, step=1))
            d_model = int(st.number_input("d_model", min_value=1, value=128, step=1))
            heads = int(st.number_input("Num heads", min_value=1, value=4, step=1))
            t_layers = int(st.number_input("Num layers", min_value=1, value=2, step=1))
        else:
            transfer_backbone = st.selectbox("Backbone", options=["resnet50", "vgg16", "mobilenet_v2"], index=0)
            transfer_classes = int(st.number_input("Num classes", min_value=2, value=2, step=1))
            col_tf1, col_tf2 = st.columns(2)
            with col_tf1:
                transfer_freeze = bool(st.checkbox("Freeze backbone", value=True))
            with col_tf2:
                transfer_finetune = bool(st.checkbox("Fine-tune backbone", value=False))

        submitted = st.form_submit_button("Create")

    if submitted:
        activations = None
        if activation_mode.startswith("Per-layer"):
            activations = [x.strip() for x in str(per_layer or "").split(",") if x.strip()] or None
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
        elif model_type == "Transformer":
            res = cmds.create_transformer(int(vocab), int(d_model), int(heads), int(t_layers))
        else:
            res = cmds.create_transfer_model(
                str(transfer_backbone),
                int(transfer_classes),
                freeze_backbone=bool(transfer_freeze),
                fine_tune=bool(transfer_finetune),
            )

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

    with st.expander("Dataset Validation & Preprocessing", expanded=False):
        st.caption("Validate CSV schema/labels and set preprocessing metadata used when saving models.")
        req_cols = st.text_input("Required columns (CSV)", value="", key="ds_required_cols")
        label_col = st.text_input("Label column (optional)", value="", key="ds_label_col")
        img_col = st.text_input("Image path column (optional)", value="", key="ds_img_col")
        csv_up = st.file_uploader("Dataset CSV (optional)", type=["csv"], key="ds_csv_upload")

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            normalize_mode = st.selectbox("Normalize", options=["none", "dataset"], index=0, key="preproc_normalize")
        with col_p2:
            resize_raw = st.text_input("Resize (H,W optional)", value="", key="preproc_resize")

        pipeline = {
            "normalize": normalize_mode,
            "resize_hw": None,
        }
        rr = str(resize_raw or "").strip()
        if rr:
            try:
                hw = [int(x.strip()) for x in rr.split(",") if x.strip()]
                if len(hw) == 2 and hw[0] > 0 and hw[1] > 0:
                    pipeline["resize_hw"] = [int(hw[0]), int(hw[1])]
            except Exception:
                st.warning("Resize must be CSV like '224,224'.")

        ctx["preprocessing_pipeline"] = pipeline

        if csv_up is not None and validate_dataset_structure is not None:
            try:
                raw_text = csv_up.getvalue().decode("utf-8", errors="replace")
                rows = list(csv.DictReader(io.StringIO(raw_text)))
                required = [x.strip() for x in str(req_cols).split(",") if x.strip()]
                report = validate_dataset_structure(
                    rows,
                    required_columns=required,
                    label_column=(label_col.strip() or None),
                    image_path_column=(img_col.strip() or None),
                    check_paths=True,
                )
                st.json(report)
                if report.get("ok"):
                    _log_event("INFO", "Dataset validation passed", rows=int(report.get("rows", 0)))
                else:
                    _log_event("WARNING", "Dataset validation reported issues", report=report)
            except Exception as e:
                st.error(f"Dataset validation failed: {e}")
                _log_event("ERROR", "Dataset validation failed", error=str(e))
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
        pbar = st.progress(0, text="Starting training...")
        status = st.empty()

        def _on_progress(ev: Dict[str, Any]) -> None:
            if not isinstance(ev, dict):
                return
            if ev.get("event") == "epoch":
                i = int(ev.get("epoch", 0))
                n = max(1, int(ev.get("epochs", 1)))
                loss_v = float(ev.get("loss", 0.0))
                frac = min(1.0, max(0.0, i / n))
                pbar.progress(frac, text=f"Epoch {i}/{n}  loss={loss_v:.6f}")
                status.caption(f"Elapsed this epoch: {float(ev.get('seconds', 0.0)):.3f}s")

        with st.spinner("Training..."):
            res = cmds.train_model(
                sel,
                epochs=int(epochs),
                learning_rate=float(lr),
                batch_size=int(batch),
                device=str(device),
                optimizer_type=str(opt),
                progress_callback=_on_progress,
            )
        pbar.progress(1.0, text="Training complete")
        if "error" in res:
            err = res.get("error")
            msg = str(err) if err is not None else ""
            if not msg.strip():
                msg = "Training failed (no error message returned)."
            st.error(msg)
            _log_event("ERROR", "Training failed", model_id=sel, error=msg)
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
                        "final_loss": float(res.get("final_loss", 0.0)),
                        "elapsed_seconds": float(res.get("elapsed_seconds", 0.0)),
                        "source": "manual",
                        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    }
                ])[-10:]
            st.success(f"Done. Final loss: {res.get('final_loss', 0.0):.6f}")
            _log_event(
                "INFO",
                "Training complete",
                model_id=sel,
                final_loss=float(res.get("final_loss", 0.0)),
                elapsed_seconds=float(res.get("elapsed_seconds", 0.0)),
                epochs=int(epochs),
            )
            try:
                trained = ctx.get("trained_models")
                if not isinstance(trained, list):
                    trained = []
                if sel not in trained:
                    trained.append(sel)
                ctx["trained_models"] = trained
                if sel in cmds.models:
                    cmds.models[sel]["preprocessing"] = dict(ctx.get("preprocessing_pipeline", {}))
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

    with st.expander("Hyperparameter Search", expanded=False):
        st.caption("Run grid/random search across optimizer, LR, batch size, and epochs.")
        search_mode = st.selectbox("Search mode", options=["grid", "random"], index=0, key="hp_mode")
        lr_csv = st.text_input("Learning rates (CSV)", value="0.001,0.0005", key="hp_lrs")
        batch_csv = st.text_input("Batch sizes (CSV)", value="16,32", key="hp_batches")
        epoch_csv = st.text_input("Epochs (CSV)", value="3,5", key="hp_epochs")
        opt_list = st.multiselect("Optimizers", options=OPTIMIZERS, default=[str(ctx.get("optimizer", "Adam"))], key="hp_opts")
        random_trials = int(st.number_input("Random trials", min_value=1, value=4, step=1, key="hp_trials"))

        def _parse_floats(txt: str, default: List[float]) -> List[float]:
            vals = []
            for part in str(txt).split(","):
                p = part.strip()
                if not p:
                    continue
                vals.append(float(p))
            return vals or default

        def _parse_ints(txt: str, default: List[int]) -> List[int]:
            vals = []
            for part in str(txt).split(","):
                p = part.strip()
                if not p:
                    continue
                vals.append(int(p))
            return vals or default

        if st.button("Run Hyperparameter Search", key="hp_run_btn"):
            try:
                lrs = _parse_floats(lr_csv, [float(ctx.get("learning_rate", 0.001))])
                batches = _parse_ints(batch_csv, [int(ctx.get("batch_size", 32))])
                epochs_list = _parse_ints(epoch_csv, [int(ctx.get("train_epochs", 5))])
                opts = [str(o) for o in (opt_list or [str(ctx.get("optimizer", "Adam"))])]
                combos = list(itertools.product(lrs, batches, epochs_list, opts))
                if not combos:
                    st.warning("No search combinations were generated.")
                else:
                    if search_mode == "random" and len(combos) > random_trials:
                        combos = random.sample(combos, int(random_trials))

                    prog = st.progress(0, text="Starting search...")
                    results: List[Dict[str, Any]] = []
                    for i, (lr0, b0, e0, o0) in enumerate(combos, start=1):
                        prog.progress(i / len(combos), text=f"Run {i}/{len(combos)}: {o0}, lr={lr0}, b={b0}, e={e0}")
                        r = cmds.train_model(
                            sel,
                            epochs=int(e0),
                            learning_rate=float(lr0),
                            batch_size=int(b0),
                            device=str(device),
                            optimizer_type=str(o0),
                        )
                        if "error" in r:
                            results.append(
                                {
                                    "model_id": sel,
                                    "optimizer": str(o0),
                                    "learning_rate": float(lr0),
                                    "batch_size": int(b0),
                                    "epochs": int(e0),
                                    "status": "error",
                                    "error": str(r.get("error")),
                                }
                            )
                            _log_event("ERROR", "Hyperparameter run failed", model_id=sel, error=str(r.get("error")))
                        else:
                            hist0 = r.get("training_history")
                            loss_series = [float(x) for x in hist0] if isinstance(hist0, list) else []
                            row = {
                                "model_id": sel,
                                "optimizer": str(o0),
                                "learning_rate": float(lr0),
                                "batch_size": int(b0),
                                "epochs": int(e0),
                                "status": "ok",
                                "final_loss": float(r.get("final_loss", 0.0)),
                                "elapsed_seconds": float(r.get("elapsed_seconds", 0.0)),
                            }
                            results.append(row)
                            ctx.setdefault("train_runs", []).append(
                                {
                                    "model_id": sel,
                                    "device": str(device),
                                    "optimizer": str(o0),
                                    "learning_rate": float(lr0),
                                    "batch_size": int(b0),
                                    "epochs": int(e0),
                                    "loss": loss_series,
                                    "final_loss": float(r.get("final_loss", 0.0)),
                                    "elapsed_seconds": float(r.get("elapsed_seconds", 0.0)),
                                    "source": "search",
                                    "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                                }
                            )
                    prog.progress(1.0, text="Search complete")
                    ctx["search_runs"] = (ctx.get("search_runs", []) + results)[-200:]
                    _log_event("INFO", "Hyperparameter search complete", runs=len(results), mode=search_mode)
                    st.success(f"Search complete: {len(results)} run(s)")
            except Exception as e:
                st.error(f"Hyperparameter search failed: {e}")
                _log_event("ERROR", "Hyperparameter search failed", model_id=sel, error=str(e))

        sr = ctx.get("search_runs", [])
        if isinstance(sr, list) and sr:
            st.caption("Search results (latest)")
            st.dataframe(sr[-50:], width="stretch", hide_index=True)
            ok_rows = [r for r in sr if isinstance(r, dict) and r.get("status") == "ok" and r.get("final_loss") is not None]
            if ok_rows:
                best = min(ok_rows, key=lambda r: float(r.get("final_loss", 1e12)))
                st.info(
                    "Best run: "
                    f"opt={best.get('optimizer')}  lr={best.get('learning_rate')}  "
                    f"batch={best.get('batch_size')}  epochs={best.get('epochs')}  "
                    f"final_loss={float(best.get('final_loss', 0.0)):.6f}"
                )


def _render_chat_page() -> None:
    st.header("Chat / REPL")
    st.caption("This is a web version of the rasptorch chat-style CLI. Type commands like `model list` or `train start`.")

    if st.button("Show help"):
        _append_log(_chat_repl_help_lines())

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


def _render_dashboard_page() -> None:
    st.header("Comparison Dashboard")
    ctx = st.session_state.repl_context

    runs = ctx.get("train_runs", [])
    if not isinstance(runs, list) or not runs:
        st.info("No training runs yet. Train a model first.")
        return

    rows: List[Dict[str, Any]] = []
    for i, r in enumerate(runs, start=1):
        if not isinstance(r, dict):
            continue
        loss_series = r.get("loss") or []
        final_loss = r.get("final_loss")
        if final_loss is None and isinstance(loss_series, list) and loss_series:
            try:
                final_loss = float(loss_series[-1])
            except Exception:
                final_loss = None
        rows.append(
            {
                "run": i,
                "timestamp": r.get("timestamp", ""),
                "model_id": str(r.get("model_id", ""))[:8],
                "source": str(r.get("source", "manual")),
                "optimizer": r.get("optimizer", ""),
                "learning_rate": r.get("learning_rate", ""),
                "batch_size": r.get("batch_size", ""),
                "epochs": r.get("epochs", ""),
                "final_loss": final_loss,
                "elapsed_seconds": r.get("elapsed_seconds", ""),
                "device": r.get("device", ""),
            }
        )

    st.subheader("Runs")
    st.dataframe(rows, width="stretch", hide_index=True)

    # Side-by-side loss curves.
    st.subheader("Loss Curves")
    max_runs = int(st.number_input("Curves to overlay", min_value=1, value=min(6, len(runs)), step=1))
    selected_runs = runs[-max_runs:]
    chart_data: Dict[str, List[float]] = {}
    max_len = 0
    for idx, run in enumerate(selected_runs, start=1):
        series = run.get("loss") if isinstance(run, dict) else None
        if not isinstance(series, list) or not series:
            continue
        label = f"run{idx}_{str(run.get('model_id', ''))[:8]}".replace(":", "_")
        vals = [float(x) for x in series]
        chart_data[label] = vals
        max_len = max(max_len, len(vals))

    if chart_data and max_len > 0:
        # Pad shorter curves with last value for a compact overlay chart.
        padded: Dict[str, List[float]] = {}
        for k, vals in chart_data.items():
            if len(vals) < max_len:
                vals = vals + [vals[-1]] * (max_len - len(vals))
            padded[k] = vals
        st.line_chart(padded)

    ok_rows = [r for r in rows if isinstance(r.get("final_loss"), (int, float))]
    if ok_rows:
        best = min(ok_rows, key=lambda x: float(x.get("final_loss", 1e12)))
        st.success(
            "Best final loss: "
            f"run={best.get('run')} model={best.get('model_id')} final_loss={float(best.get('final_loss', 0.0)):.6f}"
        )


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
        nav_options = ["Models", "Build & Train", "Dashboard", "Chat/REPL", "Logs"]
        current_nav = str(st.session_state.get("nav", "Models"))
        if current_nav not in nav_options:
            current_nav = "Models"
        nav = st.radio(
            "Page",
            options=nav_options,
            index=nav_options.index(current_nav),
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
                _log_event("ERROR", "Device switch failed", requested=str(device_choice), error=msg)
            else:
                st.success(msg)
                _log_event("INFO", "Device switched", device=str(ctx.get("device", "cpu")))

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
            st.session_state.app_log = []
            st.session_state.repl_context = {
                "current_model": None,
                "train_epochs": 5,
                "batch_size": 32,
                "learning_rate": 0.001,
                "optimizer": "Adam",
                "device": "cpu",
                "loaded_files": {},
                "upload_cache": {},
                "last_loaded_hash": None,
                "preprocessing_pipeline": {"normalize": "none", "resize_hw": None},
                "trained_models": [],
                "search_runs": [],
            }
            st.rerun()

    if nav == "Models":
        _render_models_page()
    elif nav == "Build & Train":
        _render_build_train_page()
    elif nav == "Dashboard":
        _render_dashboard_page()
    elif nav == "Logs":
        _render_log_viewer()
    else:
        _render_chat_page()


if __name__ == "__main__":
    main()
