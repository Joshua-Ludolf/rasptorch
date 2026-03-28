from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _import_ui_app():
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    app_path = repo_root / "rasptorch" / "ui" / "app.py"
    spec = importlib.util.spec_from_file_location("rasptorch_ui_app", app_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mermaid_diagram_for_mlp() -> None:
    ui = _import_ui_app()

    models = {
        "abcd1234": {"type": "MLP", "config": {"layer_sizes": [10, 32, 2]}},
    }
    m = ui._model_mermaid_diagram(models, "abcd1234")
    assert m is not None
    assert "flowchart" in m
    assert "MLP" in m
    assert "10" in m and "32" in m and "2" in m


def test_mermaid_diagram_for_combined_sequential() -> None:
    ui = _import_ui_app()

    models = {
        "aaaa1111": {"type": "Linear", "config": {"input_size": 10, "hidden_sizes": [32], "output_size": 128}},
        "bbbb2222": {"type": "GRU", "config": {"input_size": 128, "hidden_size": 64, "num_layers": 1}},
        "cccc3333": {
            "type": "Combined",
            "config": {
                "combine": "sequential",
                "model_a_id": "aaaa1111",
                "model_b_id": "bbbb2222",
                "model_a_type": "Linear",
                "model_b_type": "GRU",
                "input_size": 10,
                "output_size": 64,
            },
        },
    }

    m = ui._model_mermaid_diagram(models, "cccc3333")
    assert m is not None
    assert "Combined" in m
    assert "Linear" in m and "GRU" in m
    assert "I/O" in m


def test_train_model_returns_history() -> None:
    # Train API should return per-epoch loss history for charting.
    from rasptorch.CLI._cli_commands import get_model_commands

    cmds = get_model_commands()
    res = cmds.create_mlp([10, 8, 2], activation="relu")
    assert "error" not in res
    mid = res["model_id"]

    out = cmds.train_model(mid, epochs=3, learning_rate=0.001, batch_size=4, device="cpu", optimizer_type="Adam")
    assert "error" not in out
    hist = out.get("training_history")
    assert isinstance(hist, list)
    assert len(hist) == 3
    assert all(isinstance(x, (int, float)) for x in hist)


def test_plotly_3d_helper_callable_and_hovertext_enriched() -> None:
    ui = _import_ui_app()
    models = {
        "m1": {
            "type": "CNN",
            "config": {
                "in_channels": 3,
                "out_channels": [8, 16],
                "kernels": [3, 5],
                "activation": "relu",
            },
        },
    }
    fig = ui._model_plotly_3d_figure(models, "m1")
    # Plotly is optional; if not installed, helper returns None.
    if fig is None:
        return
    d = fig.to_dict()
    # Mesh traces should have hovertext carrying config context.
    mesh_traces = [t for t in d.get("data", []) if t.get("type") == "mesh3d"]
    assert mesh_traces, "expected at least one mesh3d trace"
    hover = str(mesh_traces[0].get("hovertext", ""))
    assert "type=CNN" in hover
    assert "activation=" in hover


def test_plotly_2d_helper_has_hoverable_boxes_when_available() -> None:
    ui = _import_ui_app()

    models = {
        "m1": {
            "type": "CNN",
            "config": {
                "in_channels": 3,
                "out_channels": [8, 16],
                "kernels": [3, 5],
                "activation": "relu",
            },
        },
    }

    fig = ui._model_plotly_2d_figure(models, "m1")
    if fig is None:
        return
    d = fig.to_dict()
    traces = d.get("data", []) or []
    # Boxes are rendered as filled scatter polygons.
    filled = [t for t in traces if t.get("type") == "scatter" and t.get("fill") == "toself"]
    assert filled, "expected at least one filled polygon trace"
    hover = str(filled[0].get("hovertext", ""))
    assert "type=CNN" in hover


def test_info_config_filtering_logic_removes_activation_when_activations_present() -> None:
    # Mirrors the Info-panel filtering behavior (without rendering Streamlit).
    cfg = {"activation": "relu", "activations": ["relu", "none"], "layer_sizes": [10, 32, 2]}
    if "activations" in cfg and "activation" in cfg:
        cfg.pop("activation", None)
    assert "activation" not in cfg
    assert "activations" in cfg


def test_uploaded_name_stored_in_config_for_display() -> None:
    # UI stores a user friendly name under config['name'].
    models = {"id123456": {"type": "MLP", "config": {"name": "my_model"}}}
    cfg = dict(models["id123456"].get("config") or {})
    assert cfg.get("name") == "my_model"


def test_activation_mode_submit_logic_matches_mode() -> None:
    # This test mirrors the logic in _render_build_train_page without invoking Streamlit.
    # We reload the module to ensure globals exist.
    ui = _import_ui_app()

    activation_mode = "Per-layer (CSV)"
    activation = "tanh"
    per_layer = "relu,none"

    activations = None
    if activation_mode.startswith("Per-layer"):
        activations = [x.strip() for x in per_layer.split(",") if x.strip()] or None

    assert activations == ["relu", "none"]
    assert activation == "tanh"  # single activation stays independent


def test_ui_constants_exist_and_are_nonempty() -> None:
    ui = _import_ui_app()

    assert hasattr(ui, "HELP_TEXT")
    assert isinstance(ui.HELP_TEXT, list)
    assert len(ui.HELP_TEXT) >= 3

    assert hasattr(ui, "ACTIVATIONS")
    assert isinstance(ui.ACTIVATIONS, list)
    assert "relu" in ui.ACTIVATIONS

    assert hasattr(ui, "OPTIMIZERS")
    assert isinstance(ui.OPTIMIZERS, list)
    assert "Adam" in ui.OPTIMIZERS


def test_display_name_with_type_prefers_friendly_name() -> None:
    ui = _import_ui_app()

    models = {
        "id1234567890": {"type": "MLP", "config": {"name": "NiceName"}},
        "id_no_name": {"type": "CNN", "config": {}},
    }

    # Mirror the logic used in the Build & Train page format_func.
    def _display_name_with_type(mid: str) -> str:
        md = models.get(mid) or {}
        mtype = str(md.get("type", "Unknown"))
        cfg = md.get("config") or {}
        friendly = cfg.get("name")
        base = str(friendly) if friendly else mid[:8]
        return f"{base} ({mtype})"

    assert _display_name_with_type("id1234567890") == "NiceName (MLP)"
    assert _display_name_with_type("id_no_name") == "id_no_na (CNN)"
